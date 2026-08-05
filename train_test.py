import os
import sys

import torch

sys.path.append('GroundingDINO')
from groundingdino.util.base_api import threshold

from utils.evaluation import calc_loc_metric, prepare_targets
from utils.image_loader import get_loader
from utils.model_utils import (build_model_for_mode, cpu_state_dict,
                               freeze_encoders, load_checkpoint, load_pretrained,
                               resolve_device, set_finetuning_mode,
                               trainable_parameters)
from utils.processor import DataProcessor
from utils.criterion import SetCriterion

# Notebook-friendly controls. Keep tiny as the default for local data-flow debugging.
MODEL_MODE = "tiny"  # "tiny", "compact_rec", or "full"
DEVICE = "auto"
IMAGE_DIR = "F:/REC-8K/rec-8k"
ANNOTATIONS_PATH = "anno/annotations.json"
SPLITS_PATH = "anno/splits.json"
CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
TEXT_ENCODER = "F:/model/bert-base-uncased"  # replace with a local/cache path when needed
PRETRAINED_CHECKPOINT = "F:/GroundingDINO/groundingdino_swint_ogc.pth"
RESUME_CHECKPOINT = None
STATS_DIR = "./stats"

device = resolve_device(DEVICE)
TEXT_TRESHOLD = 0.25
BOX_THRESHOLD = 0.25
TOKEN_THRESHOLD = 0.35

""" data """
processor = DataProcessor(IMAGE_DIR, ANNOTATIONS_PATH, SPLITS_PATH)
annotations = processor.annotations

BATCH_SIZE = 2
train_loader = get_loader(processor, 'train', BATCH_SIZE)
val_loader = get_loader(processor, 'val', BATCH_SIZE)
test_loader = get_loader(processor, 'test', BATCH_SIZE)

loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
print("Data loaded!")
print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

""" model """
model, args, model_overrides = build_model_for_mode(
    CONFIG_PATH, MODEL_MODE, device=device, text_encoder=TEXT_ENCODER,
    extra_overrides={"anno_path": ANNOTATIONS_PATH},
)
if MODEL_MODE == "full":
    model = load_pretrained(model, PRETRAINED_CHECKPOINT)
freeze_encoders(model)
model = set_finetuning_mode(model)

""" criterion """
criterion = SetCriterion()
optimizer = torch.optim.AdamW(trainable_parameters(model), lr=1e-5, weight_decay=0.0001)


def train(epoch, *, box_threshold=BOX_THRESHOLD, token_threshold=TOKEN_THRESHOLD):
    print(f"Training on train set data")
    model.train()
    model.backbone.eval()
    model.bert.eval()
    loader = loaders['train']

    train_mae = 0
    train_rmse = 0

    train_tp = 0
    train_fp = 0
    train_fn = 0

    counter = 0
    counter_for_image = 0
    train_size = len(loader.dataset)

    for images, captions, shapes, img_caps in loader:  # list of tensors, list of list [caption] for each image
        # images: [b1_img, b2_img,...] captions: [ [b1_cap1, b1_cap2], [b2_cap1, b2_cap2], ...]

        mask_bi = [i for i, img_cap_list in enumerate(img_caps) for _ in
                   img_cap_list]  # index for each img,cap pair in the batch
        anno_b = [annotations[img_cap] for img_cap_list in img_caps for img_cap in img_cap_list]
        img_caps = [img_cap for img_cap_list in img_caps for img_cap in img_cap_list]
        shapes = [shapes[i] for i, caption_list in enumerate(captions) for _ in caption_list]

        optimizer.zero_grad()

        # duplicate each image number of times that is equal to the number of captions for that image
        # Keep variable-sized images as a list; GroundingDINO builds the padding mask.
        images = [images[i].to(device) for i, caption_list in enumerate(captions)
                  for _ in caption_list]
        captions = [caption for caption_list in captions for caption in caption_list]
        outputs = model(images, captions=captions)

        outputs["pred_points"] = outputs["pred_boxes"][:, :, :2]

        # prepare targets
        emb_size = outputs["pred_logits"].shape[2]
        targets = prepare_targets(
            anno_b, captions, shapes, model.tokenizer, emb_size,
            image_group_ids=mask_bi, device=device, max_text_len=model.max_text_len
        )

        loss_dict = criterion(outputs, targets, mask_bi)
        weight_dict = criterion.weight_dict

        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss.backward()
        optimizer.step()

        counter_for_image += 1
        results = threshold(
            outputs, captions, model.tokenizer, TEXT_TRESHOLD,
            box_threshold=box_threshold, token_threshold=token_threshold)
        for b in range(len(results)):  # (bs*num_cap)
            boxes, logits, phrases = results[b]
            boxes = [box.tolist() for box in boxes]
            logits = logits.tolist()

            points = [[box[0], box[1]] for box in boxes]  # center points

            # calculate error
            pred_cnt = len(points)
            gt_cnt = len(targets[b]["points"])
            cnt_err = abs(pred_cnt - gt_cnt)
            train_mae += cnt_err
            train_rmse += cnt_err ** 2

            # calculate loc metric
            TP, FP, FN, precision, recall, f1 = calc_loc_metric(boxes, targets[b]["points"])
            train_tp += TP
            train_fp += FP
            train_fn += FN

            counter += 1

            print(
                f'[train] ep {epoch} ({counter_for_image}/{train_size}), {img_caps[b]}, caption: {captions[b]}, actual-predicted: {gt_cnt} vs {pred_cnt}, error: {pred_cnt - gt_cnt}. Current MAE: {int(train_mae / counter)}, RMSE: {int((train_rmse / counter) ** 0.5)} | TP = {TP}, FP = {FP}, FN = {FN}, precision = {precision:.2f}, recall = {recall:.2f}, F1 = {f1:.2f}')

    train_mae = train_mae / counter
    train_rmse = (train_rmse / counter) ** 0.5

    train_precision = train_tp / (train_tp + train_fp) if train_tp + train_fp != 0 else 0.0
    train_recall = train_tp / (train_tp + train_fn) if train_tp + train_fn != 0 else 0.0
    train_f1 = 2 * train_precision * train_recall / (
            train_precision + train_recall) if train_precision + train_recall != 0 else 0.0

    return train_mae, train_rmse, train_tp, train_fp, train_fn, train_precision, train_recall, train_f1


def eval(split, epoch=None, *, box_threshold=BOX_THRESHOLD, token_threshold=TOKEN_THRESHOLD):
    print(f"Evaluation on {split} set")
    model.eval()
    loader = loaders[split]

    eval_mae = 0
    eval_rmse = 0

    eval_tp = 0
    eval_fp = 0
    eval_fn = 0

    counter = 0
    counter_for_image = 0
    eval_size = len(loader.dataset)

    for images, captions, shapes, img_caps in loader:  # list of tensors, list, list, list

        anno_b = [annotations[img_cap] for img_cap_list in img_caps for img_cap in img_cap_list]
        img_caps = [img_cap for img_cap_list in img_caps for img_cap in img_cap_list]
        shapes = [shapes[i] for i, caption_list in enumerate(captions) for _ in caption_list]

        images = [images[i].to(device) for i, caption_list in enumerate(captions)
                  for _ in caption_list]
        captions = [caption for caption_list in captions for caption in caption_list]
        with torch.no_grad():
            outputs = model(images, captions=captions)

        outputs["pred_points"] = outputs["pred_boxes"][:, :, :2]

        # prepare targets
        emb_size = outputs["pred_logits"].shape[2]
        targets = prepare_targets(
            anno_b, captions, shapes, model.tokenizer, emb_size,
            image_group_ids=mask_bi, device=device, max_text_len=model.max_text_len
        )

        counter_for_image += 1

        results = threshold(
            outputs, captions, model.tokenizer, TEXT_TRESHOLD,
            box_threshold=box_threshold, token_threshold=token_threshold)
        for b in range(len(results)):
            boxes, logits, phrases = results[b]
            boxes = [box.tolist() for box in boxes]
            logits = logits.tolist()

            points = [[box[0], box[1]] for box in boxes]

            # calculate error
            pred_cnt = len(points)
            gt_cnt = len(targets[b]["points"])
            cnt_err = abs(pred_cnt - gt_cnt)
            eval_mae += cnt_err
            eval_rmse += cnt_err ** 2

            # calculate loc metric
            TP, FP, FN, precision, recall, f1 = calc_loc_metric(boxes, targets[b]["points"])
            eval_tp += TP
            eval_fp += FP
            eval_fn += FN

            counter += 1

            print(
                f'[{split}] ep {epoch} ({counter_for_image}/{eval_size}), {img_caps[b]}, caption: {captions[b]}, actual-predicted: {gt_cnt} vs {pred_cnt}, error: {pred_cnt - gt_cnt}. Current MAE: {int(eval_mae / counter)}, RMSE: {int((eval_rmse / counter) ** 0.5)} | TP = {TP}, FP = {FP}, FN = {FN}, precision = {precision:.2f}, recall = {recall:.2f}, F1 = {f1:.2f}')

    eval_mae = eval_mae / counter
    eval_rmse = (eval_rmse / counter) ** 0.5

    eval_precision = eval_tp / (eval_tp + eval_fp) if eval_tp + eval_fp != 0 else 0.0
    eval_recall = eval_tp / (eval_tp + eval_fn) if eval_tp + eval_fn != 0 else 0.0
    eval_f1 = 2 * eval_precision * eval_recall / (
            eval_precision + eval_recall) if eval_precision + eval_recall != 0 else 0.0

    return eval_mae, eval_rmse, eval_tp, eval_fp, eval_fn, eval_precision, eval_recall, eval_f1


# Target construction and localization metrics live in utils.evaluation so
# notebook training and standalone inference use exactly the same contract.


# main 

stats_dir = STATS_DIR
os.makedirs(stats_dir, exist_ok=True)

stats_file = f"{stats_dir}/stats.txt"
stats = list()

print(f"Saving stats to {stats_file}")

with open(stats_file, 'a') as f:
    header = ['train_mae', 'train_rmse', 'train_TP', 'train_FP', 'train_FN', 'train_precision', 'train_recall',
              'train_f1', '||', 'val_mae', 'val_rmse', 'val_TP', 'val_FP', 'val_FN', 'val_precision', 'val_recall',
              'val_f1', '||', 'test_mae', 'test_rmse', 'test_TP', 'test_FP', 'test_FN', 'test_precision', 'test_recall',
              'test_f1']
    f.write("%s\n" % ' | '.join(header))

best_f1 = float('-inf')
best_state = None
for epoch in range(0, 2):

    train_mae, train_rmse, train_TP, train_FP, train_FN, train_precision, train_recall, train_f1 = train(epoch)
    val_mae, val_rmse, val_TP, val_FP, val_FN, val_precision, val_recall, val_f1 = eval('val', epoch)

    if best_f1 < val_f1:
        best_f1 = val_f1
        print(f"New best F1: {best_f1}")
        best_state = cpu_state_dict(model)

    stats.append(
        [train_mae, train_rmse, train_TP, train_FP, train_FN, train_precision, train_recall, train_f1, "||", val_mae,
         val_rmse, val_TP, val_FP, val_FN, val_precision, val_recall, val_f1, "||", 0, 0, 0, 0, 0, 0, 0, 0])

    with open(stats_file, 'a') as f:
        s = stats[-1]
        for i, x in enumerate(s):
            if type(x) != str:
                s[i] = str(round(x, 4))
        f.write("%s\n" % ' | '.join(s))

model_name = f'{stats_dir}/model-{MODEL_MODE}.pth'
if best_state is None:
    raise RuntimeError("No valid validation checkpoint was produced")
torch.save({
    "format_version": 1,
    "model_mode": MODEL_MODE,
    "config_overrides": model_overrides,
    "model": best_state,
    "best_val_f1": best_f1,
}, model_name)

# Inference on test set using the same notebook-selected architecture.
print(f"Inference on test set using best model: {model_name}")
model, args, _ = build_model_for_mode(
    CONFIG_PATH, MODEL_MODE, device=device, text_encoder=TEXT_ENCODER,
    extra_overrides={"anno_path": ANNOTATIONS_PATH},
)
checkpoint, _, _ = load_checkpoint(model, model_name, requested_mode=MODEL_MODE, strict=True)
model = set_finetuning_mode(freeze_encoders(model))
model.eval()
test_mae, test_rmse, test_TP, test_FP, test_FN, test_precision, test_recall, test_f1 = eval('test', -1)
print(
    f"test MAE: {test_mae:5.2f}, RMSE: {test_rmse:5.2f}, TP: {test_TP}, FP: {test_FP}, FN: {test_FN}, precision: {test_precision:5.2f}, recall: {test_recall:5.2f}, f1: {test_f1:5.2f}")
# write to stats file
line_inference = [0, 0, 0, 0, 0, 0, 0, 0, "||", 0, 0, 0, 0, 0, 0, 0, 0, "||", test_mae, test_rmse, test_TP, test_FP,
                  test_FN, test_precision, test_recall, test_f1]
with open(stats_file, 'a') as f:
    s = line_inference
    for i, x in enumerate(s):
        if type(x) != str:
            s[i] = str(round(x, 4))
    f.write("%s\n" % ' | '.join(s))
