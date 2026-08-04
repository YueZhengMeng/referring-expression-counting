import argparse
import csv
import os
import re
import sys
import textwrap

import matplotlib

matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

sys.path.append('GroundingDINO')
from groundingdino.util.base_api import threshold, load_model

from utils.processor import DataProcessor
from utils.image_loader import get_loader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
TEXT_TRESHOLD = 0.25
BOX_THRESHOLD = 0.25
TOKEN_THRESHOLD = 0.35


def sanitize_filename(s):
    """Sanitize a string to be safe for use as a filename."""
    return re.sub(r'[<>:"/\\|?*\s]', '_', s)


def visualize_prediction(image_path, gt_points_pixel, pred_points_norm, shape, caption, save_path):
    """Draw ground-truth (green circles) and predicted (red crosses) points on the image.

    Args:
        image_path: path to the original image file.
        gt_points_pixel: list of [x, y] in pixel coordinates.
        pred_points_norm: list of [x, y] in normalised [0, 1] coordinates.
        shape: (h, w) of the original image.
        caption: the referring-expression caption to display as title.
        save_path: where to save the resulting figure.
    """
    img = Image.open(image_path).convert("RGB")
    img = np.array(img)
    h, w = shape

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(img)

    # Ground-truth points — green circles
    if len(gt_points_pixel) > 0:
        gt_pts = np.array(gt_points_pixel)
        ax.scatter(gt_pts[:, 0], gt_pts[:, 1], c='lime', s=80, marker='o',
                   edgecolors='darkgreen', linewidths=1.5,
                   label=f'GT ({len(gt_pts)})', zorder=5)

    # Predicted points — red crosses (convert normalised → pixel)
    if len(pred_points_norm) > 0:
        pred_pts = np.asarray(pred_points_norm, dtype=np.float64).copy()
        if pred_pts.ndim != 2 or pred_pts.shape[1] < 2:
            raise ValueError(f"Expected predicted points with shape (N, 2), got {pred_pts.shape}")
        pred_pts = pred_pts[:, :2]
        finite_mask = np.isfinite(pred_pts).all(axis=1)
        in_image_mask = ((pred_pts >= 0.0) & (pred_pts <= 1.0)).all(axis=1)
        valid_mask = finite_mask & in_image_mask
        if not valid_mask.all():
            print(f"  [WARN] ignoring {int((~valid_mask).sum())} invalid normalized predictions for {image_path}")
        pred_pts = pred_pts[valid_mask]
        if len(pred_pts) > 0:
            pred_pts[:, 0] *= w
            pred_pts[:, 1] *= h
            ax.scatter(pred_pts[:, 0], pred_pts[:, 1], c='red', s=80, marker='x',
                       linewidths=2, label=f'Pred ({len(pred_pts)})', zorder=5)

    ax.legend(loc='upper right')
    # display image filename + caption as title
    img_name = os.path.basename(image_path)
    wrapped_caption = '\n'.join(textwrap.wrap(f'{img_name}  |  {caption}', width=100))
    ax.set_title(wrapped_caption, fontsize=9, loc='center', pad=6)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def eval(model, loader, annotations, image_dir, output_dir, split,
         box_threshold=BOX_THRESHOLD, token_threshold=TOKEN_THRESHOLD):
    print(f"Inference on {split} set")
    model.eval()

    # --- set up output directories ---
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "case_study_results.csv")

    eval_mae = 0
    eval_rmse = 0

    eval_tp = 0
    eval_fp = 0
    eval_fn = 0

    counter = 0
    eval_size = sum(len(tuples) for tuples in loader.dataset.img_cap_tuples)

    # --- open CSV writer ---
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'image_id', 'caption', 'actual', 'predicted',
        'error', 'abs_error', 'vis_path'
    ])

    for images, captions, shapes, img_caps in loader:  # list of tensors, list, list, list

        anno_b = [annotations[img_cap] for img_cap_list in img_caps for img_cap in img_cap_list]
        img_caps = [img_cap for img_cap_list in img_caps for img_cap in img_cap_list]
        shapes = [shapes[i] for i, caption_list in enumerate(captions) for _ in caption_list]

        # save original GT points (pixel coords)
        orig_gt_points = [list(anno['points']) for anno in anno_b]

        # Keep each image unpadded so GroundingDINO can build a per-image mask.
        images = [images[i].to(device) for i, caption_list in enumerate(captions)
                  for _ in caption_list]
        captions = [caption for caption_list in captions for caption in caption_list]
        with torch.no_grad():
            outputs = model(images, captions=captions)

        outputs["pred_points"] = outputs["pred_boxes"][:, :, :2]

        # prepare targets without mutating shared annotations
        emb_size = outputs["pred_logits"].shape[2]
        targets = prepare_targets(anno_b, captions, shapes, emb_size,
                                  target_device=outputs['pred_logits'].device)

        results = threshold(
            outputs, captions, model.tokenizer, TEXT_TRESHOLD,
            box_threshold=box_threshold, token_threshold=token_threshold)
        for b in range(len(results)):
            boxes, logits, phrases = results[b]
            boxes = [box.tolist() for box in boxes]
            logits = logits.tolist()

            # convert boxes to points (normalised cx, cy)
            points = [[box[0], box[1]] for box in boxes]

            # calculate error
            pred_cnt = len(points)
            if pred_cnt == 0:
                print(f"  [INFO] no prediction passed thresholds for {img_caps[b]} "
                      f"(box>{box_threshold}, token>{token_threshold})")
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

            # --- visualization ---
            image_id, _ = img_caps[b]
            image_path = os.path.join(image_dir, image_id)
            vis_filename = (f"{counter:04d}_{sanitize_filename(image_id)}_"
                            f"{sanitize_filename(captions[b][:60])}.png")
            vis_path = os.path.join(vis_dir, vis_filename)
            try:
                visualize_prediction(image_path, orig_gt_points[b], points,
                                     shapes[b], captions[b], vis_path)
            except FileNotFoundError:
                vis_path = 'FILE_NOT_FOUND'
                print(f"  [WARN] image not found: {image_path}")

            # --- CSV row ---
            csv_writer.writerow([
                image_id, captions[b], gt_cnt, pred_cnt,
                pred_cnt - gt_cnt, cnt_err, vis_path
            ])

            print(
                f'[{split}] ({counter}/{eval_size}), {img_caps[b]}, caption: {captions[b]}, actual-predicted: {gt_cnt} vs {pred_cnt}, error: {pred_cnt - gt_cnt}. '
                f'Current MAE: {int(eval_mae / counter)}, RMSE: {int((eval_rmse / counter) ** 0.5)} | TP = {TP}, FP = {FP}, FN = {FN}, precision = {precision:.2f}, recall = {recall:.2f}, F1 = {f1:.2f}'
            )

    csv_file.close()
    print(f"\nVisualizations saved to: {vis_dir}")
    print(f"CSV results saved to: {csv_path}")

    eval_mae = eval_mae / counter
    eval_rmse = (eval_rmse / counter) ** 0.5

    eval_precision = eval_tp / (eval_tp + eval_fp) if eval_tp + eval_fp != 0 else 0.0
    eval_recall = eval_tp / (eval_tp + eval_fn) if eval_tp + eval_fn != 0 else 0.0
    eval_f1 = 2 * eval_precision * eval_recall / (
            eval_precision + eval_recall) if eval_precision + eval_recall != 0 else 0.0

    return eval_mae, eval_rmse, eval_tp, eval_fp, eval_fn, eval_precision, eval_recall, eval_f1


def prepare_targets(anno_b, captions, shapes, emb_size, target_device=None):
    target_device = target_device or device
    gt_points_b = [
        np.asarray(anno['points'], dtype=np.float32).reshape(-1, 2) / np.array(shape)[::-1]
        for anno, shape in zip(anno_b, shapes)
    ]
    gt_points = [torch.from_numpy(img_points).to(torch.float32) for img_points in gt_points_b]

    gt_logits = [torch.zeros((img_points.shape[0], emb_size)) for img_points in gt_points]

    tokenized = model.tokenizer(captions, padding="longest", return_tensors="pt")

    # find last index of special token (.)
    end_idxes = [torch.where(input_ids == 1012)[0][-1] for input_ids in tokenized['input_ids']]
    for i, end_idx in enumerate(end_idxes):
        gt_logits[i][:, :end_idx] = 1.0

    caption_sizes = [end_idx + 2 for end_idx in end_idxes]  # incl. CLS and SEP

    targets = [{"points": img_gt_points.to(target_device), "labels": img_gt_logits.to(target_device), "caption_size": caption_size}
               for img_gt_points, img_gt_logits, caption_size in zip(gt_points, gt_logits, caption_sizes)]

    return targets


def distance_threshold_func(boxes):  # list of [xc,yc,w,h]
    if len(boxes) == 0:
        return 0.0
    # find median index of boxes areas
    areas = [box[2] * box[3] for box in boxes]
    median_idx = np.argsort(areas)[len(areas) // 2]
    median_box = boxes[median_idx]
    w = median_box[2]
    h = median_box[3]

    threshold = np.sqrt(w ** 2 + h ** 2) / 2.0

    return threshold


def calc_loc_metric(pred_boxes, gt_points):  # list of [xc,yc,w,h], tensor of (nt,2)
    if len(pred_boxes) == 0:
        FN = len(gt_points)
        return 0, 0, FN, 0, 0, 0

    dist_threshold = distance_threshold_func(pred_boxes)
    pred_points = np.array([[box[0], box[1]] for box in pred_boxes])
    gt_points = gt_points.cpu().detach().numpy()

    # create a cost matrix
    cost_matrix = cdist(pred_points, gt_points, metric='euclidean')

    # use Hungarian algorithm to find optimal assignment
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

    # determine TP, FP, FN
    TP = 0
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        if cost_matrix[pred_idx, gt_idx] < dist_threshold:
            TP += 1

    FP = len(pred_points) - TP
    FN = len(gt_points) - TP

    Precision = TP / (TP + FP) if TP + FP != 0 else 0.0
    Recall = TP / (TP + FN) if TP + FN != 0 else 0.0
    F1 = 2 * (Precision * Recall) / (Precision + Recall) if Precision + Recall != 0 else 0.0

    return TP, FP, FN, Precision, Recall, F1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference on train/val/test sets')
    parser.add_argument('--checkpoint', type=str,
                        default='F://GroundingREC/rec_model.pth',
                        help='Path to the model checkpoint (.pth file)')
    parser.add_argument('--config', type=str,
                        default='GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
                        help='Path to the model config file')
    parser.add_argument('--split', type=str,
                        default='test',
                        choices=['train', 'val', 'test'],
                        help='Split to evaluate on')
    parser.add_argument('--batch_size', type=int,
                        default=2,
                        help='Batch size for data loaders')
    parser.add_argument('--output_dir', type=str,
                        default='./case_study_output',
                        help='Directory to save visualizations and CSV')
    parser.add_argument('--box_threshold', type=float,
                        default=BOX_THRESHOLD,
                        help='Minimum global score for keeping a prediction')
    parser.add_argument('--token_threshold', type=float,
                        default=TOKEN_THRESHOLD,
                        help='Minimum local token score for keeping a prediction')
    args = parser.parse_args()

    """ model """
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model = load_model(args.config, args.checkpoint, device=device)
    model = model.to(device)

    """ data """
    processor = DataProcessor()
    annotations = processor.annotations
    image_dir = processor.get_image_path()

    split = args.split
    assert split in ['train', 'val', 'test']

    loader = get_loader(processor, split, args.batch_size)
    print("Data loaded!")
    print(f"{split}: {len(loader.dataset)}")

    """ inference """
    output_dir = os.path.join(args.output_dir, split)
    mae, rmse, TP, FP, FN, precision, recall, f1 = eval(
        model, loader, annotations, image_dir, output_dir, split,
        box_threshold=args.box_threshold, token_threshold=args.token_threshold)
    print(
        f'[{split}] MAE: {mae:5.2f}, RMSE: {rmse:5.2f}, TP: {TP}, FP: {FP}, FN: {FN}, '
        f'precision: {precision:5.2f}, recall: {recall:5.2f}, F1: {f1:5.2f}'
    )
