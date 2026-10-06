import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

import torch
from tqdm import tqdm

sys.path.append("GroundingDINO")
from groundingdino.util.base_api import (
    _valid_caption_token_indices,
    load_model,
)

from utils.evaluation import calc_loc_metric, prepare_targets, seed_everything
from utils.image_loader import get_loader
from utils.processor import DataProcessor

DEFAULT_BOX_THRESHOLDS = [round(value, 2) for value in torch.arange(0.10, 0.501, 0.05).tolist()]
DEFAULT_TOKEN_THRESHOLDS = [round(value, 2) for value in torch.arange(0.1, 0.501, 0.05).tolist()]
MODEL_KWARGS = {
    "groundingdino": {},
    "groundingrec": {"global_local_fusion": True},
    "crex_groundingrec": {"global_local_fusion": True},
    "crex_groundingdino": {},
    "dcount": {"DCount": True},
    "w2net": {"W2NET": True},
}


class MetricAccumulator:
    def __init__(self):
        self.caption_count = 0
        self.gt_count_total = 0
        self.pred_count_total = 0
        self.signed_count_error_total = 0
        self.mae_sum = 0.0
        self.rmse_sum = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.under_count = 0
        self.over_count = 0
        self.exact_count = 0
        self.within_one = 0

    def update(self, boxes, target):
        boxes = [box.tolist() for box in boxes]
        gt_count = len(target["points"])
        pred_count = len(boxes)
        tp, fp, fn, _precision, _recall, _f1 = calc_loc_metric(
            boxes, target["points"]
        )
        signed_error = pred_count - gt_count
        abs_error = abs(signed_error)

        self.caption_count += 1
        self.gt_count_total += gt_count
        self.pred_count_total += pred_count
        self.signed_count_error_total += signed_error
        self.mae_sum += abs_error
        self.rmse_sum += abs_error ** 2
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.under_count += int(signed_error < 0)
        self.over_count += int(signed_error > 0)
        self.exact_count += int(signed_error == 0)
        self.within_one += int(abs_error <= 1)

    def row(self, box_threshold, token_threshold):
        n = self.caption_count
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "box_threshold": box_threshold,
            "token_threshold": token_threshold,
            "caption_count": n,
            "gt_count_total": self.gt_count_total,
            "pred_count_total": self.pred_count_total,
            "mean_gt_count": self.gt_count_total / n if n else 0.0,
            "mean_pred_count": self.pred_count_total / n if n else 0.0,
            "mean_signed_count_error": (
                self.signed_count_error_total / n if n else 0.0
            ),
            "mae": self.mae_sum / n if n else 0.0,
            "rmse": (self.rmse_sum / n) ** 0.5 if n else 0.0,
            "under_count_rate": self.under_count / n if n else 0.0,
            "over_count_rate": self.over_count / n if n else 0.0,
            "exact_count_rate": self.exact_count / n if n else 0.0,
            "within_one_rate": self.within_one / n if n else 0.0,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


def freeze_encoders(model):
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    for parameter in model.bert.parameters():
        parameter.requires_grad_(False)
    return model


def load_model_for_variant(model_name, config, checkpoint, device):
    try:
        kwargs = MODEL_KWARGS[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown model variant: {model_name}") from exc
    model = load_model(config, checkpoint, device=device, **kwargs).to(device)
    model = freeze_encoders(model)
    model.eval()
    return model


def _score_cache(outputs, captions, tokenizer, max_text_len):
    """Cache threshold-independent scores using base_api.threshold semantics."""
    logits_batch = outputs["pred_logits"].detach().cpu().sigmoid()
    boxes_batch = outputs["pred_boxes"].detach().cpu()
    tokenized_batch = tokenizer(
        captions,
        padding="longest",
        truncation=True,
        max_length=max_text_len,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    punctuation = tokenizer(".", add_special_tokens=False)["input_ids"]
    punctuation_ids = (
        punctuation[0]
        if punctuation and isinstance(punctuation[0], list)
        else punctuation
    )
    model_text_mask = outputs.get("text_logit_mask")
    if model_text_mask is not None:
        model_text_mask = model_text_mask.detach().cpu().bool()

    cache = []
    for batch_index in range(logits_batch.shape[0]):
        encoded = {
            key: value[batch_index:batch_index + 1]
            if torch.is_tensor(value) else value[batch_index:batch_index + 1]
            for key, value in tokenized_batch.items()
        }
        cls_index, content_indices = _valid_caption_token_indices(
            tokenizer, encoded, punctuation_ids
        )
        if model_text_mask is not None:
            content_indices = [
                index for index in torch.where(model_text_mask[batch_index])[0].tolist()
                if index != cls_index
            ]
        prediction_logits = logits_batch[batch_index]
        cls_scores = prediction_logits[:, cls_index]
        if content_indices:
            content_scores = prediction_logits[:, content_indices].min(dim=1).values
        else:
            content_scores = torch.ones(prediction_logits.shape[0])
        cache.append({
            "cls_scores": cls_scores,
            "content_scores": content_scores,
            "boxes": boxes_batch[batch_index],
        })
    return cache


def _flatten_batch(images, captions, shapes, img_caps, annotations, device):
    image_group_ids = [
        i for i, caption_list in enumerate(captions) for _ in caption_list
    ]
    anno_b = [
        annotations[img_cap]
        for img_cap_list in img_caps
        for img_cap in img_cap_list
    ]
    flat_img_caps = [
        img_cap for img_cap_list in img_caps for img_cap in img_cap_list
    ]
    flat_shapes = [
        shapes[i] for i, caption_list in enumerate(captions) for _ in caption_list
    ]
    flat_images = [
        images[i].to(device)
        for i, caption_list in enumerate(captions)
        for _ in caption_list
    ]
    flat_captions = [
        caption for caption_list in captions for caption in caption_list
    ]
    return (
        flat_images,
        flat_captions,
        flat_shapes,
        flat_img_caps,
        anno_b,
        image_group_ids,
    )


def evaluate_grid(model, loader, annotations, device, box_thresholds,
                  token_thresholds, text_threshold):
    accumulators = {
        (float(box_threshold), float(token_threshold)): MetricAccumulator()
        for box_threshold in box_thresholds
        for token_threshold in token_thresholds
    }
    model.eval()
    with torch.no_grad():
        for images, captions, shapes, img_caps in tqdm(loader):
            (
                images, captions, shapes, img_caps, anno_b, image_group_ids,
            ) = _flatten_batch(
                images, captions, shapes, img_caps, annotations, device
            )
            outputs = model(images, captions=captions)
            targets = prepare_targets(
                anno_b,
                captions,
                shapes,
                model.tokenizer,
                image_group_ids=image_group_ids,
                device=device,
                max_text_len=model.max_text_len,
            )
            cache = _score_cache(
                outputs, captions, model.tokenizer, model.max_text_len
            )
            for box_threshold in box_thresholds:
                for token_threshold in token_thresholds:
                    key = (float(box_threshold), float(token_threshold))
                    accumulator = accumulators[key]
                    for item, target in zip(cache, targets):
                        keep = (
                                       item["cls_scores"] > box_threshold
                               ) & (
                                       item["content_scores"] > token_threshold
                               )
                        accumulator.update(item["boxes"][keep], target)
    return [
        accumulators[(float(box_threshold), float(token_threshold))].row(
            float(box_threshold), float(token_threshold)
        )
        for box_threshold in box_thresholds
        for token_threshold in token_thresholds
    ]


def _pareto_front(rows):
    for row in rows:
        row["is_pareto"] = True
    for index, candidate in enumerate(rows):
        for other_index, other in enumerate(rows):
            if index == other_index:
                continue
            no_worse = (
                    other["mae"] <= candidate["mae"]
                    and other["f1"] >= candidate["f1"]
            )
            strictly_better = (
                    other["mae"] < candidate["mae"]
                    or other["f1"] > candidate["f1"]
            )
            if no_worse and strictly_better:
                candidate["is_pareto"] = False
                break
    return [row for row in rows if row["is_pareto"]]


def select_threshold(rows, policy):
    best_mae = min(rows, key=lambda row: (row["mae"], -row["f1"]))
    best_f1 = max(rows, key=lambda row: (row["f1"], -row["mae"]))
    pareto = _pareto_front(rows)
    if policy == "mae":
        selected = best_mae
    elif policy == "f1":
        selected = best_f1
    else:
        mae_min = min(row["mae"] for row in pareto)
        mae_max = max(row["mae"] for row in pareto)
        f1_min = min(row["f1"] for row in pareto)
        f1_max = max(row["f1"] for row in pareto)
        mae_range = max(mae_max - mae_min, 1e-12)
        f1_range = max(f1_max - f1_min, 1e-12)
        selected = min(
            pareto,
            key=lambda row: (
                    ((row["mae"] - mae_min) / mae_range) ** 2
                    + ((f1_max - row["f1"]) / f1_range) ** 2
            ),
        )
    return selected, best_mae, best_f1, pareto


def write_rows(path, rows):
    fields = [
        "box_threshold", "token_threshold", "is_pareto", "caption_count",
        "gt_count_total", "pred_count_total", "mean_gt_count", "mean_pred_count",
        "mean_signed_count_error", "mae", "rmse", "under_count_rate",
        "over_count_rate", "exact_count_rate", "within_one_rate", "tp", "fp",
        "fn", "precision", "recall", "f1",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)

def json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def main():
    parser = argparse.ArgumentParser(
        description="Search post-training box/token thresholds on validation data"
    )
    parser.add_argument(
        "--model", choices=list(MODEL_KWARGS), required=True,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config",
        default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    )
    parser.add_argument("--image-dir", default="/home/pwb/pwb/rec-8k")
    parser.add_argument("--annotations", default="anno/annotations.json")
    parser.add_argument("--splits", default="anno/splits.json")
    parser.add_argument("--search-split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default=None)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument(
        "--box-thresholds", nargs="+", type=float, default=DEFAULT_BOX_THRESHOLDS,
    )
    parser.add_argument(
        "--token-thresholds", nargs="+", type=float, default=DEFAULT_TOKEN_THRESHOLDS,
    )
    parser.add_argument(
        "--selection", choices=["mae", "f1", "pareto"], default="pareto",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(
        f"Searching {len(args.box_thresholds)} x {len(args.token_thresholds)} "
        f"threshold pairs on {args.search_split}"
    )

    processor = DataProcessor(args.image_dir, args.annotations, args.splits)
    annotations = processor.annotations
    loaders = {
        split: get_loader(processor, split, args.batch_size)
        for split in (args.search_split, "test")
    }
    model = load_model_for_variant(args.model, args.config, args.checkpoint, device)
    rows = evaluate_grid(
        model,
        loaders[args.search_split],
        annotations,
        device,
        args.box_thresholds,
        args.token_thresholds,
        args.text_threshold,
    )
    selected, best_mae, best_f1, pareto = select_threshold(rows, args.selection)
    os.makedirs(args.output_dir, exist_ok=True)
    write_rows(os.path.join(args.output_dir, "threshold_grid_val.csv"), rows)

    selection = {
        "model": args.model,
        "checkpoint": os.path.abspath(args.checkpoint),
        "search_split": args.search_split,
        "selection": args.selection,
        "selected": selected,
        "best_mae": best_mae,
        "best_f1": best_f1,
        "pareto_count": len(pareto),
        "box_thresholds": args.box_thresholds,
        "token_thresholds": args.token_thresholds,
        "text_threshold": args.text_threshold,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(
            os.path.join(args.output_dir, "threshold_selection.json"),
            "w",
            encoding="utf-8",
    ) as file:
        json.dump(selection, file, indent=4, sort_keys=True, default=json_default)

    print(
        f"Selected box={selected['box_threshold']:.4f}, "
        f"token={selected['token_threshold']:.4f}, "
        f"val MAE={selected['mae']:.4f}, val F1={selected['f1']:.4f}"
    )

    if args.evaluate_test:
        test_rows = evaluate_grid(
            model,
            loaders["test"],
            annotations,
            device,
            [selected["box_threshold"]],
            [selected["token_threshold"]],
            args.text_threshold,
        )
        write_rows(os.path.join(args.output_dir, "threshold_selected_test.csv"), test_rows)
        print(
            f"Test MAE={test_rows[0]['mae']:.4f}, "
            f"test F1={test_rows[0]['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
