import csv
import math
from pathlib import Path

_LEVELS = {"off": 0, "epoch": 1, "batch": 2, "caption": 3}

_EPOCH_FIELDS = [
    "epoch", "split", "is_final", "caption_count", "gt_count_total",
    "pred_count_total", "mean_gt_count", "mean_pred_count",
    "mean_signed_count_error", "mae", "rmse", "under_count_rate",
    "over_count_rate", "exact_count_rate", "within_one_rate", "tp", "fp",
    "fn", "precision", "recall", "f1", "lr_mean", "lr_min", "lr_max",
    "attribute_token_mean", "attribute_token_min", "attribute_token_max",
    "empty_attribute_rate",
    "loss_label", "loss_point", "loss_contrast_rec", "loss_contrast_attr",
    "loss_contrast_asd", "loss_contrast_img_hm", "loss_contrast_img",
    "weighted_loss_label", "weighted_loss_point", "weighted_loss_contrast_rec",
    "weighted_loss_contrast_attr", "weighted_loss_contrast_asd",
    "weighted_loss_contrast_img_hm", "weighted_loss_contrast_img", "total_loss",
    "valid_token_mean", "target_count_total", "empty_target_count",
    "matched_count_total", "matched_per_target", "class_cost_mean",
    "point_cost_mean", "repulsive_cost_mean", "weighted_class_cost_mean",
    "weighted_point_cost_mean", "weighted_repulsive_cost_mean", "total_cost_mean",
    "matched_class_cost_mean", "matched_point_cost_mean",
    "matched_repulsive_cost_mean", "matched_total_cost_mean",
    "class_point_cost_ratio",
]

_BATCH_FIELDS = [
    "epoch", "split", "batch", "optimizer_step", "caption_count",
    "gt_count_total", "pred_count_total", "signed_count_error_total",
    "mae_sum", "rmse_sum", "tp", "fp", "fn", "lr", "grad_norm_total",
    "grad_norm_class", "grad_norm_point", "grad_norm_contrast",
    "grad_norm_w2c_layers", "grad_norm_w2s_layers", "grad_norm_w2c_update",
    "grad_norm_w2s_update", "grad_norm_w2c_bbox", "grad_norm_w2s_bbox",
    "grad_norm_w2c_total", "grad_norm_w2s_total", "attribute_token_mean",
    "attribute_token_min", "attribute_token_max", "empty_attribute_rate",
    "loss_label", "loss_point", "loss_contrast_rec", "loss_contrast_attr",
    "loss_contrast_asd", "loss_contrast_img_hm", "loss_contrast_img", "weighted_loss_label",
    "weighted_loss_point", "weighted_loss_contrast_rec",
    "weighted_loss_contrast_attr", "weighted_loss_contrast_asd",
    "weighted_loss_contrast_img_hm", "weighted_loss_contrast_img", "total_loss",
    "valid_token_mean", "valid_token_min", "valid_token_max", "target_count_total",
    "empty_target_count", "matched_count_total", "matched_per_target",
    "class_cost_mean", "point_cost_mean", "repulsive_cost_mean",
    "weighted_class_cost_mean", "weighted_point_cost_mean",
    "weighted_repulsive_cost_mean", "total_cost_mean", "matched_class_cost_mean",
    "matched_point_cost_mean", "matched_repulsive_cost_mean",
    "matched_total_cost_mean", "class_point_cost_ratio", "score_mean", "score_p50",
    "score_max", "kept_count",
]

_CAPTION_FIELDS = [
    "epoch", "split", "batch", "caption_index", "image_id", "image_group_id",
    "class_name", "attribute_name", "gt_count", "pred_count", "signed_count_error",
    "fp", "fn", "precision", "recall", "f1", "valid_token_count",
    "selected_score_mean", "selected_score_max", "kept_count",
]


def _as_float(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    value = float(value)
    return value if math.isfinite(value) else None


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _percentile(values, percentile):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    index = (len(values) - 1) * percentile / 100.0
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    fraction = index - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _zero_if_none(value):
    return 0.0 if value is None else value


class DiagnosticsWriter:
    """Write stable, append-only CSV diagnostics for one training run."""

    def __init__(self, stats_dir, level="batch"):
        if level not in _LEVELS:
            raise ValueError(f"unknown diagnostics level: {level}")
        self.level = level
        self._files = {}
        self._writers = {}
        if level == "off":
            return
        directory = Path(stats_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self._open("epoch", directory / "epoch_metrics.csv", _EPOCH_FIELDS)
        if _LEVELS[level] >= _LEVELS["batch"]:
            self._open("batch", directory / "batch_diagnostics.csv", _BATCH_FIELDS)
        if _LEVELS[level] >= _LEVELS["caption"]:
            self._open("caption", directory / "caption_diagnostics.csv", _CAPTION_FIELDS)

    def _open(self, name, path, fields):
        handle = path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        self._files[name] = handle
        self._writers[name] = writer

    def _write(self, name, record):
        writer = self._writers.get(name)
        if writer is None:
            return
        writer.writerow({key: _as_float(value) if isinstance(value, (int, float)) else value
                         for key, value in record.items()})
        self._files[name].flush()

    def write_epoch(self, record):
        if _LEVELS[self.level] >= _LEVELS["epoch"]:
            self._write("epoch", record)

    def write_batch(self, record):
        if _LEVELS[self.level] >= _LEVELS["batch"]:
            self._write("batch", record)

    def write_caption(self, record):
        if _LEVELS[self.level] >= _LEVELS["caption"]:
            self._write("caption", record)

    def close(self):
        for handle in self._files.values():
            handle.close()
        self._files.clear()
        self._writers.clear()


def _loss_values(criterion):
    values = {
        key: _as_float(value)
        for key, value in criterion.last_loss_breakdown.items()
    }
    if "total" in values:
        values["total_loss"] = values.pop("total")
    return values


def _matcher_values(criterion):
    return dict(criterion.matcher.last_diagnostics)


def gradient_norms(model):
    """Return total, head, and optional W2Net branch gradient norms."""
    groups = {
        "total": [], "class": [], "point": [], "contrast": [],
        "w2c_layers": [], "w2s_layers": [],
        "w2c_update": [], "w2s_update": [],
        "w2c_bbox": [], "w2s_bbox": [],
        "w2c_total": [], "w2s_total": [],
    }
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        norm = parameter.grad.detach().float().norm(2)
        groups["total"].append(norm)
        lowered = name.lower()
        is_w2s = "w2s" in lowered
        if "class_embed" in lowered or "logit" in lowered:
            groups["class"].append(norm)
        if "bbox_embed" in lowered or "w2s_bbox" in lowered:
            groups["point"].append(norm)
        if any(token in lowered for token in ("fusion", "img_emb", "txt_emb", "contrast")):
            groups["contrast"].append(norm)

        if "w2s_layers" in lowered:
            groups["w2s_layers"].append(norm)
            groups["w2s_total"].append(norm)
        elif "w2c_update_layers" in lowered:
            groups["w2c_update"].append(norm)
            groups["w2c_total"].append(norm)
        elif "w2s_update_layers" in lowered:
            groups["w2s_update"].append(norm)
            groups["w2s_total"].append(norm)
        elif "w2s_bbox_embed" in lowered:
            groups["w2s_bbox"].append(norm)
            groups["w2s_total"].append(norm)
        elif "decoder.layers" in lowered and not is_w2s:
            groups["w2c_layers"].append(norm)
            groups["w2c_total"].append(norm)
        elif "bbox_embed" in lowered and not is_w2s:
            groups["w2c_bbox"].append(norm)
            groups["w2c_total"].append(norm)

    result = {}
    for name, values in groups.items():
        result[f"grad_norm_{name}"] = (
            float(torch_stack_norm(values)) if values else None
        )
    return result


def attribute_token_stats(outputs):
    """Summarize W2Net's attribute-token input without retaining graph tensors."""
    mask = outputs.get("attribute_token_mask")
    fields = {
        "attribute_token_mean": None,
        "attribute_token_min": None,
        "attribute_token_max": None,
        "empty_attribute_rate": None,
    }
    if mask is None or mask.numel() == 0:
        return fields
    import torch
    counts = mask.detach().to(dtype=torch.float32).sum(dim=-1)
    fields.update({
        "attribute_token_mean": float(counts.mean().item()),
        "attribute_token_min": float(counts.min().item()),
        "attribute_token_max": float(counts.max().item()),
        "empty_attribute_rate": float((counts == 0).float().mean().item()),
    })
    return fields


def torch_stack_norm(values):
    # Keep this helper local so diagnostics only needs torch when gradients are used.
    import torch
    return torch.stack(values).norm(2).item()


def collect_prediction_batch(results, targets, calc_loc_metric, epoch, split,
                             batch_index, image_ids=None, image_group_ids=None):
    """Summarize thresholded predictions and produce optional caption rows."""
    summary = {
        "caption_count": 0, "gt_count_total": 0, "pred_count_total": 0,
        "signed_count_error_total": 0, "mae_sum": 0, "rmse_sum": 0,
        "tp": 0, "fp": 0, "fn": 0, "under_count": 0, "over_count": 0,
        "exact_count": 0, "within_one": 0, "kept_count": 0,
        "scores": [], "caption_rows": [],
    }
    for index, (boxes, scores, _phrases) in enumerate(results):
        boxes = [box.tolist() for box in boxes]
        gt_count = len(targets[index]["points"])
        pred_count = len(boxes)
        tp, fp, fn, precision, recall, f1 = calc_loc_metric(
            boxes, targets[index]["points"]
        )
        signed_error = pred_count - gt_count
        abs_error = abs(signed_error)
        score_values = []
        if scores is not None and hasattr(scores, "numel") and scores.numel():
            score_values = [float(value) for value in scores.detach().cpu().reshape(-1)]
        summary["caption_count"] += 1
        summary["gt_count_total"] += gt_count
        summary["pred_count_total"] += pred_count
        summary["signed_count_error_total"] += signed_error
        summary["mae_sum"] += abs_error
        summary["rmse_sum"] += abs_error ** 2
        summary["tp"] += tp
        summary["fp"] += fp
        summary["fn"] += fn
        summary["under_count"] += int(signed_error < 0)
        summary["over_count"] += int(signed_error > 0)
        summary["exact_count"] += int(signed_error == 0)
        summary["within_one"] += int(abs_error <= 1)
        summary["kept_count"] += pred_count
        summary["scores"].extend(score_values)
        valid_tokens = targets[index].get("valid_token_mask")
        valid_token_count = int(valid_tokens.sum().item()) if valid_tokens is not None else None
        summary["caption_rows"].append({
            "epoch": epoch,
            "split": split,
            "batch": batch_index,
            "caption_index": index,
            "image_id": image_ids[index] if image_ids else None,
            "image_group_id": image_group_ids[index] if image_group_ids else None,
            "class_name": targets[index].get("class_name", ""),
            "attribute_name": targets[index].get("attribute_name", ""),
            "gt_count": gt_count,
            "pred_count": pred_count,
            "signed_count_error": signed_error,
            "abs_count_error": abs_error,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "valid_token_count": valid_token_count,
            "selected_score_mean": _mean(score_values),
            "selected_score_max": max(score_values) if score_values else None,
            "kept_count": pred_count,
        })
    summary["score_mean"] = _mean(summary["scores"])
    summary["score_p50"] = _percentile(summary["scores"], 50)
    summary["score_max"] = max(summary["scores"]) if summary["scores"] else None
    return summary


class EpochAccumulator:
    def __init__(self):
        self.prediction = {
            "caption_count": 0, "gt_count_total": 0, "pred_count_total": 0,
            "signed_count_error_total": 0, "mae_sum": 0, "rmse_sum": 0,
            "tp": 0, "fp": 0, "fn": 0, "under_count": 0, "over_count": 0,
            "exact_count": 0, "within_one": 0,
        }
        self.loss_sums = {}
        self.matcher_sums = {}
        self.matcher_totals = {
            "target_count_total": 0,
            "empty_target_count": 0,
            "matched_count_total": 0,
        }
        self.branch_sums = {}
        self.branch_steps = 0
        self.loss_steps = 0
        self.matcher_steps = 0
        self.lrs = []

    def add_prediction(self, summary):
        for key in self.prediction:
            self.prediction[key] += summary.get(key, 0)

    def add_loss(self, criterion):
        for key, value in _loss_values(criterion).items():
            if value is not None:
                self.loss_sums[key] = self.loss_sums.get(key, 0.0) + value
        self.loss_steps += 1

    def add_matcher(self, criterion):
        for key, value in _matcher_values(criterion).items():
            if value is None:
                continue
            if key in self.matcher_totals:
                self.matcher_totals[key] += value
            else:
                self.matcher_sums[key] = self.matcher_sums.get(key, 0.0) + value
        self.matcher_steps += 1

    def add_branch(self, branch_stats):
        added = False
        for key, value in branch_stats.items():
            if value is not None:
                self.branch_sums[key] = self.branch_sums.get(key, 0.0) + value
                added = True
        if added:
            self.branch_steps += 1

    def add_lr(self, lr):
        if lr is not None:
            self.lrs.append(float(lr))

    def means(self, values, count):
        return {key: value / count for key, value in values.items()} if count else {}

    def epoch_record(self, epoch, split, is_final=False):
        p = self.prediction
        n = p["caption_count"]
        lr_mean = _mean(self.lrs)
        record = {
            "epoch": epoch,
            "split": split,
            "is_final": bool(is_final),
            "caption_count": n,
            "gt_count_total": p["gt_count_total"],
            "pred_count_total": p["pred_count_total"],
            "mean_gt_count": p["gt_count_total"] / n if n else None,
            "mean_pred_count": p["pred_count_total"] / n if n else None,
            "mean_signed_count_error": p["signed_count_error_total"] / n if n else None,
            "mae": p["mae_sum"] / n if n else None,
            "rmse": math.sqrt(p["rmse_sum"] / n) if n else None,
            "under_count_rate": p["under_count"] / n if n else None,
            "over_count_rate": p["over_count"] / n if n else None,
            "exact_count_rate": p["exact_count"] / n if n else None,
            "within_one_rate": p["within_one"] / n if n else None,
            "tp": p["tp"], "fp": p["fp"], "fn": p["fn"],
            "precision": p["tp"] / (p["tp"] + p["fp"]) if p["tp"] + p["fp"] else 0.0,
            "recall": p["tp"] / (p["tp"] + p["fn"]) if p["tp"] + p["fn"] else 0.0,
            "lr_mean": lr_mean,
            "lr_min": min(self.lrs) if self.lrs else None,
            "lr_max": max(self.lrs) if self.lrs else None,
        }
        precision = record["precision"]
        recall = record["recall"]
        record["f1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        record.update(self.means(self.loss_sums, self.loss_steps))
        record.update(self.means(self.matcher_sums, self.matcher_steps))
        record.update(self.matcher_totals)
        record.update(self.means(self.branch_sums, self.branch_steps))
        record["matched_per_target"] = (
            record.get("matched_count_total", 0.0) / record.get("target_count_total", 1.0)
            if record.get("target_count_total", 0.0) else None
        )
        return record


def batch_record(epoch, split, batch_index, optimizer_step, lr, summary,
                 criterion, gradient=None, extra=None):
    record = {
        "epoch": epoch,
        "split": split,
        "batch": batch_index,
        "optimizer_step": optimizer_step,
        "lr": lr,
        "caption_count": summary.get("caption_count"),
        "gt_count_total": summary.get("gt_count_total"),
        "pred_count_total": summary.get("pred_count_total"),
        "signed_count_error_total": summary.get("signed_count_error_total"),
        "mae_sum": summary.get("mae_sum"),
        "rmse_sum": summary.get("rmse_sum"),
        "tp": summary.get("tp"), "fp": summary.get("fp"), "fn": summary.get("fn"),
        "score_mean": summary.get("score_mean"), "score_p50": summary.get("score_p50"),
        "score_max": summary.get("score_max"), "kept_count": summary.get("kept_count"),
    }
    record["matched_per_target"] = (
        record.get("matched_count_total", 0.0) / record.get("target_count_total", 1.0)
        if record.get("target_count_total", 0.0) else None
    )
    record.update(_loss_values(criterion))
    record.update(_matcher_values(criterion))
    if gradient:
        record.update(gradient)
    if extra:
        record.update(extra)
    return record
