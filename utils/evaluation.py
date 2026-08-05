import math

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def _period_token_ids(tokenizer):
    ids = tokenizer(".", add_special_tokens=False).get("input_ids", [])
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return {int(token_id) for token_id in ids}


def tokenize_caption_targets(tokenizer, captions, max_text_len):
    """Return target labels/masks without assuming a particular tokenizer id."""
    tokenized = tokenizer(
        captions, padding="longest", truncation=True, max_length=max_text_len,
        return_tensors="pt", return_special_tokens_mask=True,
    )
    input_ids = tokenized["input_ids"]
    attention = tokenized["attention_mask"].bool()
    labels = torch.zeros((len(captions), max_text_len), dtype=torch.float32)
    valid = torch.zeros((len(captions), max_text_len), dtype=torch.bool)
    period_ids = _period_token_ids(tokenizer)
    for row in range(len(captions)):
        valid_positions = torch.where(attention[row])[0]
        end = int(valid_positions[-1]) + 1 if valid_positions.numel() else 0
        period_positions = [int(i) for i in valid_positions.tolist()
                            if int(input_ids[row, i]) in period_ids]
        if period_positions:
            end = period_positions[-1]
        if end:
            valid[row, :end] = True
            labels[row, :end] = 1.0
    return labels, valid, tokenized


def prepare_targets(anno_b, captions, shapes, tokenizer, emb_size, image_group_ids=None,
                    device=None, max_text_len=256):
    labels, valid_masks, _ = tokenize_caption_targets(tokenizer, captions, max_text_len)
    target_device = torch.device(device) if device is not None else labels.device
    image_group_ids = image_group_ids or list(range(len(anno_b)))
    targets = []
    for row, (anno, shape) in enumerate(zip(anno_b, shapes)):
        height, width = shape
        points = np.asarray(anno.get("points", []), dtype=np.float32).reshape(-1, 2)
        if points.size:
            points = points / np.asarray([width, height], dtype=np.float32)
        point_tensor = torch.as_tensor(points, dtype=torch.float32, device=target_device).reshape(-1, 2)
        n_points = point_tensor.shape[0]
        targets.append({
            "points": point_tensor,
            "labels": labels[row].expand(n_points, -1).clone().to(target_device),
            "valid_token_mask": valid_masks[row].to(target_device),
            "caption_size": valid_masks[row].sum().to(target_device),
            "image_group_id": int(image_group_ids[row]),
            "class_name": anno.get("class", ""),
            "attribute_name": anno.get("attribute", ""),
        })
    return targets


def distance_threshold_func(boxes):
    """Paper threshold; strict even median averages the two middle half-diagonals."""
    if not boxes:
        return 0.0
    ordered = sorted(boxes, key=lambda box: float(box[2]) * float(box[3]))
    middle = len(ordered) // 2
    candidates = ordered[middle:middle + 1] if len(ordered) % 2 else ordered[middle - 1:middle + 1]
    return float(sum(math.hypot(float(box[2]), float(box[3])) / 2.0
                     for box in candidates) / len(candidates))


def calc_loc_metric(pred_boxes, gt_points):
    pred_boxes = list(pred_boxes)
    gt_points = gt_points.detach().cpu().numpy() if torch.is_tensor(gt_points) else np.asarray(gt_points)
    gt_points = np.asarray(gt_points, dtype=np.float64).reshape(-1, 2)
    if not pred_boxes:
        return 0, 0, len(gt_points), 0.0, 0.0, 0.0
    if len(gt_points) == 0:
        return 0, len(pred_boxes), 0, 0.0, 0.0, 0.0
    pred_points = np.asarray([[box[0], box[1]] for box in pred_boxes], dtype=np.float64)
    cost_matrix = cdist(pred_points, gt_points, metric="euclidean")
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
    threshold = distance_threshold_func(pred_boxes)
    tp = sum(cost_matrix[p, g] < threshold for p, g in zip(pred_indices, gt_indices))
    fp, fn = len(pred_points) - tp, len(gt_points) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return tp, fp, fn, precision, recall, f1
