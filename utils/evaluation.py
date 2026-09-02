import random

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def seed_everything(seed):
    # 基础随机种子设置
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # GPU相关设置
    if torch.cuda.is_available():
        # 为当前GPU设置随机种子
        torch.cuda.manual_seed(seed)
        # 多GPU情况
        torch.cuda.manual_seed_all(seed)
    # 禁用 cuDNN 自动寻找最佳算法
    # 这可能会稍微降低性能，但对于可复现性至关重要
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # 实验发现这部分配置虽然影响可复现性，但会导致训练速度大幅变慢
    # 此外，即使使用这部分配置，也无法完全消除不确定性，所以不使用
    # 强制pytorch使用确定性算法
    # torch.use_deterministic_algorithms(True, warn_only=True)
    # 关闭两种不确定性attention算法——Flash Attention和Memory-Efficient Attention
    # Flash Attention在Windows平台无法安装，本来就没有启用
    # torch.backends.cuda.enable_flash_sdp(False)
    # 下面这一行会导致训练速度大幅变慢
    # torch.backends.cuda.enable_mem_efficient_sdp(False)

    # 设置环境变量
    # python为了安全考虑（防止哈希碰撞攻击），默认启用了哈希随机化
    # 这里需要关闭，确保程序每次运行时对相同对象的哈希值计算结果一致
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # 固定cuBLAS工作空间配置
    # cuBLAS是一个用于基础线性代数运算（BLAS），比如矩阵乘法的算子库
    # 强制 cuBLAS 始终使用固定大小的预分配工作空间，消除了动态内存分配带来的不确定性
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return seed


def _period_token_ids(tokenizer):
    """获取句号的token id，放入set中返回"""
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
    # 预设填充到 max_text_len
    labels = torch.zeros((len(captions), max_text_len), dtype=torch.float32)
    valid = torch.zeros((len(captions), max_text_len), dtype=torch.bool)
    period_ids = _period_token_ids(tokenizer)
    for row in range(len(captions)):
        # 获取非padding token的索引
        valid_positions = torch.where(attention[row])[0]
        # 默认以最后一个非padding token为结尾
        end = int(valid_positions[-1]) + 1 if valid_positions.numel() else 0
        # 获取句号的索引
        period_positions = [int(i) for i in valid_positions.tolist()
                            if int(input_ids[row, i]) in period_ids]
        if period_positions:
            # 有句号，则以句号为结尾
            end = period_positions[-1]
        if end:
            # 存在结尾，即存在有效token，记录区间（不包括这个结尾token）
            valid[row, :end] = True
            labels[row, :end] = 1.0
    # valid[row] == labels[row].bool()
    return labels, valid


def prepare_targets(anno_b, captions, shapes, tokenizer, image_group_ids=None,
                    device=None, max_text_len=256):
    labels, valid_masks = tokenize_caption_targets(tokenizer, captions, max_text_len)
    target_device = torch.device(device) if device is not None else labels.device
    image_group_ids = image_group_ids or list(range(len(anno_b)))
    targets = []
    for row, (anno, shape) in enumerate(zip(anno_b, shapes)):
        height, width = shape
        # 获取points坐标并进行归一化
        points = np.asarray(anno.get("points", []), dtype=np.float32).reshape(-1, 2)
        if points.size:
            points = points / np.asarray([width, height], dtype=np.float32)
        point_tensor = torch.as_tensor(points, dtype=torch.float32, device=target_device).reshape(-1, 2)
        n_points = point_tensor.shape[0]
        # 整理格式
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
    # 空预测时返回 0.0
    if not boxes:
        return 0.0
    # 按 width * height 排序
    # 使用 len(boxes) // 2 获取中间框的索引
    # 偶数个框时选择排序后中点右侧的框，而不是平均中点两侧的框
    areas = [float(box[2]) * float(box[3]) for box in boxes]
    median_index = int(np.argsort(areas)[len(areas) // 2])
    median_box = boxes[median_index]
    # 使用该单个框的宽高计算半对角线
    width = float(median_box[2])
    height = float(median_box[3])
    return float(np.sqrt(width ** 2 + height ** 2) / 2.0)


def calc_loc_metric(pred_boxes, gt_points):
    # 整理数据格式以适配scipy库的接口
    pred_boxes = list(pred_boxes)
    gt_points = gt_points.detach().cpu().numpy() if torch.is_tensor(gt_points) else np.asarray(gt_points)
    gt_points = np.asarray(gt_points, dtype=np.float64).reshape(-1, 2)
    if not pred_boxes:
        # 无预测，真实点全为FN
        return 0, 0, len(gt_points), 0.0, 0.0, 0.0
    if len(gt_points) == 0:
        # 无GT，模型预测全为FP
        return 0, len(pred_boxes), 0, 0.0, 0.0, 0.0
    pred_points = np.asarray([[box[0], box[1]] for box in pred_boxes], dtype=np.float64)
    # 计算欧式距离作为cost_matrix
    cost_matrix = cdist(pred_points, gt_points, metric="euclidean")
    # 使用 Hungarian 算法进行一对一匹配
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
    # 基于预测框面积的中位框决定TP阈值
    threshold = distance_threshold_func(pred_boxes)
    # 匹配距离小于阈值为TP
    tp = sum(cost_matrix[p, g] < threshold for p, g in zip(pred_indices, gt_indices))
    # 计算FP和FN
    # 预测框多余的为FP，GT框多余的为FN
    fp, fn = len(pred_points) - tp, len(gt_points) - tp
    # 计算精确率、召回率和F1分数
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return tp, fp, fn, precision, recall, f1
