import argparse
import csv
import os
import re
import sys
import textwrap

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.append('GroundingDINO')
from groundingdino.util.base_api import threshold, load_model

from utils.evaluation import seed_everything, calc_loc_metric, prepare_targets
from utils.image_loader import get_loader
from utils.processor import DataProcessor
from utils.criterion import SetCriterion

def sanitize_filename(s):
    """Sanitize a string to be safe for use as a filename."""
    return re.sub(r'[<>:"/\\|?*\s]', '_', s)


def freeze_encoders(model):
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    for parameter in model.bert.parameters():
        parameter.requires_grad_(False)
    return model


def _finalize_metrics(mae, rmse, tp, fp, fn, counter):
    if counter == 0:
        return 0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0
    mae /= counter
    rmse = (rmse / counter) ** 0.5
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return mae, rmse, tp, fp, fn, precision, recall, f1


def _caption_count(loader):
    return sum(len(tuples) for tuples in loader.dataset.img_cap_tuples)


def visualize_prediction(image_path, gt_points_pixel, pred_points_norm, caption, save_path):
    """Draw ground-truth (green circles) and predicted (red crosses) points on the image.

    Args:
        image_path: path to the original image file.
        gt_points_pixel: list of [x, y] in pixel coordinates.
        pred_points_norm: list of [x, y] in normalized [0, 1] coordinates.
        shape: (h, w) of the original image.
        caption: the referring-expression caption to display as title.
        save_path: where to save the resulting figure.
    """
    img = Image.open(image_path).convert("RGB")
    img = np.array(img)
    h, w = img.shape[0], img.shape[1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(img)

    # Ground-truth points — green circles
    if len(gt_points_pixel) > 0:
        gt_pts = np.array(gt_points_pixel)
        ax.scatter(gt_pts[:, 0], gt_pts[:, 1], c='lime', s=80, marker='o',
                   edgecolors='darkgreen', linewidths=1.5,
                   label=f'GT ({len(gt_pts)})', zorder=5)

    # Predicted points — red crosses (convert normalized → pixel)
    if len(pred_points_norm) > 0:
        pred_pts = np.asarray(pred_points_norm, dtype=np.float64).copy()
        if pred_pts.ndim != 2 or pred_pts.shape[1] < 2:
            raise ValueError(f"Expected predicted points with shape (N, 2), got {pred_pts.shape}")

        # 取出中心点坐标，去掉边界框宽高
        pred_pts = pred_pts[:, :2]
        # 去掉无效值点和图片边界外的点
        finite_mask = np.isfinite(pred_pts).all(axis=1)
        in_image_mask = ((pred_pts >= 0.0) & (pred_pts <= 1.0)).all(axis=1)
        valid_mask = finite_mask & in_image_mask
        if not valid_mask.all():
            print(f"  [WARN] ignoring {int((~valid_mask).sum())} invalid normalized predictions for {image_path}")
        # 取出有效点
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


def eval(model, loader, annotations, split, device, text_threshold,
         box_threshold, token_threshold, image_dir, output_dir):
    print(f"Inference on {split} set")
    # 设置整个模型为eval模式
    model.eval()

    eval_mae = 0
    eval_rmse = 0
    eval_tp = 0
    eval_fp = 0
    eval_fn = 0
    counter = 0
    eval_size = _caption_count(loader)

    # --- set up output directories ---
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "case_study_results.csv")

    # --- open CSV writer ---
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'image_id', 'caption', 'actual', 'predicted',
        'error', 'abs_error', 'vis_path'
    ])

    for images, captions, shapes, img_caps in tqdm(loader):
        # Keep the image-to-caption relation before flattening the batch.
        # 每个caption对应的image的id
        image_group_ids = [
            i for i, caption_list in enumerate(captions) for _ in caption_list
        ]
        # 每个caption对应的annotation点集
        anno_b = [
            annotations[img_cap]
            for img_cap_list in img_caps
            for img_cap in img_cap_list
        ]
        # 每个caption对应的image文件名
        img_caps = [
            img_cap for img_cap_list in img_caps for img_cap in img_cap_list
        ]
        # 每个caption对应的image的原始尺寸
        shapes = [
            shapes[i] for i, caption_list in enumerate(captions) for _ in caption_list
        ]
        # 每个caption对应的原始gt点集
        orig_gt_points = [list(anno['points']) for anno in anno_b]
        # 转移image到GPU上
        images = [
            images[i].to(device)
            for i, caption_list in enumerate(captions)
            for _ in caption_list
        ]
        # 将caption列表展平为一个列表
        captions = [
            caption for caption_list in captions for caption in caption_list
        ]

        with torch.no_grad():
            # 前向推理
            outputs = model(images, captions=captions)

        # 只保留中心点，放弃bbox的宽和高
        outputs['pred_points'] = outputs['pred_boxes'][:, :, :2]

        # 整理为适合损失函数的格式
        targets = prepare_targets(
            anno_b, captions, shapes, model.tokenizer,
            image_group_ids=image_group_ids, device=device,
            max_text_len=model.max_text_len,
        )

        # 计算损失
        loss = criterion(outputs, targets, image_group_ids)

        # 筛选预测框、置信度和对应的文本
        results = threshold(
            outputs, captions, model.tokenizer, model.max_text_len,
            text_threshold=text_threshold,
            box_threshold=box_threshold,
            token_threshold=token_threshold,
        )

        for b, (boxes, logits, phrases) in enumerate(results):
            # 计算计数指标
            boxes = [box.tolist() for box in boxes]
            # 预测的数量
            pred_cnt = len(boxes)
            # 实际的数量
            gt_cnt = len(targets[b]['points'])
            cnt_err = abs(pred_cnt - gt_cnt)
            eval_mae += cnt_err
            eval_rmse += cnt_err ** 2

            # 计算分类指标
            TP, FP, FN, precision, recall, f1 = calc_loc_metric(
                boxes, targets[b]['points']
            )
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
            # convert boxes to points (normalised cx, cy)
            points = [[box[0], box[1]] for box in boxes]
            try:
                visualize_prediction(image_path, orig_gt_points[b], points,
                                     captions[b], vis_path)
            except FileNotFoundError:
                vis_path = 'FILE_NOT_FOUND'
                print(f"  [WARN] image not found: {image_path}")

            # --- CSV row ---
            csv_writer.writerow([
                image_id, captions[b], gt_cnt, pred_cnt,
                pred_cnt - gt_cnt, cnt_err, vis_path
            ])
            """
            print(
                f'[{split}] ({counter}/{eval_size}), {img_caps[b]}, '
                f'caption: {captions[b]}, actual-predicted: {gt_cnt} vs {pred_cnt}, '
                f'error: {pred_cnt - gt_cnt}. Current Loss: {loss.item():.2f}, '
                f'MAE: {(eval_mae / counter):.2f}, RMSE: {((eval_rmse / counter) ** 0.5):.2f}, '
                f'TP = {TP}, FP = {FP}, FN = {FN}, precision = {precision:.2f}, '
                f'recall = {recall:.2f}, F1 = {f1:.2f}'
            )
            """
    csv_file.close()
    print(f"\nVisualizations saved to: {vis_dir}")
    print(f"CSV results saved to: {csv_path}")

    return _finalize_metrics(
        eval_mae, eval_rmse, eval_tp, eval_fp, eval_fn, counter
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference on train/val/test sets')
    parser.add_argument(
        '--seed', type=int,
        default=2025,
        help='Random seed',
    )
    parser.add_argument(
        '--config', type=str,
        default='GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
        help='Path to the GroundingDINO configuration file',
    )
    parser.add_argument('--image-dir', default='F:/REC-8K/rec-8k')
    parser.add_argument('--annotations', default='anno/annotations.json')
    parser.add_argument('--splits', default='anno/splits.json')
    parser.add_argument(
        '--batch_size', '--batch-size', dest='batch_size', type=int, default=3,
        help='Batch size for data loaders',
    )
    parser.add_argument(
        '--text_threshold', '--text-threshold', dest='text_threshold', type=float,
        default=0.25, help='Minimum text score for keeping a prediction',
    )
    parser.add_argument(
        '--box_threshold', '--box-threshold', dest='box_threshold', type=float,
        default=0.25, help='Minimum global score for keeping a prediction',
    )
    parser.add_argument(
        '--token_threshold', '--token-threshold', dest='token_threshold', type=float,
        default=0.35, help='Minimum local token score for keeping a prediction',
    )
    parser.add_argument('--stats_dir', default='./stats')
    parser.add_argument('--split', type=str,
                        default='test', choices=['train', 'val', 'test'], help='Split to evaluate on')
    args = parser.parse_args()

    # 设置随机种子
    seed_everything(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    """ data """
    processor = DataProcessor(args.image_dir, args.annotations, args.splits)
    annotations = processor.annotations
    image_dir = processor.get_image_path()

    split = args.split
    assert split in ['train', 'val', 'test']

    loader = get_loader(processor, split, args.batch_size)
    print("Data loaded!")
    print(f"{split}: {len(loader.dataset)}")

    """ model """
    print(f"Loading model from checkpoint: {args.stats_dir}")
    model = load_model(args.config, args.stats_dir, device=device)
    model = model.to(device)

    # 冻结backbone和bert
    model = freeze_encoders(model)

    # 损失函数
    criterion = SetCriterion()

    """ inference """
    output_dir = os.path.join(args.output_dir, split)
    mae, rmse, TP, FP, FN, precision, recall, f1 = eval(
        model, loader, annotations, criterion, split, device,
        args.text_threshold, args.box_threshold, args.token_threshold, image_dir, output_dir)
    print(
        f'test MAE: {mae:5.2f}, RMSE: {rmse:5.2f}, TP: {TP}, '
        f'FP: {FP}, FN: {FN}, precision: {precision:5.2f}, '
        f'recall: {recall:5.2f}, f1: {f1:5.2f}'
    )
