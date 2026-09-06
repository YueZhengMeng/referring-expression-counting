import argparse
import csv
import os
import sys

import torch
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

sys.path.append('GroundingDINO')
from groundingdino.util.base_api import threshold, load_model

from utils.criterion import SetCriterion
from utils.evaluation import seed_everything, calc_loc_metric, prepare_targets
from utils.image_loader import get_loader
from utils.processor import DataProcessor


def freeze_encoders(model):
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    for parameter in model.bert.parameters():
        parameter.requires_grad_(False)
    return model


def set_finetuning_mode(model):
    model.train()
    model.backbone.eval()
    model.bert.eval()
    return model


def trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def cpu_state_dict(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


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


def train(model, loader, annotations, criterion, optimizer,
          lr_scheduler, device, epoch, text_threshold, box_threshold, token_threshold):
    print('Training on train set data')
    # 设置backbone和bert为eval模式，其他需要训练的部分为train模式
    model = set_finetuning_mode(model)

    train_mae = 0
    train_rmse = 0
    train_tp = 0
    train_fp = 0
    train_fn = 0
    counter = 0
    train_size = _caption_count(loader)

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

        # 重置梯度
        optimizer.zero_grad()

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
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 更新学习率
        lr_scheduler.step()

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
            train_mae += cnt_err
            train_rmse += cnt_err ** 2

            # 计算分类指标
            TP, FP, FN, precision, recall, f1 = calc_loc_metric(
                boxes, targets[b]['points']
            )
            train_tp += TP
            train_fp += FP
            train_fn += FN
            counter += 1
            """
            print(
                f'[train] ep {epoch} ({counter}/{train_size}), {img_caps[b]}, '
                f'caption: {captions[b]}, actual-predicted: {gt_cnt} vs {pred_cnt}, '
                f'error: {pred_cnt - gt_cnt}. Current Loss: {loss.item():.2f}, '
                f'MAE: {(train_mae / counter):.2f}, RMSE: {((train_rmse / counter) ** 0.5):.2f}, '
                f'TP = {TP}, FP = {FP}, FN = {FN}, precision = {precision:.2f}, '
                f'recall = {recall:.2f}, F1 = {f1:.2f}'
            )
            """
    return _finalize_metrics(
        train_mae, train_rmse, train_tp, train_fp, train_fn, counter
    )


def eval(model, loader, annotations, criterion, split, device, text_threshold,
         box_threshold, token_threshold, epoch=None):
    print(f'Evaluation on {split} set')
    # 设置整个模型为eval模式
    model.eval()

    eval_mae = 0
    eval_rmse = 0
    eval_tp = 0
    eval_fp = 0
    eval_fn = 0
    counter = 0
    eval_size = _caption_count(loader)

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
            """
            print(
                f'[{split}] ep {epoch} ({counter}/{eval_size}), {img_caps[b]}, '
                f'caption: {captions[b]}, actual-predicted: {gt_cnt} vs {pred_cnt}, '
                f'error: {pred_cnt - gt_cnt}. Current Loss: {loss.item():.2f}, '
                f'MAE: {(eval_mae / counter):.2f}, RMSE: {((eval_rmse / counter) ** 0.5):.2f}, '
                f'TP = {TP}, FP = {FP}, FN = {FN}, precision = {precision:.2f}, '
                f'recall = {recall:.2f}, F1 = {f1:.2f}'
            )
            """
    return _finalize_metrics(
        eval_mae, eval_rmse, eval_tp, eval_fp, eval_fn, counter
    )


def _write_stats_line(stats_file, values):
    formatted = [value if isinstance(value, str) else round(value, 4)
                 for value in values]
    with open(stats_file, 'a', newline='') as file:
        csv.writer(file).writerow(formatted)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train, validate, and test the referring-expression counting model'
    )
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
    parser.add_argument(
        '--pretrained-checkpoint', default='/home/pwb/pwb/checkpoints/GroundingDINO/groundingdino_swint_ogc.pth',
        help='Pretrained checkpoint used by the full model',
    )
    parser.add_argument('--image-dir', default='/home/pwb/pwb/rec-8k')
    parser.add_argument('--annotations', default='anno/annotations.json')
    parser.add_argument('--splits', default='anno/splits.json')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--early_stop_patience', type=int, default=5)
    parser.add_argument(
        '--batch_size', '--batch-size', dest='batch_size', type=int, default=1,
        help='Batch size for data loaders',
    )
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
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
    parser.add_argument(
        '--resume-checkpoint', default=None,
        help='Checkpoint containing model weights to resume from',
    )
    parser.add_argument('--stats_dir', default='./stats')
    args = parser.parse_args()

    # 设置随机种子
    seed_everything(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    processor = DataProcessor(args.image_dir, args.annotations, args.splits)
    annotations = processor.annotations
    loaders = {
        split: get_loader(processor, split, args.batch_size)
        for split in ('train', 'val', 'test')
    }
    print('Data loaded!')
    print(
        f"Train: {len(loaders['train'].dataset)} | "
        f"Val: {len(loaders['val'].dataset)} | "
        f"Test: {len(loaders['test'].dataset)}"
    )

    if args.resume_checkpoint:
        print(f'Loading resume checkpoint: {args.resume_checkpoint}')
        model = load_model(args.config, args.resume_checkpoint, device=device).to(device)
    else:
        model = load_model(args.config, args.pretrained_checkpoint, device=device).to(device)

    # 冻结backbone和bert
    model = freeze_encoders(model)
    # 损失函数
    criterion = SetCriterion()
    # 优化器
    optimizer = torch.optim.AdamW(
        trainable_parameters(model),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    # 计算训练步数
    steps_every_epoch = len(loaders['train'])
    total_train_steps = steps_every_epoch * int(args.epochs)
    # 定义带预热的线性递减学习率调度器
    # 从最大学习率的25分之一开始，在总步数的前10%逐渐线性增大到最大学习率，然后逐步线性递减到最大学习率的万分之一
    # 避免模型不收敛或者过拟合
    lr_scheduler = OneCycleLR(optimizer, max_lr=args.learning_rate, total_steps=total_train_steps,
                              anneal_strategy='linear', pct_start=0.1, div_factor=25.0, final_div_factor=10000.0)

    os.makedirs(args.stats_dir, exist_ok=True)
    stats_file = os.path.join(args.stats_dir, 'stats.csv')
    print(f'Saving stats to {stats_file}')
    header = [
        'train_mae', 'train_rmse', 'train_TP', 'train_FP', 'train_FN',
        'train_precision', 'train_recall', 'train_f1', '||',
        'val_mae', 'val_rmse', 'val_TP', 'val_FP', 'val_FN',
        'val_precision', 'val_recall', 'val_f1', '||',
        'test_mae', 'test_rmse', 'test_TP', 'test_FP', 'test_FN',
        'test_precision', 'test_recall', 'test_f1',
    ]
    with open(stats_file, 'w', newline='') as file:
        csv.writer(file).writerow(header)

    best_mae = float('inf')
    best_state = None
    early_stop_count = 0
    model_name = os.path.join(args.stats_dir, f'best_model.pth')
    for epoch in range(args.epochs):
        train_metrics = train(
            model, loaders['train'], annotations, criterion, optimizer, lr_scheduler, device,
            epoch, args.text_threshold, args.box_threshold, args.token_threshold,
        )
        val_metrics = eval(
            model, loaders['val'], annotations, criterion, 'val', device,
            args.text_threshold, args.box_threshold, args.token_threshold, epoch,
        )
        test_metrics = eval(
            model, loaders['test'], annotations, criterion, 'test', device,
            args.text_threshold, args.box_threshold, args.token_threshold, epoch,
        )
        val_mae = val_metrics[0]

        _write_stats_line(
            stats_file,
            list(train_metrics) + ['||'] + list(val_metrics) +
            ['||'] + list(test_metrics),
        )

        if best_mae > val_mae:
            best_mae = val_mae
            print(f'New best MAE: {best_mae}')
            best_state = cpu_state_dict(model)
            torch.save({'model': best_state, }, model_name)
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= args.early_stop_patience:
                print(f'Early stopping at epoch {epoch}')
                break

    if best_state is None or not os.path.isfile(model_name):
        raise RuntimeError(
            'No valid validation checkpoint was produced; cannot run final test.'
        )

    print(f'Inference on test set using best model: {model_name}')
    # 加载最佳模型，并设置为测试模式
    test_model = load_model(args.config, model_name, device=device).to(device)
    test_model = freeze_encoders(test_model)
    test_model.eval()
    # 测试集评估
    test_metrics = eval(
        test_model, loaders['test'], annotations, criterion, 'test', device,
        args.text_threshold, args.box_threshold, args.token_threshold, -1,
    )
    test_mae, test_rmse, test_TP, test_FP, test_FN, test_precision, test_recall, test_f1 = test_metrics
    print(
        f'test MAE: {test_mae:5.2f}, RMSE: {test_rmse:5.2f}, TP: {test_TP}, '
        f'FP: {test_FP}, FN: {test_FN}, precision: {test_precision:5.2f}, '
        f'recall: {test_recall:5.2f}, f1: {test_f1:5.2f}'
    )
    _write_stats_line(
        stats_file,
        [0] * 8 + ['||'] + [0] * 8 + ['||'] + list(test_metrics),
    )
