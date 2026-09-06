import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=5.0, cost_point=1.0, **kwargs):
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        if cost_class == 0 and cost_point == 0:
            raise ValueError("all matching costs cannot be zero")

    @torch.no_grad()  # 匈牙利匹配不可微，不需要计算梯度
    def forward(self, outputs, targets):
        logits = outputs["pred_logits"]
        points = outputs["pred_points"]
        indices = []
        for batch_index, target in enumerate(targets):
            # 取出每个target的 真实点集合 与 有效token标签
            target_points = target["points"].to(points.device)
            target_labels = target["labels"].to(logits.device)
            n_targets = target_points.shape[0]
            empty = torch.empty(0, dtype=torch.int64, device=logits.device)
            if n_targets == 0:
                indices.append((empty, empty.clone()))
                continue
            # 取出有效token集合，默认全为True
            valid = target.get("valid_token_mask")
            if valid is None:
                valid = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
            valid = valid.to(device=logits.device, dtype=torch.bool)
            if valid.ndim != 1 or valid.numel() != logits.shape[-1]:
                raise ValueError("valid_token_mask must have shape [T]")
            if not valid.any():
                # 没有有效token，则class_cost为0
                class_cost = logits[batch_index].new_zeros((logits.shape[1], n_targets))
            else:
                # 取出有效token对应的logits和标签
                selected_logits = torch.nan_to_num(
                    logits[batch_index, :, valid], nan=0.0, posinf=1e4, neginf=-1e4
                )
                selected_labels = target_labels[:, valid]
                # BCE-with-logits without ever evaluating padding -inf values.
                # selected_logits[:, None, :] : [Q, 1, Tv]
                # selected_labels[None, :, :] : [1, N, Tv]
                # 利用广播机制，二者相减后得到：
                # element_cost: [Q, N, Tv]
                # 等价于二分类交叉熵
                element_cost = F.softplus(selected_logits[:, None, :]) \
                               - selected_logits[:, None, :] * selected_labels[None, :, :]
                # 在 token 维度求平均
                # class_cost: [Q, N]
                class_cost = element_cost.mean(dim=-1)

            # 计算预测点与真实点之间的L1/曼哈顿距离
            point_cost = torch.cdist(points[batch_index], target_points, p=1)
            # 总成本 = 类别成本 + 点成本
            cost = self.cost_class * class_cost + self.cost_point * point_cost
            # 匈牙利算法求解最优匹配，返回源点和目标点对应的索引
            src, tgt = linear_sum_assignment(cost.detach().cpu().numpy())
            indices.append((
                torch.as_tensor(src, dtype=torch.int64, device=logits.device),
                torch.as_tensor(tgt, dtype=torch.int64, device=logits.device),
            ))
        return indices


class SetCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.matcher = HungarianMatcher(cost_class=5, cost_point=1)
        self.losses = ["labels", "points", "contrast"]
        self.weight_dict = {"loss_label": 5, "loss_point": 1, "loss_contrast": 0.06}

    @staticmethod
    def _matched(outputs, targets, indices, key):
        # 根据匹配结果，从 batch 中取出所有已匹配的 query
        values = outputs[key]
        chunks = [values[b, src] for b, (src, _) in enumerate(indices) if src.numel()]
        if not chunks:
            return values.new_empty((0,) + values.shape[2:]), []
        return torch.cat(chunks, dim=0), [src.numel() for src, _ in indices]

    def loss_label(self, outputs, targets, indices, **kwargs):
        logits = outputs["pred_logits"]
        # 未匹配的query保持全0，作为背景/no-object目标
        target_labels = torch.zeros_like(logits)
        valid_masks = []
        for batch_index, (src, tgt) in enumerate(indices):
            valid = targets[batch_index].get("valid_token_mask")
            if valid is None:
                valid = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
            valid = valid.to(logits.device, dtype=torch.bool)
            valid_masks.append(valid)
            if src.numel():
                labels = targets[batch_index]["labels"].to(logits.device)
                target_labels[batch_index, src] = labels[tgt]

        # 只在有效文本token上计算损失，避免padding位置的-inf参与计算
        valid_masks = torch.stack(valid_masks).unsqueeze(1).expand_as(logits)
        safe_logits = torch.where(valid_masks, logits, torch.zeros_like(logits))

        # 对全部query计算focal loss，使未匹配query得到负样本监督
        element = F.binary_cross_entropy_with_logits(
            safe_logits, target_labels, reduction="none"
        )
        probability = safe_logits.sigmoid()
        p_t = probability * target_labels + (1 - probability) * (1 - target_labels)
        alpha_t = 0.25 * target_labels + 0.75 * (1 - target_labels)
        element = alpha_t * ((1 - p_t) ** 2) * element
        element = element.masked_fill(~valid_masks, 0)

        # 每个caption按query数和有效token数归一化，再对batch求平均
        per_sample = element.sum(dim=(1, 2)) / valid_masks.sum(dim=(1, 2)).clamp_min(1)
        return {"loss_label": per_sample.mean()}

    def loss_point(self, outputs, targets, indices, **kwargs):
        pred = []
        truth = []
        for batch_index, (src, tgt) in enumerate(indices):
            # 取出已匹配的query的预测点和真实点
            if src.numel():
                pred.append(outputs["pred_points"][batch_index, src])
                truth.append(targets[batch_index]["points"][tgt].to(outputs["pred_points"].device))
        if not pred:
            return {"loss_point": outputs["pred_points"].sum() * 0}
        # 计算L1损失并求均值
        return {"loss_point": F.l1_loss(torch.cat(pred), torch.cat(truth), reduction="sum") / len(torch.cat(pred))}

    @staticmethod
    def _attribute_embedding(txt, mask):
        # 返回属性 token 的平均特征
        mask = mask.to(device=txt.device, dtype=torch.bool)
        if mask.ndim != 1 or mask.shape[0] != txt.shape[0] or not mask.any():
            return None
        return txt[mask].mean(dim=0)

    def loss_contrast(self, outputs, targets, indices, image_group_ids=None, **kwargs):
        img_embs = outputs["img_embs"]
        txt_embs = outputs["txt_embs"]
        token_masks = outputs.get("attribute_token_mask", outputs.get("token_masks"))
        if token_masks is None:
            return {"loss_contrast": img_embs.sum() * 0}
        groups = image_group_ids
        if groups is None:
            groups = [target.get("image_group_id", i) for i, target in enumerate(targets)]
        class_names = [target.get("class_name", "") for target in targets]
        attr_names = [target.get("attribute_name", "") for target in targets]
        # 计算属性嵌入
        pooled = [self._attribute_embedding(txt_embs[i], token_masks[i]) for i in range(len(targets))]
        terms = []
        # 遍历每个 caption 的匹配 query
        for i, (src, _) in enumerate(indices):
            if not src.numel() or pooled[i] is None:
                continue
            # 归一化 图片嵌入 和 属性嵌入
            image_embedding = F.normalize(img_embs[i, src], dim=-1)
            positive = F.normalize(pooled[i].unsqueeze(0), dim=-1)[0]
            # 计算正样本余弦相似度
            # image_embedding: [M_i, D]
            # positive:        [D]
            # positive_logits: [M_i]
            positive_logits = image_embedding @ positive
            # 正样本损失
            positive_term = -F.logsigmoid(positive_logits).sum()

            # 来自同一张图像、同一类别、不同属性的负样本
            negative_ids = [j for j in range(len(targets))
                            if j != i and groups[j] == groups[i]
                            and class_names[j] == class_names[i]
                            and attr_names[j] != attr_names[i]
                            and pooled[j] is not None]
            if negative_ids:
                # 计算负样本余弦相似度
                # image_embedding: [M_i, D]
                # negative.T:      [D, K]
                # negative_logits: [M_i, K]
                negative = F.normalize(torch.stack([pooled[j] for j in negative_ids]), dim=-1)
                negative_logits = image_embedding @ negative.transpose(0, 1)
                negative_term = -F.logsigmoid(-negative_logits).sum()
            else:
                negative_term = positive_term.new_zeros(())
            # 不同 caption 的目标数量不同时，损失大致按每个 query 平均
            terms.append((positive_term + negative_term) / src.numel())
        if not terms:
            return {"loss_contrast": img_embs.sum() * 0}
        # 返回所有 caption 的平均损失
        return {"loss_contrast": torch.stack(terms).mean()}

    def forward(self, outputs, targets, image_group_ids=None):
        indices = self.matcher(outputs, targets)
        losses = {}
        for loss in self.losses:
            if loss == "labels":
                losses.update(self.loss_label(outputs, targets, indices))
            elif loss == "points":
                losses.update(self.loss_point(outputs, targets, indices))
            else:
                losses.update(self.loss_contrast(
                    outputs, targets, indices, image_group_ids=image_group_ids
                ))
        # 加权求和
        weighted_losses = [
            losses[key] * self.weight_dict[key]
            for key in losses
            if key in self.weight_dict
        ]
        loss = torch.sum(torch.stack(weighted_losses))
        return loss
