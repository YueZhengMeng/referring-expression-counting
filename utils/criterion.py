import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=5.0, cost_point=1.0, cost_rep=0.2, **kwargs):
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        self.cost_rep = cost_rep
        if cost_rep < 0:
            raise ValueError("cost_rep must be non-negative")
        if cost_class == 0 and cost_point == 0 and cost_rep == 0:
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

            if self.cost_rep > 0:
                # 找到同一张图、同一 class 下的其他 target，作为 Y_neg
                negative_points_list = [
                    other_target["points"].to(points.device)
                    for other_target in targets
                    if other_target is not target
                       and other_target["image_group_id"] == target["image_group_id"]
                       and other_target["class_name"] == target["class_name"]
                ]

                if negative_points_list:
                    negative_points = torch.cat(negative_points_list, dim=0)
                else:
                    negative_points = target_points.new_empty((0, target_points.shape[-1]))

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
                # 在有效 token 维度求平均，class_cost: [Q, N]
                class_cost = element_cost.mean(dim=-1)

            # 计算预测点与真实点之间的L1/曼哈顿距离
            point_cost = torch.cdist(points[batch_index], target_points, p=1)

            # SSM repulsive cost；cost_rep=0 时保持为零，即关闭该项
            repulsive_cost = points[batch_index].new_zeros(
                (points[batch_index].shape[0], n_targets)
            )
            if self.cost_rep > 0 and negative_points.numel() > 0:
                # [Q, N_neg] -> [Q]
                nearest_negative_dist = torch.cdist(
                    points[batch_index], negative_points, p=2
                ).min(dim=1).values

                # [Q, N]
                positive_dist = torch.cdist(
                    points[batch_index], target_points, p=2
                )

                # R(p_hat_i, p_j)
                ambiguity_ratio = nearest_negative_dist[:, None] / positive_dist.clamp_min(1e-6)

                # C_rep(p_hat_i, p_j) = exp(-R)
                repulsive_cost = torch.exp(-ambiguity_ratio)

            # 总成本 = 类别成本 + 点成本 + SSM repulsive cost
            cost = self.cost_class * class_cost + self.cost_point * point_cost + self.cost_rep * repulsive_cost

            # 匈牙利算法求解最优匹配，返回源点和目标点对应的索引
            src, tgt = linear_sum_assignment(cost.detach().cpu().numpy())
            indices.append((
                torch.as_tensor(src, dtype=torch.int64, device=logits.device),
                torch.as_tensor(tgt, dtype=torch.int64, device=logits.device),
            ))
        return indices


class SetCriterion(nn.Module):
    def __init__(self, use_contrast_rec=False, contrast_rec_weight=0.06,
                 use_contrast_attr=False, contrast_attr_weight=0.06,
                 use_contrast_asd=False, contrast_asd_weight=0.015,
                 use_contrast_img_hm=False, contrast_img_hm_weight=0.005,
                 use_contrast_img=False, contrast_img_weight=0.005,
                 cost_rep=0.2):
        super().__init__()
        self.matcher = HungarianMatcher(cost_class=5, cost_point=1, cost_rep=cost_rep)
        self.losses = ["labels", "points", "contrast"]
        self.weight_dict = {"loss_label": 5, "loss_point": 1}

        # 三种 text 对比最多只选一种
        if sum(map(bool, (use_contrast_rec, use_contrast_asd, use_contrast_attr))) > 1:
            raise ValueError("only one text contrast can be used")
        self.use_contrast_rec = use_contrast_rec
        self.use_contrast_asd = use_contrast_asd
        self.use_contrast_attr = use_contrast_attr
        if use_contrast_rec:
            self.weight_dict["loss_contrast_rec"] = contrast_rec_weight
        elif self.use_contrast_attr:
            self.weight_dict["loss_contrast_attr"] = contrast_attr_weight
        elif self.use_contrast_asd:
            self.weight_dict["loss_contrast_asd"] = contrast_asd_weight
        # C-REX 论文/参考 README 里真正使用的目标（Hungarian 匹配引导的 SupCon）
        self.use_contrast_img_hm = use_contrast_img_hm
        # 是否使用 C-REX 图像对比（文本引导的版本）
        self.use_contrast_img = use_contrast_img
        if self.use_contrast_img_hm:
            self.weight_dict["loss_contrast_img_hm"] = contrast_img_hm_weight
        elif self.use_contrast_img:
            self.weight_dict["loss_contrast_img"] = contrast_img_weight

    @staticmethod
    def _matched(outputs, targets, indices, key):
        # 根据匹配结果，从 batch 中取出所有已匹配的 query
        values = outputs[key]
        chunks = [values[b, src] for b, (src, _) in enumerate(indices) if src.numel()]
        if not chunks:
            return values.new_empty((0,) + values.shape[2:]), []
        return torch.cat(chunks, dim=0), [src.numel() for src, _ in indices]

    def loss_label(self, outputs, targets, indices, alpha=0.25, gamma=2, **kwargs):
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
        alpha_t = alpha * target_labels + (1 - alpha) * (1 - target_labels)
        element = alpha_t * ((1 - p_t) ** gamma) * element
        element = element.masked_fill(~valid_masks, 0)

        # 每个 caption 按实际参与监督的 query-token 对数归一化，再对 batch 求平均
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
        # 计算所有实际匹配对象的平均 L1 定位损失
        matched_pred = torch.cat(pred)
        matched_truth = torch.cat(truth)
        num_matches = matched_pred.shape[0]
        return {"loss_point": F.l1_loss(matched_pred, matched_truth, reduction="sum") / max(num_matches, 1)}

    @staticmethod
    def _attribute_embedding(txt, mask):
        # 返回属性 token 的平均特征
        mask = mask.to(device=txt.device, dtype=torch.bool)
        if mask.ndim != 1 or mask.shape[0] != txt.shape[0] or not mask.any():
            return None
        return txt[mask].mean(dim=0)

    def loss_contrast_attr(self, outputs, targets, indices, image_group_ids=None, **kwargs):
        img_embs = outputs["img_embs"]
        txt_embs = outputs["txt_embs"]
        # 取出属性 token 的 mask
        token_masks = outputs["attribute_token_mask"]
        if token_masks is None:
            return {"loss_contrast_attr": img_embs.sum() * 0}
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
            return {"loss_contrast_attr": img_embs.sum() * 0}
        # 返回所有 caption 的平均损失
        return {"loss_contrast_attr": torch.stack(terms).mean()}

    def loss_contrast_rec(self, outputs, targets, indices, image_group_ids=None, **kwargs):
        """GroundingREC image-text contrastive loss using matched image tokens."""
        img_embs = outputs["img_embs"]
        txt_embs = outputs["txt_embs"]

        # 取所有有效 token 的 mask
        token_masks = outputs["text_token_mask"]
        if token_masks is None:
            return {"loss_contrast_rec": img_embs.sum() * 0}

        terms = []
        for batch_index, (src, _) in enumerate(indices):
            if not src.numel():
                continue

            mask = token_masks[batch_index].to(device=txt_embs.device, dtype=torch.bool)
            if mask.ndim != 1 or mask.shape[0] != txt_embs.shape[1] or not mask.any():
                continue
            # 当前 RE 匹配的 image token
            matched_img_embs = F.normalize(img_embs[batch_index, src], p=2, dim=-1)
            # 当前 RE 有效 token 的平均特征向量
            positive_txt = F.normalize(txt_embs[batch_index, mask].mean(dim=0), p=2, dim=-1)
            # 计算正样本余弦相似度
            positive_logits = matched_img_embs @ positive_txt

            # Uses other captions of the same image as text negatives.
            # 同一张图像的其他 caption 作为 text 负样本
            group = (image_group_ids[batch_index]
                     if image_group_ids is not None else
                     targets[batch_index].get("image_group_id", batch_index))
            negative_embeddings = []
            for j, target in enumerate(targets):
                other_group = (image_group_ids[j]
                               if image_group_ids is not None else
                               target.get("image_group_id", j))
                if j == batch_index or other_group != group:
                    continue
                other_mask = token_masks[j].to(device=txt_embs.device, dtype=torch.bool)
                if other_mask.ndim == 1 and other_mask.shape[0] == txt_embs.shape[1] and other_mask.any():
                    negative_embeddings.append(txt_embs[j, other_mask].mean(dim=0))

            # The original loss is defined only when a same-image negative caption exists
            # Otherwise this sample contributes zero.
            if not negative_embeddings:
                continue

            # 计算负样本余弦相似度
            negative_txt = F.normalize(torch.stack(negative_embeddings), p=2, dim=-1)
            negative_logits = matched_img_embs @ negative_txt.transpose(0, 1)

            # 正样本损失和负样本损失的平均值
            positive_term = -F.logsigmoid(positive_logits).sum()
            negative_term = -F.logsigmoid(-negative_logits).sum()
            terms.append((positive_term + negative_term) / src.numel())

        if not terms:
            return {"loss_contrast_rec": img_embs.sum() * 0}
        return {"loss_contrast_rec": torch.stack(terms).mean()}

    def loss_contrast_img(self, outputs, targets, indices, image_group_ids=None,
                          temperature=0.07, **kwargs):
        """
        C-REX
        文本引导的正样本选择
        """
        assert "img_embs" in outputs and "txt_embs" in outputs

        img_embs = outputs["img_embs"]
        txt_embs = outputs["txt_embs"]
        # token_masks = outputs["text_token_mask"]
        # 与参考实现保持一致：文本端池化用的是 attribute token mask，
        token_masks = outputs["attribute_token_mask"]
        if token_masks is None:
            return {"loss_contrast_img": img_embs.sum() * 0}
        # 每个 caption 的 GT 数量
        num_points_per_cap = torch.tensor(
            [target["points"].shape[0] for target in targets],
            device=img_embs.device,
        )

        # 计算每个 image query 与 caption 属性平均向量的余弦相似度
        txt_emb_sum = torch.sum(txt_embs * token_masks.unsqueeze(-1), dim=1)
        # clamp_min(1)：参考实现的属性 mask 用 token-id 集合匹配，至少含 [CLS]/[SEP] 不会为空；
        # 本仓库的 _role_mask 在属性文本未命中时可能全为 False，这里避免除零得到 NaN
        txt_embs = txt_emb_sum / token_masks.sum(dim=1, keepdim=True).clamp_min(1)
        img_embs_norm = F.normalize(img_embs, p=2, dim=2)
        txt_embs_norm = F.normalize(txt_embs, p=2, dim=1)
        token_text_sim = torch.matmul(img_embs_norm, txt_embs_norm.unsqueeze(2)).squeeze(-1)

        idx = []
        t_n = []
        # 根据文本相似度选择 top-k 正样本
        for i in range(token_text_sim.shape[0]):
            # 选择数量等于真实目标数量，但最多 900 个
            _, top_n_indices = torch.topk(
                token_text_sim[i], min(900, num_points_per_cap[i].item()), dim=-1
            )
            # 生成正样本 mask
            t_ = torch.zeros_like(img_embs[0, :, 0], dtype=torch.int)
            t_.scatter_(0, top_n_indices, 1)
            t_n.append(top_n_indices)
            idx.append(t_)

        t_ = torch.stack(idx, 0)

        # 计算 query 与 query 之间的温度缩放余弦相似度
        logits = torch.matmul(
            img_embs_norm, img_embs_norm.permute(0, 2, 1)
        ) / temperature

        # 对每个 anchor，从分母中去掉其自身
        num_queries = logits.shape[-1]
        self_mask = torch.eye(
            num_queries, device=img_embs.device, dtype=torch.bool
        ).unsqueeze(0)
        log_denominator = torch.logsumexp(
            logits.masked_fill(self_mask, float("-inf")),
            dim=-1,
        )

        # 对每个 anchor，只在文本相似度选出的正 query 上计算分子，并排除自身
        positive_mask = t_.to(dtype=torch.bool).unsqueeze(1).expand(-1, num_queries, -1)
        positive_mask = positive_mask & ~self_mask
        positive_logits = logits.masked_fill(~positive_mask, float("-inf"))
        positive_logsumexp = torch.logsumexp(positive_logits, dim=-1)

        ll = []
        for i in range(token_text_sim.shape[0]):
            num_positives = t_n[i].numel()
            if num_positives < 2:
                # 一个正样本无法构成正样本对
                ll.append(logits[i].sum() * 0)
                continue

            # -log(mean positive probability)
            anchor_log_denominator = torch.gather(log_denominator[i], 0, t_n[i])
            anchor_positive_logsumexp = torch.gather(positive_logsumexp[i], 0, t_n[i])
            normalizer = torch.log(img_embs.new_tensor(float(num_positives - 1)))
            ll.append(
                (anchor_log_denominator - anchor_positive_logsumexp + normalizer).mean()
            )

        loss = torch.stack(ll).mean()

        losses = {}
        losses["loss_contrast_img"] = loss

        return losses

    def loss_contrast_img_hm(self, outputs, targets, indices, temperature=0.07, **kwargs):
        """C-REX image-space supervised contrastive loss.
        Hungarian 匹配引导的正样本选择
        Matched decoder queries are positive samples and all remaining queries from the same image are negatives.
        Only positive queries are used as anchors, as in the modified SupCon objective from C-REX.
        """
        img_embs = outputs["img_embs"]  # [B, num_queries, embedding_dim]
        temperature = max(float(temperature), 1e-6)
        num_queries = img_embs.shape[1]

        terms = []
        for batch_index, (src, _) in enumerate(indices):
            # At least two positive queries are needed to form a positive pair.
            # 其中一个是 anchor，至少还需要一个
            if src.numel() < 2:
                continue
            # 转移索引设备，并过滤掉无效的索引
            src = src.to(device=img_embs.device, dtype=torch.long)
            src = src[(src >= 0) & (src < num_queries)]
            src = torch.unique(src)
            if src.numel() < 2:
                continue

            # logits[i, j] 表示第 i 个正 query 与第 j 个 query 的温度缩放余弦相似度
            embeddings = F.normalize(img_embs[batch_index], dim=-1)
            logits = embeddings[src] @ embeddings.transpose(0, 1)
            logits = logits / temperature  # [num_positive, num_queries]

            # Remove each anchor's self-similarity from the denominator.
            # 去掉 anchor 自身
            anchor_rows = torch.arange(src.numel(), device=img_embs.device)
            self_mask = torch.zeros_like(logits, dtype=torch.bool)
            self_mask[anchor_rows, src] = True
            # log-sum-exp：计算每个 anchor 的除自身外所有 query 的 log 概率和
            log_denominator = torch.logsumexp(
                logits.masked_fill(self_mask, float("-inf")),
                dim=-1,
            )

            # All other matched queries are positives; unmatched queries are negatives.
            # Negative tokens are not pulled toward each other.
            # 标记正样本
            positive_mask = torch.zeros_like(logits, dtype=torch.bool)
            positive_mask[:, src] = True
            positive_mask &= ~self_mask
            # 负样本不聚合
            positive_logits = logits.masked_fill(~positive_mask, float("-inf"))
            positive_logsumexp = torch.logsumexp(positive_logits, dim=-1)

            # Eq. (2): -log(mean positive probability).
            # 按每个 anchor 的正样本数量求平均
            normalizer = torch.log(img_embs.new_tensor(float(src.numel() - 1)))
            terms.append((log_denominator - positive_logsumexp + normalizer).mean())

        if not terms:
            return {"loss_contrast_img_hm": img_embs.sum() * 0}
        return {"loss_contrast_img_hm": torch.stack(terms).mean()}

    def loss_contrast_asd(self, outputs, targets, indices, image_group_ids=None, **kwargs):
        """
        Attribute Semantic Discrimination (ASD) loss:
            L_attr = (L_i2t_cons + L_t2i_cons) / 2

        img_embs:    [B, Nq, D]    - all decoder queries
        txt_embs:    [B, W, D]     - text token embeddings
        pred_points: [B, Nq, 2]    - predicted normalized coordinates
        targets[i]["points"]: [N, 2] - normalized GT coordinates
        indices[i][0]: matched query indices from Hungarian matching
        """

        img_embs = outputs["img_embs"]
        txt_embs = outputs["txt_embs"]
        pred_points = outputs["pred_points"]
        # 取出所有有效 token 的 mask
        text_token_masks = outputs["text_token_mask"]

        if text_token_masks is None:
            return {"loss_contrast_asd": img_embs.sum() * 0}

        # Hyperparameters from the paper / default setting.
        sigma = 0.35  # semantic threshold
        gamma = 16.0  # distance threshold in pixels
        tau = 0.07  # temperature; paper does not report it

        if image_group_ids is None:
            groups = [target.get("image_group_id", i) for i, target in enumerate(targets)]
        else:
            groups = image_group_ids

        class_names = [target.get("class_name", "") for target in targets]
        attr_names = [target.get("attribute_name", "") for target in targets]

        # ------------------------------------------------------------
        # Text representations:
        # Tf       : all valid token embeddings, [W, D]
        # Tf_bar   : average pooled text embedding, [D]
        # ------------------------------------------------------------
        token_texts, pooled_texts = [], []

        for i in range(len(targets)):
            mask = text_token_masks[i].bool()
            Tf = txt_embs[i][mask]

            if Tf.numel() == 0:
                token_texts.append(None)
                pooled_texts.append(None)
            else:
                token_texts.append(Tf)
                pooled_texts.append(Tf.mean(dim=0))

        # ------------------------------------------------------------
        # Image-to-Text contrastive loss, Eq. (13)
        # ------------------------------------------------------------
        i2t_terms = []

        for i, (src, _) in enumerate(indices):
            if not src.numel() or pooled_texts[i] is None:
                continue

            S_m = F.normalize(img_embs[i, src], dim=-1)  # [Nm, D]
            Tf_bar = F.normalize(pooled_texts[i], dim=-1)  # [D]
            positive_logits = (S_m @ Tf_bar) / tau  # [Nm]

            negative_ids = [j for j in range(len(targets))
                            if j != i and groups[j] == groups[i]
                            and class_names[j] == class_names[i]
                            and attr_names[j] != attr_names[i]
                            and pooled_texts[j] is not None]

            if not negative_ids:
                continue

            Tf_h_bar = F.normalize(torch.stack([pooled_texts[j] for j in negative_ids]), dim=-1)  # [M, D]
            negative_logits = (S_m @ Tf_h_bar.transpose(0, 1)) / tau  # [Nm, M]

            logits = torch.cat([positive_logits.unsqueeze(-1), negative_logits], dim=-1)
            i2t_terms.append((torch.logsumexp(logits, dim=-1) - positive_logits).mean())

        loss_i2t = torch.stack(i2t_terms).mean() if i2t_terms else img_embs.sum() * 0

        # ------------------------------------------------------------
        # Text-to-Image contrastive loss, Eqs. (14)-(16)
        # ------------------------------------------------------------
        t2i_terms = []

        for i, (src, _) in enumerate(indices):
            if not src.numel() or pooled_texts[i] is None:
                continue

            num_query = img_embs.shape[1]
            unmatched_mask = torch.ones(num_query, dtype=torch.bool, device=img_embs.device)
            unmatched_mask[src] = False
            unmatched_idx = unmatched_mask.nonzero(as_tuple=False).flatten()

            if not unmatched_idx.numel():
                continue

            # S_um and its corresponding predicted coordinates R'_um.
            S_um = img_embs[i, unmatched_idx]  # [Num, D]
            R_um = pred_points[i, unmatched_idx]  # [Num, 2]

            # Eq. (14): C_conf = sigmoid(S_um @ Tf^T).
            Tf = token_texts[i]
            S_um_norm = F.normalize(S_um, dim=-1)
            Tf_norm = F.normalize(Tf, dim=-1)
            C_confidence = torch.sigmoid(S_um_norm @ Tf_norm.transpose(0, 1))

            semantic_mask = C_confidence.min(dim=1).values > sigma
            if not semantic_mask.any():
                continue

            S_um = S_um[semantic_mask]
            R_um = R_um[semantic_mask]

            # Collect GT points from the same image, same class, different attributes.
            gt_points_list = []
            for j in range(len(targets)):
                if (j == i or groups[j] != groups[i] or class_names[j] != class_names[i] or
                        attr_names[j] == attr_names[i]):
                    continue

                points = targets[j].get("points", None)
                if points is not None and len(points) > 0:
                    points = torch.as_tensor(points, dtype=R_um.dtype, device=R_um.device)
                    gt_points_list.append(points)

            if not gt_points_list:
                continue

            R_gt_h = torch.cat(gt_points_list, dim=0)  # [Nh, 2]

            # Convert normalized coordinates to pixel coordinates.
            shape = targets[i]["shape"]
            shape = torch.as_tensor(shape, device=R_um.device, dtype=R_um.dtype)
            H, W = shape[0], shape[1]
            scale = torch.stack([W, H])

            R_um_pixel = R_um * scale
            R_gt_h_pixel = R_gt_h * scale

            # Eq. (15): select queries whose prediction is within 16 pixels
            # of at least one GT object having a different attribute.
            distance_matrix = torch.cdist(R_um_pixel, R_gt_h_pixel, p=2)
            hard_negative_mask = distance_matrix.min(dim=1).values < gamma

            if not hard_negative_mask.any():
                continue

            S_um_h = S_um[hard_negative_mask]  # [N_um^h, D]

            # Eq. (16): positive visual samples are the Hungarian-matched
            # queries S_m, while S_um_h are hard-negative visual samples.
            S_m = F.normalize(img_embs[i, src], dim=-1)  # [Nm, D]
            S_um_h = F.normalize(S_um_h, dim=-1)  # [Nh, D]
            Tf_bar = F.normalize(pooled_texts[i], dim=-1)  # [D]

            positive_logits = (S_m @ Tf_bar) / tau  # [Nm]
            negative_logits = (S_m @ S_um_h.transpose(0, 1)) / tau  # [Nm, Nh]

            logits = torch.cat([positive_logits.unsqueeze(-1), negative_logits], dim=-1)
            t2i_terms.append((torch.logsumexp(logits, dim=-1) - positive_logits).mean())

        loss_t2i = torch.stack(t2i_terms).mean() if t2i_terms else img_embs.sum() * 0

        # Eq. (17)
        loss_attr = (loss_i2t + loss_t2i) / 2

        return {"loss_contrast_asd": loss_attr}

    def forward(self, outputs, targets, image_group_ids=None):
        indices = self.matcher(outputs, targets)
        losses = {}
        for loss in self.losses:
            if loss == "labels":
                losses.update(self.loss_label(outputs, targets, indices))
            elif loss == "points":
                losses.update(self.loss_point(outputs, targets, indices))
            else:
                if self.use_contrast_rec:
                    losses.update(self.loss_contrast_rec(
                        outputs, targets, indices, image_group_ids=image_group_ids
                    ))
                elif self.use_contrast_attr:
                    losses.update(self.loss_contrast_attr(
                        outputs, targets, indices, image_group_ids=image_group_ids
                    ))
                elif self.use_contrast_asd:
                    losses.update(self.loss_contrast_asd(
                        outputs, targets, indices, image_group_ids=image_group_ids
                    ))

                if self.use_contrast_img_hm:
                    losses.update(self.loss_contrast_img_hm(outputs, targets, indices))
                elif self.use_contrast_img:
                    losses.update(self.loss_contrast_img(outputs, targets, indices))
        # 加权求和
        weighted_losses = [
            losses[key] * self.weight_dict[key]
            for key in losses
            if key in self.weight_dict
        ]
        loss = torch.sum(torch.stack(weighted_losses))
        return loss
