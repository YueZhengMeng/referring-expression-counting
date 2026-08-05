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

    @torch.no_grad()
    def forward(self, outputs, targets):
        logits = outputs["pred_logits"]
        points = outputs["pred_points"]
        indices = []
        for batch_index, target in enumerate(targets):
            target_points = target["points"].to(points.device)
            target_labels = target["labels"].to(logits.device)
            n_targets = target_points.shape[0]
            empty = torch.empty(0, dtype=torch.int64, device=logits.device)
            if n_targets == 0:
                indices.append((empty, empty.clone()))
                continue
            valid = target.get("valid_token_mask")
            if valid is None:
                valid = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
            valid = valid.to(device=logits.device, dtype=torch.bool)
            if valid.ndim != 1 or valid.numel() != logits.shape[-1]:
                raise ValueError("valid_token_mask must have shape [T]")
            if not valid.any():
                class_cost = logits[batch_index].new_zeros((logits.shape[1], n_targets))
            else:
                selected_logits = torch.nan_to_num(
                    logits[batch_index, :, valid], nan=0.0, posinf=1e4, neginf=-1e4
                )
                selected_labels = target_labels[:, valid]
                # BCE-with-logits without ever evaluating padding -inf values.
                element_cost = F.softplus(selected_logits[:, None, :]) \
                               - selected_logits[:, None, :] * selected_labels[None, :, :]
                class_cost = element_cost.mean(dim=-1)
            point_cost = torch.cdist(points[batch_index], target_points, p=1)
            cost = self.cost_class * class_cost + self.cost_point * point_cost
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
        values = outputs[key]
        chunks = [values[b, src] for b, (src, _) in enumerate(indices) if src.numel()]
        if not chunks:
            return values.new_empty((0,) + values.shape[2:]), []
        return torch.cat(chunks, dim=0), [src.numel() for src, _ in indices]

    def loss_label(self, outputs, targets, indices, **kwargs):
        logits = outputs["pred_logits"]
        matched_logits = []
        matched_labels = []
        valid_masks = []
        for batch_index, (src, tgt) in enumerate(indices):
            if not src.numel():
                continue
            valid = targets[batch_index].get("valid_token_mask")
            if valid is None:
                valid = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
            valid = valid.to(logits.device, dtype=torch.bool)
            labels = targets[batch_index]["labels"].to(logits.device)
            matched_logits.append(logits[batch_index, src])
            matched_labels.append(labels[tgt])
            valid_masks.append(valid.expand(src.numel(), -1))
        if not matched_logits:
            return {"loss_label": logits.sum() * 0}
        matched_logits = torch.cat(matched_logits)
        matched_labels = torch.cat(matched_labels)
        valid_masks = torch.cat(valid_masks)
        safe_logits = torch.where(valid_masks, matched_logits, torch.zeros_like(matched_logits))
        safe_labels = torch.where(valid_masks, matched_labels, torch.zeros_like(matched_labels))
        element = F.binary_cross_entropy_with_logits(safe_logits, safe_labels, reduction="none")
        per_query = element.masked_fill(~valid_masks, 0).sum(-1)
        per_query = per_query / valid_masks.sum(-1).clamp_min(1)
        return {"loss_label": per_query.mean()}

    def loss_point(self, outputs, targets, indices, **kwargs):
        pred = []
        truth = []
        for batch_index, (src, tgt) in enumerate(indices):
            if src.numel():
                pred.append(outputs["pred_points"][batch_index, src])
                truth.append(targets[batch_index]["points"][tgt].to(outputs["pred_points"].device))
        if not pred:
            return {"loss_point": outputs["pred_points"].sum() * 0}
        return {"loss_point": F.l1_loss(torch.cat(pred), torch.cat(truth), reduction="sum") / len(torch.cat(pred))}

    @staticmethod
    def _attribute_embedding(txt, mask):
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
        pooled = [self._attribute_embedding(txt_embs[i], token_masks[i]) for i in range(len(targets))]
        terms = []
        for i, (src, _) in enumerate(indices):
            if not src.numel() or pooled[i] is None:
                continue
            image_embedding = F.normalize(img_embs[i, src], dim=-1)
            positive = F.normalize(pooled[i].unsqueeze(0), dim=-1)[0]
            positive_logits = image_embedding @ positive
            negative_ids = [j for j in range(len(targets))
                            if j != i and groups[j] == groups[i]
                            and class_names[j] == class_names[i]
                            and attr_names[j] != attr_names[i]
                            and pooled[j] is not None]
            positive_term = -F.logsigmoid(positive_logits).sum()
            if negative_ids:
                negative = F.normalize(torch.stack([pooled[j] for j in negative_ids]), dim=-1)
                negative_logits = image_embedding @ negative.transpose(0, 1)
                negative_term = -F.logsigmoid(-negative_logits).sum()
            else:
                negative_term = positive_term.new_zeros(())
            terms.append((positive_term + negative_term) / src.numel())
        if not terms:
            return {"loss_contrast": img_embs.sum() * 0}
        return {"loss_contrast": torch.stack(terms).mean()}

    def forward(self, outputs, targets, mask_bi=None):
        indices = self.matcher(outputs, targets)
        losses = {}
        for loss in self.losses:
            if loss == "labels":
                losses.update(self.loss_label(outputs, targets, indices))
            elif loss == "points":
                losses.update(self.loss_point(outputs, targets, indices))
            else:
                losses.update(self.loss_contrast(
                    outputs, targets, indices, image_group_ids=mask_bi
                ))
        return losses
