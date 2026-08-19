from typing import Tuple

import numpy as np
import torch
from PIL import Image

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import get_phrases_from_posmap


def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    # 加载检查点（映射到 CPU 避免显存占用）
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    # clean_state_dict 清洗键名（若需要）
    # 加载参数到模型，strict=False 允许部分匹配
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


def _valid_caption_token_indices(tokenizer, caption):
    encoded = tokenizer(caption, return_tensors="pt", return_special_tokens_mask=True)
    ids = encoded["input_ids"][0]
    attention = encoded["attention_mask"][0].bool()
    special = encoded.get("special_tokens_mask", torch.zeros_like(attention.unsqueeze(0)))[0].bool()
    punctuation = tokenizer(".", add_special_tokens=False)["input_ids"]
    punctuation = set(punctuation[0] if punctuation and isinstance(punctuation[0], list) else punctuation)
    content = [int(i) for i in torch.where(attention & ~special)[0]
               if int(ids[i]) not in punctuation]
    cls = 0
    if getattr(tokenizer, "cls_token_id", None) in ids.tolist():
        cls = ids.tolist().index(tokenizer.cls_token_id)
    return cls, content


def threshold(
        outputs,
        captions: str,
        tokenizer,
        text_threshold: float=0.25,
        box_threshold: float = 0.25,
        token_threshold: float = 0.35):
    bs = outputs["pred_logits"].shape[0]

    ret = []
    for b in range(bs):
        prediction_logits = outputs["pred_logits"].detach().cpu().sigmoid()[b]
        prediction_boxes = outputs["pred_boxes"].detach().cpu()[b]
        tokenized = tokenizer(captions[b], return_tensors="pt", return_special_tokens_mask=True)
        cls_index, content_indices = _valid_caption_token_indices(tokenizer, captions[b])
        mask1 = prediction_logits[:, cls_index].gt(box_threshold)
        if content_indices:
            local_scores = prediction_logits[:, content_indices]
            mask2 = local_scores.gt(token_threshold).all(dim=1)
        else:
            mask2 = torch.ones(prediction_logits.shape[0], dtype=torch.bool)
        mask = mask1 & mask2

        logits = prediction_logits[mask]
        boxes = prediction_boxes[mask]
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit in logits
        ]
        ret.append((boxes, logits.max(dim=1)[0], phrases))
    return ret
