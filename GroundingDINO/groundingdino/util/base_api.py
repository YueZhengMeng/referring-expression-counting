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


def _valid_caption_token_indices(tokenizer, encoded, punctuation_ids):
    """Return CLS and content-token positions from one encoded caption."""
    ids = torch.as_tensor(encoded["input_ids"])
    attention = torch.as_tensor(encoded["attention_mask"]).bool()
    special = encoded.get("special_tokens_mask")
    if special is None:
        special = torch.zeros_like(attention)
    else:
        special = torch.as_tensor(special).bool()

    # Accept either a single encoded row or a one-item batch.
    if ids.ndim > 1:
        ids = ids[0]
    if attention.ndim > 1:
        attention = attention[0]
    if special.ndim > 1:
        special = special[0]

    punctuation_ids = {int(token_id) for token_id in punctuation_ids}
    # 既不是 padding，又不是特殊 token 或句号的位置
    content = [int(i) for i in torch.where(attention & ~special)[0]
               if int(ids[i]) not in punctuation_ids]
    cls = 0
    # CLS 第一次出现的位置
    cls_token_id = getattr(tokenizer, "cls_token_id", None)
    if cls_token_id is not None and cls_token_id in ids.tolist():
        cls = ids.tolist().index(cls_token_id)
    return cls, content


def threshold(
        outputs,
        captions: list[str],
        tokenizer,
        max_text_len: int = 256,
        text_threshold: float = 0.25,
        box_threshold: float = 0.25,
        token_threshold: float = 0.35):

    bs = outputs["pred_logits"].shape[0]
    # Tokenize all captions once; the same encoded rows are reused below.
    tokenized_batch = tokenizer(
        captions,
        padding="longest",
        truncation=True,
        max_length=max_text_len,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    # 获取句号的token id
    punctuation = tokenizer(".", add_special_tokens=False)["input_ids"]
    punctuation_ids = punctuation[0] if punctuation and isinstance(punctuation[0], list) else punctuation

    ret = []
    for b in range(bs):
        prediction_logits = outputs["pred_logits"].detach().cpu().sigmoid()[b]
        prediction_boxes = outputs["pred_boxes"].detach().cpu()[b]
        # Keep a list-like tokenization for get_phrases_from_posmap decoding.
        tokenized = {
            key: value[b].tolist() if torch.is_tensor(value) else value[b]
            for key, value in tokenized_batch.items()
        }
        encoded = {
            key: value[b:b + 1] if torch.is_tensor(value) else value[b:b + 1]
            for key, value in tokenized_batch.items()
        }
        # 获取 cls token 的索引和其他有效 token 的索引
        cls_index, content_indices = _valid_caption_token_indices(
            tokenizer, encoded, punctuation_ids
        )
        # cls token 得分大于box_threshold阈值
        mask1 = prediction_logits[:, cls_index].gt(box_threshold)
        # 且其他有效 token 得分大于token_threshold阈值
        if content_indices:
            local_scores = prediction_logits[:, content_indices]
            mask2 = local_scores.gt(token_threshold).all(dim=1)
        else:
            mask2 = torch.ones(prediction_logits.shape[0], dtype=torch.bool)

        # 两个mask都为True的预测框才保留
        mask = mask1 & mask2
        logits = prediction_logits[mask]
        boxes = prediction_boxes[mask]

        # 防止没有预测框时 logits.max() 报错
        # 此时返回空列表
        if logits.numel() == 0:
            ret.append((boxes, prediction_logits.new_empty((0,)), []))
            continue

        # 获取预测框对应的文本，并去掉句号
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit in logits
        ]
        # 预测框、置信度、短语
        ret.append((boxes, logits.max(dim=1)[0], phrases))
    return ret
