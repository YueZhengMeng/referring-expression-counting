import copy
import os
from pathlib import Path

import torch

from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig

CHECKPOINT_FORMAT_VERSION = 1
TINY_OVERRIDES = {
    "hidden_dim": 64,
    "dim_feedforward": 256,
    "enc_layers": 1,
    "dec_layers": 1,
    "num_queries": 10,
    "nheads": 4,
    "num_feature_levels": 1,
    "return_interm_indices": [3],
    "two_stage_type": "no",
    "use_checkpoint": False,
    "use_transformer_ckpt": False,
}
COMPACT_OVERRIDES = {
    "hidden_dim": 256,
    "dim_feedforward": 1024,
    "enc_layers": 2,
    "dec_layers": 2,
    "num_queries": 300,
    "num_feature_levels": 4,
    "return_interm_indices": [1, 2, 3],
    "two_stage_type": "standard",
    "use_checkpoint": False,
    "use_transformer_ckpt": False,
}


def resolve_device(value="auto"):
    if value in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return result


def mode_overrides(mode):
    if mode == "full":
        return {}
    if mode == "compact_rec":
        return copy.deepcopy(COMPACT_OVERRIDES)
    if mode == "tiny":
        return copy.deepcopy(TINY_OVERRIDES)
    raise ValueError(f"Unknown model mode: {mode}")


def build_model_for_mode(config_path, mode="full", device="cpu", text_encoder=None,
                         extra_overrides=None):
    if mode == "auto":
        raise ValueError("model mode 'auto' requires checkpoint metadata")
    args = SLConfig.fromfile(config_path)
    overrides = mode_overrides(mode)
    if extra_overrides:
        overrides.update(extra_overrides)
    for key, value in overrides.items():
        setattr(args, key, value)
    if text_encoder:
        args.text_encoder_type = text_encoder
    args.device = str(device)
    args.anno_path = getattr(args, "anno_path", "./anno/annotations.json")
    model = build_model(args).to(device)
    return model, args, overrides


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


def make_checkpoint(model, mode, overrides=None, **metadata):
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_mode": mode,
        "config_overrides": copy.deepcopy(overrides or mode_overrides(mode)),
        "model": cpu_state_dict(model),
    }
    checkpoint.update(metadata)
    return checkpoint


def checkpoint_mode(checkpoint, requested="auto"):
    stored = checkpoint.get("model_mode")
    if stored is None:
        if requested == "auto":
            return "full"
        return requested
    if requested != "auto" and requested != stored:
        raise ValueError(f"Checkpoint was saved as '{stored}', not requested '{requested}'")
    return stored


def load_checkpoint(model, checkpoint_path, requested_mode="auto", strict=True):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    mode = checkpoint_mode(checkpoint, requested_mode)
    incompatible = model.load_state_dict(clean_state_dict(state), strict=strict)
    return checkpoint, mode, incompatible


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_pretrained(model, checkpoint_path):
    if not checkpoint_path:
        return model
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    return model
