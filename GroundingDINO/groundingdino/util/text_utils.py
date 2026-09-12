import torch


def build_text_logit_mask(tokenizer, tokenized):
    """Return the token positions used by the text classification head.

    The mask keeps the CLS objectness token and non-special tokens before the
    final period, while excluding padding, other special tokens, and the
    period itself.
    """
    input_ids = tokenized["input_ids"]
    attention = tokenized["attention_mask"].bool()
    special = tokenized.get("special_tokens_mask")
    if special is None:
        special = torch.zeros_like(attention)
    else:
        special = special.bool()

    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
        attention = attention.unsqueeze(0)
        special = special.unsqueeze(0)

    period_ids = tokenizer(".", add_special_tokens=False).get("input_ids", [])
    if period_ids and isinstance(period_ids[0], list):
        period_ids = period_ids[0]
    period_ids = {int(token_id) for token_id in period_ids}
    cls_token_id = getattr(tokenizer, "cls_token_id", None)
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    mask = torch.zeros_like(attention, dtype=torch.bool)

    for row in range(input_ids.shape[0]):
        valid_positions = torch.where(attention[row])[0]
        if not valid_positions.numel():
            continue

        period_positions = [
            int(position) for position in valid_positions.tolist()
            if int(input_ids[row, position]) in period_ids
        ]
        end = (max(period_positions) if period_positions
               else int(valid_positions[-1]))
        prefix = positions <= end if not period_positions else positions < end

        # Keep ordinary content tokens in the caption prefix.
        mask[row] = attention[row] & ~special[row] & prefix
        # CLS is the model's objectness token and is intentionally retained.
        if cls_token_id is not None:
            mask[row] |= attention[row] & (input_ids[row] == cls_token_id)

    return mask


def pad_text_logit_mask(mask, max_text_len):
    """Pad or truncate a token mask to the model's classification width."""
    if mask.shape[-1] > max_text_len:
        return mask[..., :max_text_len]
    if mask.shape[-1] < max_text_len:
        return torch.nn.functional.pad(mask, (0, max_text_len - mask.shape[-1]))
    return mask
