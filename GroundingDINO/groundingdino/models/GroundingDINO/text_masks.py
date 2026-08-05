import torch


def _content_token_ids(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    punctuation_ids = set()
    for punctuation in (".", "?", ","):
        punctuation_ids.update(tokenizer(punctuation, add_special_tokens=False)["input_ids"])
    return {int(token_id) for token_id in ids
            if int(token_id) not in special_ids and int(token_id) not in punctuation_ids}


def _role_mask(tokenizer, captions, role_texts, input_ids, offsets, attention, special):
    masks = torch.zeros_like(attention, dtype=torch.bool)
    for row, (caption, role_text) in enumerate(zip(captions, role_texts)):
        if not role_text:
            continue
        caption_body = caption.rstrip().rstrip(".")
        role_body = role_text.strip().rstrip(".")
        start = caption_body.lower().find(role_body.lower())
        if start >= 0 and role_body:
            end = start + len(role_body)
            row_offsets = offsets[row]
            for token_index, (token_start, token_end) in enumerate(row_offsets.tolist()):
                if token_end > token_start and token_start < end and token_end > start:
                    masks[row, token_index] = True
        if not masks[row].any():
            role_ids = _content_token_ids(tokenizer, role_text)
            if role_ids:
                for token_index, token_id in enumerate(input_ids[row].tolist()):
                    masks[row, token_index] = int(token_id) in role_ids
        masks[row] &= attention[row] & ~special[row]
    return masks
