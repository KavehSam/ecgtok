"""Pack token ids without inserting a space id between them."""

from __future__ import annotations


def resolve_id(token_id: int | None, unk_id: int | None) -> int:
    if token_id is None:
        if unk_id is None:
            raise ValueError("unseen token and no unk id")
        return int(unk_id)
    return int(token_id)


def pack_ids(token_ids: list[int], pad_id: int, max_length: int = 24) -> tuple[list[int], list[int]]:
    if max_length < 1:
        raise ValueError("max_length must be positive")
    kept = list(token_ids)[:max_length]
    mask = [1] * len(kept)
    missing = max_length - len(kept)
    return kept + [pad_id] * missing, mask + [0] * missing
