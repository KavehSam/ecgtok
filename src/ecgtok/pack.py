"""Pack token ids without inserting a space id between them."""

from __future__ import annotations


def resolve_id(token_id: int | None, unk_id: int | None) -> int:
    """Normalize a tokenizer id, substituting the unknown id when needed.

    Args:
        token_id: Id returned by the tokenizer, or ``None`` when unseen.
        unk_id: Id to use for unknown tokens.

    Returns:
        A concrete integer id safe to place in a batch.

    Raises:
        ValueError: When ``token_id`` is ``None`` and ``unk_id`` is also ``None``.
    """
    if token_id is None:
        if unk_id is None:
            raise ValueError("unseen token and no unk id")
        return int(unk_id)
    return int(token_id)


def pack_ids(token_ids: list[int], pad_id: int, max_length: int = 24) -> tuple[list[int], list[int]]:
    """Right-pad a token id list to a fixed length.

    Args:
        token_ids: Ordered ids for one beat string.
        pad_id: Id used for padding positions.
        max_length: Maximum sequence length after padding.

    Returns:
        A pair ``(ids, mask)`` where mask is 1 on real tokens and 0 on padding.

    Raises:
        ValueError: When ``max_length`` is less than 1.
    """
    if max_length < 1:
        raise ValueError("max_length must be positive")
    kept = list(token_ids)[:max_length]
    mask = [1] * len(kept)
    missing = max_length - len(kept)
    return kept + [pad_id] * missing, mask + [0] * missing