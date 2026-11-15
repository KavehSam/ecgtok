"""Learned rows for new tokens. The pretrained rows stay fixed."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def overlay_special(
    base: torch.Tensor,
    extra: torch.Tensor,
    input_ids: torch.Tensor,
    row_of: torch.Tensor,
) -> torch.Tensor:
    """Blend frozen and learned embedding rows per token id.

    Args:
        base: Pretrained embedding table of shape ``(vocab, hidden)``.
        extra: Learned rows for special tokens, indexed by ``row_of``.
        input_ids: Batch of token indices.
        row_of: For each vocab id, the row in ``extra``, or -1 to use ``base``.

    Returns:
        Embeddings of shape ``(batch, seq, hidden)`` with specials taken from ``extra``.
    """
    gathered = functional.embedding(input_ids, base)
    mapped = row_of[input_ids]
    learned = functional.embedding(mapped.clamp(min=0), extra.to(dtype=base.dtype))
    return torch.where(mapped.unsqueeze(-1) >= 0, learned, gathered)