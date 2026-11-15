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
    """Use the learned row where one exists. Every other id stays on the frozen table."""
    gathered = functional.embedding(input_ids, base)
    mapped = row_of[input_ids]
    learned = functional.embedding(mapped.clamp(min=0), extra.to(dtype=base.dtype))
    return torch.where(mapped.unsqueeze(-1) >= 0, learned, gathered)
