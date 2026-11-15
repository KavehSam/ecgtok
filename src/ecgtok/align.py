"""Align beat patches to a frozen language-model input space.

TIME-LLM embeds short patches and cross-attends them to text prototypes taken
from the language model. Here the prototypes are a fixed sample of Qwen's
input rows. The attention that reads them is trained. The language model is
not asked to forecast the trace.
"""

from __future__ import annotations

import torch
from torch import nn

PATCH = 25
FRAMES = 13
PROTOTYPES = 64


class PatchAlign(nn.Module):
    """Map a beat window to one vector per patch, in the language-model width."""

    def __init__(self, hidden: int, heads: int = 4, prototypes: int = PROTOTYPES) -> None:
        """Build projection and cross-attention over fixed prototypes.

        Args:
            hidden: Language-model hidden size.
            heads: Number of attention heads.
            prototypes: Count of frozen prototype rows in ``self.prototypes``.

        Raises:
            ValueError: When ``hidden`` is not divisible by ``heads``.
        """
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by the number of heads")
        self.patch = PATCH
        self.frames = FRAMES
        self.embed = nn.Linear(PATCH, hidden)
        self.attend = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.register_buffer("prototypes", torch.zeros(prototypes, hidden))

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """Cross-attend 13 patches against the prototype bank.

        Args:
            windows: Batch of beat windows of shape ``(batch, window_len)``.

        Returns:
            Embeddings of shape ``(batch, 13, hidden)``.

        Raises:
            ValueError: When ``windows`` is shorter than ``13 * PATCH`` samples.
        """
        needed = self.frames * self.patch
        if windows.shape[-1] < needed:
            raise ValueError("window is shorter than the patch grid")
        patches = windows[..., :needed].reshape(windows.shape[0], self.frames, self.patch)
        query = self.embed(patches)
        memory = self.prototypes.unsqueeze(0).expand(windows.shape[0], -1, -1)
        mixed, _weights = self.attend(query, memory, memory, need_weights=False)
        return mixed