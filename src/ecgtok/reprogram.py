"""Patch reprogramming into a frozen language-model space.

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


class PatchReprogram(nn.Module):
    """Map a beat window to one vector per patch, in the language-model width."""

    def __init__(self, hidden: int, heads: int = 4, prototypes: int = PROTOTYPES) -> None:
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by the number of heads")
        self.patch = PATCH
        self.frames = FRAMES
        self.embed = nn.Linear(PATCH, hidden)
        self.attend = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.register_buffer("prototypes", torch.zeros(prototypes, hidden))

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """Return ``(batch, 13, hidden)`` for a window that covers the 13 patches."""
        needed = self.frames * self.patch
        if windows.shape[-1] < needed:
            raise ValueError("window is shorter than the patch grid")
        patches = windows[..., :needed].reshape(windows.shape[0], self.frames, self.patch)
        query = self.embed(patches)
        memory = self.prototypes.unsqueeze(0).expand(windows.shape[0], -1, -1)
        mixed, _weights = self.attend(query, memory, memory, need_weights=False)
        return mixed
