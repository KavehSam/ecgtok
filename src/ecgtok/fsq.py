"""One-lead fixed-scale quantizer.

The idea follows HeartLLM: squash a latent, round each dimension to a few
levels, and pack the levels into one integer. This module reads a single
MLII beat, not a 12-lead record, and it does not pretrain the language model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

LEVELS = 5
LATENT = 4
FRAMES = 13
VOCAB = LEVELS**LATENT


def fsq_ids(latent: torch.Tensor, levels: int = LEVELS) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize latent vectors with straight-through rounding.

    Args:
        latent: Tensor whose last dimension is the code dimension.
        levels: Number of discrete levels per dimension.

    Returns:
        ``(ste_codes, token_ids)`` where ids use mixed-radix packing.

    Raises:
        ValueError: When ``levels`` is less than 2.
    """
    if levels < 2:
        raise ValueError("levels must be at least 2")
    scaled = torch.sigmoid(latent) * (levels - 1)
    rounded = torch.round(scaled)
    ste = scaled + (rounded - scaled).detach()
    digits = rounded.detach().to(dtype=torch.long).clamp(0, levels - 1)
    place = levels ** torch.arange(digits.shape[-1], device=digits.device)
    return ste, (digits * place).sum(dim=-1)


class BeatQuantizer(nn.Module):
    """Encode one beat as 13 quantized codes and reconstruct the window."""

    def __init__(self) -> None:
        """Initialize the 1-D convolutional encoder and decoder."""
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=5, padding=4),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(FRAMES),
        )
        self.to_latent = nn.Linear(64, LATENT)
        self.from_latent = nn.Linear(LATENT, 64)
        self.decoder = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct the input and emit integer FSQ codes.

        Args:
            windows: Batch of shape ``(batch, window_len)`` in millivolts.

        Returns:
            ``(reconstruction, ids)`` with ids shaped ``(batch, 13)``.
        """
        hidden = self.encoder(windows.unsqueeze(1)).transpose(1, 2)
        ste, ids = fsq_ids(self.to_latent(hidden), LEVELS)
        decoded = self.decoder(self.from_latent(ste).transpose(1, 2))
        wave = functional.interpolate(decoded, size=windows.shape[-1], mode="linear", align_corners=False)
        return wave.squeeze(1), ids


def codes_to_text(ids: torch.Tensor) -> list[str]:
    """Format quantized codes as tokenizer-ready strings.

    Args:
        ids: Integer codes of shape ``(batch, 13)``.

    Returns:
        One space-joined string per row, using ``<FSQ_n>`` tokens.
    """
    rows: list[str] = []
    for beat in ids.detach().cpu().tolist():
        rows.append(" ".join(f"<FSQ_{int(code)}>" for code in beat))
    return rows