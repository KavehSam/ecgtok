"""One token per named measurement."""

from __future__ import annotations

import numpy as np

from ecgtok.measures import beat_features, named_clauses


def named_beat_tokens(
    window: np.ndarray,
    *,
    fs: int,
    r_idx: int,
    rr_prev_ms: float | None,
    rr_next_ms: float | None,
    rr_baseline_ms: float | None,
) -> tuple[str, ...]:
    feat = beat_features(np.asarray(window, dtype=np.float64), fs=fs, r_idx=r_idx)
    clauses = named_clauses(
        feat,
        fs=fs,
        rr_prev_ms=rr_prev_ms,
        rr_next_ms=rr_next_ms,
        rr_baseline_ms=rr_baseline_ms,
    )
    return tuple(f"<{clause.upper()}>" for clause in clauses)
