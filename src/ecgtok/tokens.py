"""Named measurements and the opaque energy codes of the same window."""

from __future__ import annotations

import numpy as np

from ecgtok.measures import beat_features, named_clauses

N_SEGMENTS = 12
MAX_TENTHS = 50


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


def opaque_beat_tokens(window: np.ndarray, n_segments: int = N_SEGMENTS) -> tuple[str, ...]:
    """RMS of equal slices, in tenths of a millivolt. No measurement names."""
    samples = np.asarray(window, dtype=np.float64).ravel()
    if samples.size == 0 or n_segments < 1:
        return ()
    edges = np.linspace(0, samples.size, n_segments + 1).astype(int)
    tokens: list[str] = []
    for segment, (start, end) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        chunk = samples[start:end]
        energy = 0.0 if chunk.size == 0 else float(np.sqrt(np.mean(np.square(chunk))))
        tenths = int(round(min(MAX_TENTHS / 10.0, max(0.0, energy)) * 10.0))
        tokens.append(f"<B{segment:02d}_{tenths}>")
    return tuple(tokens)
