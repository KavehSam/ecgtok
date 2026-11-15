"""Beat windows, AAMI symbols, and neighboring R-R intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import firwin, lfilter

FS = 360
PRE_SAMPLES = 108
POST_SAMPLES = 216
FILTER_DELAY = 126

AAMI_N = frozenset({"N", "L", "R", "e", "j", "n", "B"})
AAMI_S = frozenset({"A", "a", "J", "S"})
AAMI_V = frozenset({"V", "E"})
AAMI_F = frozenset({"F"})
AAMI_Q = frozenset({"/", "f", "Q", "?"})
AAMI_NAMES = ("N", "S", "V", "F", "Q")


def slice_window(
    sig: np.ndarray,
    center: int,
    pre_samples: int = PRE_SAMPLES,
    post_samples: int = POST_SAMPLES,
) -> np.ndarray | None:
    """Samples from ``pre_samples`` before ``center`` through ``post_samples`` after it.

    Returns ``None`` when that span would leave the record.
    """
    start = int(center) - pre_samples
    end = int(center) + post_samples + 1
    values = np.asarray(sig)
    if start < 0 or end > len(values):
        return None
    return np.asarray(values[start:end])


def design_bandpass(fs: int = FS) -> tuple[np.ndarray, np.ndarray]:
    """FIR highpass at 0.6 Hz and lowpass at 40 Hz."""
    highpass = firwin(181, 0.6, pass_zero=False, fs=fs, window="hamming")
    lowpass = firwin(73, 40.0, pass_zero=True, fs=fs, window="hamming")
    return highpass, lowpass


def apply_bandpass(sig: np.ndarray, highpass: np.ndarray, lowpass: np.ndarray) -> np.ndarray:
    """Highpass, then lowpass. The caller shifts by the filter delay when slicing."""
    filtered = lfilter(lowpass, 1.0, lfilter(highpass, 1.0, np.asarray(sig, dtype=np.float64)))
    return np.asarray(filtered, dtype=np.float64)


def symbol_to_aami(symbol: str) -> int | None:
    """AAMI class index for a MIT-BIH beat symbol. ``None`` when it is not a QRS class."""
    token = symbol.strip()
    groups = ((AAMI_N, 0), (AAMI_S, 1), (AAMI_V, 2), (AAMI_F, 3), (AAMI_Q, 4))
    for symbols, label in groups:
        if token in symbols or token.upper() in symbols:
            return label
    return None


def rhythm_intervals(record_names: list[str], samples: list[int | None]) -> pd.DataFrame:
    """Previous R–R, next R–R, and the median of up to ten earlier intervals, in milliseconds."""
    previous = [np.nan] * len(record_names)
    following = [np.nan] * len(record_names)
    baseline = [np.nan] * len(record_names)
    grouped: dict[str, list[tuple[int, int]]] = {}
    for index, (record, sample) in enumerate(zip(record_names, samples, strict=True)):
        if sample is None:
            continue
        grouped.setdefault(str(record), []).append((int(sample), index))
    for pairs in grouped.values():
        _fill_record_rhythm(pairs, previous, following, baseline)
    return pd.DataFrame(
        {"rr_prev_ms": previous, "rr_next_ms": following, "rr_baseline_ms": baseline}
    )


def _fill_record_rhythm(
    pairs: list[tuple[int, int]],
    previous: list[float],
    following: list[float],
    baseline: list[float],
) -> None:
    ordered = sorted(pairs)
    sample_values = [sample for sample, _ in ordered]
    intervals = [
        (sample_values[pos] - sample_values[pos - 1]) * 1000.0 / FS
        for pos in range(1, len(sample_values))
    ]
    for pos, (_, row) in enumerate(ordered):
        if pos > 0:
            previous[row] = intervals[pos - 1]
        if pos < len(intervals):
            following[row] = intervals[pos]
        earlier = intervals[: pos - 1][-10:] if pos > 0 else []
        if len(earlier) >= 3:
            baseline[row] = float(np.median(earlier))
