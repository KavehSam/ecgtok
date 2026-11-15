"""P, QRS, and T fiducials on one beat window."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


def ms_to_samples(ms: float, fs: int) -> int:
    """Convert a duration to an integer sample count.

    Args:
        ms: Duration in milliseconds.
        fs: Sample rate in hertz.

    Returns:
        Rounded sample count at ``fs``.
    """
    return int(round(ms / 1000.0 * fs))


@dataclass
class BeatFiducials:
    """Sample indices and baseline for one beat-centered window.

    Attributes:
        r_idx: R peak sample index.
        p_peak: P-wave peak index, if detected.
        p_on: P-wave onset index, if detected.
        p_off: P-wave offset index, if detected.
        qrs_on: QRS onset sample index.
        qrs_off: QRS offset sample index.
        t_peak: T-wave peak index, if detected.
        t_on: T-wave onset index, if detected.
        t_off: T-wave offset index, if detected.
        retrograde_p: Retrograde P peak index, if detected.
        baseline: Isoelectric level in millivolts.
    """

    r_idx: int
    p_peak: int | None
    p_on: int | None
    p_off: int | None
    qrs_on: int
    qrs_off: int
    t_peak: int | None
    t_on: int | None
    t_off: int | None
    retrograde_p: int | None
    baseline: float


def _baseline(sig: np.ndarray, r_idx: int) -> float:
    """Estimate local baseline just before the R peak.

    Args:
        sig: Beat window samples.
        r_idx: R peak index inside ``sig``.

    Returns:
        Median baseline level in millivolts.
    """
    pre = sig[max(0, r_idx - 40) : max(0, r_idx - 10)]
    if len(pre) < 5:
        pre = sig[: max(1, r_idx)]
    return float(np.median(pre)) if len(pre) else float(np.median(sig))


def _qrs_bounds(x: np.ndarray, r_idx: int) -> tuple[int, int]:
    """Locate QRS onset and offset from the derivative envelope.

    Args:
        x: Beat window samples.
        r_idx: R peak index inside ``x``.

    Returns:
        Inclusive ``(qrs_on, qrs_off)`` sample indices.
    """
    n = len(x)
    win = min(45, r_idx, n - r_idx - 1)
    lo = max(0, r_idx - win)
    hi = min(n, r_idx + win + 1)
    derivative = np.abs(np.diff(x[lo:hi]))
    if len(derivative) < 3:
        return lo, hi - 1
    active = np.where(derivative >= np.percentile(derivative, 70))[0]
    if len(active) == 0:
        qrs_on, qrs_off = r_idx - 15, r_idx + 15
    else:
        qrs_on = lo + int(active[0])
        qrs_off = lo + int(active[-1]) + 1
    return max(0, min(qrs_on, r_idx)), min(n - 1, max(qrs_off, r_idx + 1))


def _signed_peak(segment: np.ndarray) -> tuple[int, float] | None:
    """Pick the largest positive or negative peak in a segment.

    Args:
        segment: One-dimensional samples.

    Returns:
        ``(index, amplitude)`` for the dominant peak, or ``None`` when empty.
    """
    if len(segment) < 5:
        return None
    candidates: list[tuple[int, float]] = []
    positive, _ = find_peaks(segment)
    negative, _ = find_peaks(-segment)
    if len(positive):
        index = int(positive[np.argmax(segment[positive])])
        candidates.append((index, float(segment[index])))
    if len(negative):
        index = int(negative[np.argmax(-segment[negative])])
        candidates.append((index, float(segment[index])))
    if not candidates:
        return None
    return max(candidates, key=lambda item: abs(item[1]))


def delineate_beat(sig: np.ndarray, *, fs: int = 360, r_idx: int | None = None) -> BeatFiducials:
    """Detect P, QRS, and T fiducials on one beat window.

    Args:
        sig: Beat window samples in millivolts.
        fs: Sample rate in hertz.
        r_idx: R peak index; defaults to the window center.

    Returns:
        A ``BeatFiducials`` structure for the window.
    """
    x = np.asarray(sig, dtype=np.float64).ravel()
    n = len(x)
    if r_idx is None:
        r_idx = n // 2
    r_idx = int(np.clip(r_idx, 0, n - 1))
    baseline = _baseline(x, r_idx)
    qrs_on, qrs_off = _qrs_bounds(x, r_idx)
    p_peak, p_on, p_off = _find_p(x, baseline, qrs_on, fs)
    t_peak, t_on, t_off = _find_t(x, baseline, qrs_off, fs)
    retrograde = _find_retrograde(x, baseline, qrs_off, t_peak, fs)
    return BeatFiducials(
        r_idx=r_idx,
        p_peak=p_peak,
        p_on=p_on,
        p_off=p_off,
        qrs_on=int(qrs_on),
        qrs_off=int(qrs_off),
        t_peak=t_peak,
        t_on=t_on,
        t_off=t_off,
        retrograde_p=retrograde,
        baseline=baseline,
    )


def _find_p(
    x: np.ndarray, baseline: float, qrs_on: int, fs: int
) -> tuple[int | None, int | None, int | None]:
    """Search for P wave before QRS onset.

    Args:
        x: Full beat window.
        baseline: Isoelectric level.
        qrs_on: QRS onset index.
        fs: Sample rate in hertz.

    Returns:
        ``(p_peak, p_on, p_off)`` indices, each possibly ``None``.
    """
    search_end = max(0, qrs_on - ms_to_samples(40, fs))
    search_start = max(0, qrs_on - ms_to_samples(280, fs))
    if search_end <= search_start + 5:
        return None, None, None
    found = _signed_peak(x[search_start:search_end] - baseline)
    if found is None or abs(found[1]) < 0.04:
        return None, None, None
    peak = search_start + found[0]
    onset = max(search_start, peak - ms_to_samples(40, fs))
    offset = min(qrs_on, peak + ms_to_samples(50, fs))
    return peak, onset, offset


def _find_t(
    x: np.ndarray, baseline: float, qrs_off: int, fs: int
) -> tuple[int | None, int | None, int | None]:
    """Search for T wave after QRS offset.

    Args:
        x: Full beat window.
        baseline: Isoelectric level.
        qrs_off: QRS offset index.
        fs: Sample rate in hertz.

    Returns:
        ``(t_peak, t_on, t_off)`` indices, each possibly ``None``.
    """
    search_start = min(len(x) - 1, qrs_off + ms_to_samples(80, fs))
    search_end = min(len(x), qrs_off + ms_to_samples(500, fs))
    if search_end <= search_start + 5:
        return None, None, None
    found = _signed_peak(x[search_start:search_end] - baseline)
    if found is None:
        return None, None, None
    peak = search_start + found[0]
    return peak, search_start, min(len(x) - 1, peak + ms_to_samples(120, fs))


def _find_retrograde(
    x: np.ndarray,
    baseline: float,
    qrs_off: int,
    t_peak: int | None,
    fs: int,
) -> int | None:
    """Search for a retrograde P between QRS end and T peak.

    Args:
        x: Full beat window.
        baseline: Isoelectric level.
        qrs_off: QRS offset index.
        t_peak: T peak index, used to cap the search window.
        fs: Sample rate in hertz.

    Returns:
        Retrograde P peak index, or ``None`` when not found.
    """
    search_start = qrs_off + ms_to_samples(40, fs)
    search_end = qrs_off + ms_to_samples(220, fs)
    if t_peak is not None and search_start < t_peak < search_end:
        search_end = t_peak
    search_start = int(np.clip(search_start, 0, len(x) - 1))
    search_end = int(np.clip(search_end, 0, len(x)))
    if search_end <= search_start + 5:
        return None
    segment = x[search_start:search_end] - baseline
    negative, _ = find_peaks(-segment)
    if len(negative) == 0:
        return None
    index = int(negative[np.argmax(-segment[negative])])
    if float(segment[index]) > -0.05:
        return None
    return search_start + index