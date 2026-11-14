"""Turn fiducials into the named beat clauses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from ecgtok.clauses import (
    pause_clause,
    pr_clause,
    qrs_duration_clause,
    r_amplitude_clause,
    timing_clause,
)
from ecgtok.delineate import delineate_beat, ms_to_samples


@dataclass
class BeatMeasures:
    qrs_ms: float
    r_amp: float
    p_amp: float | None
    t_amp: float | None
    st_amp: float
    qrs_pos: float
    qrs_neg: float
    qrs_seg: np.ndarray
    pr_ms: float | None
    retrograde: bool


def beat_features(sig: np.ndarray, *, fs: int = 360, r_idx: int | None = None) -> BeatMeasures:
    x = np.asarray(sig, dtype=np.float64).ravel()
    fid = delineate_beat(x, fs=fs, r_idx=r_idx)
    qrs_seg = x[fid.qrs_on : fid.qrs_off + 1] - fid.baseline
    qrs_pos = float(np.max(qrs_seg)) if len(qrs_seg) else 0.0
    qrs_neg = float(-np.min(qrs_seg)) if len(qrs_seg) else 0.0
    p_amp = float(x[fid.p_peak] - fid.baseline) if fid.p_peak is not None else None
    t_amp = float(x[fid.t_peak] - fid.baseline) if fid.t_peak is not None else None
    st_idx = fid.qrs_off + ms_to_samples(80, fs)
    st_amp = float(x[st_idx] - fid.baseline) if 0 <= st_idx < len(x) else 0.0
    pr_ms = None if fid.p_on is None else (fid.qrs_on - fid.p_on) / fs * 1000.0
    return BeatMeasures(
        qrs_ms=(fid.qrs_off - fid.qrs_on) / fs * 1000.0,
        r_amp=float(x[fid.r_idx] - fid.baseline),
        p_amp=p_amp,
        t_amp=t_amp,
        st_amp=st_amp,
        qrs_pos=qrs_pos,
        qrs_neg=qrs_neg,
        qrs_seg=qrs_seg,
        pr_ms=pr_ms,
        retrograde=fid.retrograde_p is not None,
    )


def _polarity(qrs_pos: float, qrs_neg: float) -> str:
    if qrs_neg > qrs_pos * 1.2 and qrs_neg > 0.1:
        return "qrs_negative_dominant"
    if qrs_pos > qrs_neg * 1.2:
        return "qrs_positive_dominant"
    return "qrs_biphasic"


def _larger_polarity(qrs_pos: float, qrs_neg: float) -> int:
    if np.isclose(qrs_pos, qrs_neg):
        return 0
    return 1 if qrs_pos > qrs_neg else -1


def _side_notched(segment: np.ndarray, sign: int, fs: int) -> bool:
    wave = segment if sign > 0 else -segment
    peak = float(np.max(wave)) if len(wave) else 0.0
    if peak <= 0:
        return False
    peaks, _ = find_peaks(wave, height=0.25 * peak)
    for left in range(len(peaks)):
        for right in range(left + 1, len(peaks)):
            if (int(peaks[right]) - int(peaks[left])) / fs * 1000.0 >= 20.0:
                return True
    return False


def _notch_clause(segment: np.ndarray, qrs_pos: float, qrs_neg: float, fs: int) -> str:
    if np.isclose(qrs_pos, qrs_neg):
        notched = _side_notched(segment, 1, fs) or _side_notched(segment, -1, fs)
    else:
        notched = _side_notched(segment, _larger_polarity(qrs_pos, qrs_neg), fs)
    return "qrs_notched" if notched else "qrs_smooth"


def _agreement(level: float, qrs_pos: float, qrs_neg: float, uncertain: str) -> str:
    if abs(level) <= 0.05 or _larger_polarity(qrs_pos, qrs_neg) == 0:
        return uncertain
    same = (level > 0 and qrs_pos > qrs_neg) or (level < 0 and qrs_neg > qrs_pos)
    stem = uncertain.split("_")[0]
    return f"{stem}_qrs_concordant" if same else f"{stem}_qrs_discordant"


def named_clauses(
    feat: BeatMeasures,
    *,
    fs: int,
    rr_prev_ms: float | None,
    rr_next_ms: float | None,
    rr_baseline_ms: float | None,
) -> list[str]:
    if feat.p_amp is None:
        p_clause = "p_absent"
    elif feat.p_amp >= 0:
        p_clause = "p_upright"
    else:
        p_clause = "p_inverted"
    if feat.t_amp is None:
        t_clause, t_level, t_known = "t_uncertain", 0.0, False
    elif feat.t_amp > 0.10:
        t_clause, t_level, t_known = "t_positive", feat.t_amp, True
    elif feat.t_amp < -0.10:
        t_clause, t_level, t_known = "t_negative", feat.t_amp, True
    else:
        t_clause, t_level, t_known = "t_flat", feat.t_amp, False
    if feat.st_amp > 0.10:
        st_token = "st_elevated"
    elif feat.st_amp < -0.10:
        st_token = "st_depressed"
    else:
        st_token = "st_isoelectric"
    t_relation = (
        "t_qrs_uncertain"
        if not t_known
        else _agreement(t_level, feat.qrs_pos, feat.qrs_neg, "t_qrs_uncertain")
    )
    return [
        qrs_duration_clause(feat.qrs_ms),
        r_amplitude_clause(feat.r_amp),
        _polarity(feat.qrs_pos, feat.qrs_neg),
        _notch_clause(feat.qrs_seg, feat.qrs_pos, feat.qrs_neg, fs),
        p_clause,
        pr_clause(feat.pr_ms),
        "retrograde_p_present" if feat.retrograde else "retrograde_p_absent",
        st_token,
        _agreement(feat.st_amp, feat.qrs_pos, feat.qrs_neg, "st_qrs_neutral"),
        t_clause,
        t_relation,
        timing_clause(rr_prev_ms, rr_baseline_ms),
        pause_clause(rr_prev_ms, rr_next_ms, rr_baseline_ms),
    ]
