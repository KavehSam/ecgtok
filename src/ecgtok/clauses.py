"""Bins a clinician would use for one beat."""

from __future__ import annotations


def qrs_duration_clause(qrs_ms: float) -> str:
    """Map QRS duration to a width clause.

    Args:
        qrs_ms: QRS duration in milliseconds.

    Returns:
        A ``qrs_*`` clause name for the duration bin.
    """
    if qrs_ms < 80:
        return "qrs_very_narrow"
    if qrs_ms < 100:
        return "qrs_narrow"
    if qrs_ms < 120:
        return "qrs_medium"
    if qrs_ms < 160:
        return "qrs_wide"
    return "qrs_very_wide"


def r_amplitude_clause(r_amp: float) -> str:
    """Map R-wave amplitude to a size clause.

    Args:
        r_amp: R peak amplitude in millivolts (sign preserved before taking magnitude).

    Returns:
        An ``r_*`` clause name for the amplitude bin.
    """
    magnitude = abs(r_amp)
    if magnitude < 0.15:
        return "r_very_low"
    if magnitude < 0.50:
        return "r_low"
    if magnitude < 0.80:
        return "r_medium"
    if magnitude < 1.50:
        return "r_high"
    return "r_very_high"


def pr_clause(pr_ms: float | None) -> str:
    """Map PR interval to a timing clause.

    Args:
        pr_ms: PR interval in milliseconds, or ``None`` when P onset is unknown.

    Returns:
        A ``pr_*`` clause name, or ``pr_unmeasurable`` when ``pr_ms`` is ``None``.
    """
    if pr_ms is None:
        return "pr_unmeasurable"
    if pr_ms < 100:
        return "pr_very_short"
    if pr_ms < 120:
        return "pr_short"
    if pr_ms <= 200:
        return "pr_normal"
    if pr_ms <= 240:
        return "pr_long"
    return "pr_very_long"


def timing_clause(rr_prev_ms: float | None, rr_baseline_ms: float | None) -> str:
    """Compare the previous R–R interval to the record baseline.

    Args:
        rr_prev_ms: Previous R–R interval in milliseconds.
        rr_baseline_ms: Median of earlier R–R intervals on the same record.

    Returns:
        A ``beat_*`` prematurity clause, or ``timing_uncertain`` when inputs are missing.
    """
    if rr_prev_ms is None or rr_baseline_ms is None or rr_baseline_ms <= 0:
        return "timing_uncertain"
    ratio = rr_prev_ms / rr_baseline_ms
    if ratio < 0.70:
        return "beat_very_early"
    if ratio < 0.85:
        return "beat_early"
    if ratio <= 1.15:
        return "beat_on_time"
    if ratio <= 1.40:
        return "beat_late"
    return "beat_very_late"


def pause_clause(
    rr_prev_ms: float | None,
    rr_next_ms: float | None,
    rr_baseline_ms: float | None,
) -> str:
    """Classify the pause pattern around the beat.

    Args:
        rr_prev_ms: Previous R–R interval in milliseconds.
        rr_next_ms: Next R–R interval in milliseconds.
        rr_baseline_ms: Median of earlier R–R intervals on the same record.

    Returns:
        A pause clause such as ``interpolated_beat`` or ``pause_uncertain``.
    """
    if rr_prev_ms is None or rr_next_ms is None or rr_baseline_ms is None or rr_baseline_ms <= 0:
        return "pause_uncertain"
    span = rr_prev_ms + rr_next_ms
    if span <= 1.25 * rr_baseline_ms:
        return "interpolated_beat"
    if 1.8 * rr_baseline_ms <= span <= 2.2 * rr_baseline_ms:
        return "full_compensatory_pause"
    return "incomplete_pause"