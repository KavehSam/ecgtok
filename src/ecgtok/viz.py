"""Load one MIT-BIH beat and render the named token string."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb

from ecgtok.delineate import delineate_beat
from ecgtok.tokens import named_beat_tokens
from ecgtok.windows import (
    AAMI_NAMES,
    FILTER_DELAY,
    FS,
    PRE_SAMPLES,
    apply_bandpass,
    design_bandpass,
    rhythm_intervals,
    slice_window,
    symbol_to_aami,
)


@dataclass(frozen=True)
class BeatVizPayload:
    """Data passed from loading to figure rendering.

    Attributes:
        window: Filtered beat samples centered on R.
        tokens: Named token tuple for the beat.
        meta: Human-readable caption for plots.
        lead_name: WFDB lead name used for the waveform.
    """

    window: np.ndarray
    tokens: tuple[str, ...]
    meta: str
    lead_name: str


def lead_names(record: object) -> list[str]:
    """List WFDB signal names on a record.

    Args:
        record: WFDB record object with ``sig_name``.

    Returns:
        Lead names in channel order.
    """
    return [str(name).strip() for name in (getattr(record, "sig_name", None) or [])]


def resolve_lead_index(record: object, *, lead: str | None, lead_index: int | None) -> tuple[int, str]:
    """Resolve a lead by name or channel index.

    Args:
        record: WFDB record object.
        lead: Lead name when ``lead_index`` is ``None``.
        lead_index: WFDB channel index; overrides ``lead`` when set.

    Returns:
        ``(channel_index, lead_name)``.

    Raises:
        ValueError: When the record has no leads or the lead is unknown.
    """
    names = lead_names(record)
    if not names:
        raise ValueError("Record has no leads")
    if lead_index is not None:
        if lead_index < 0 or lead_index >= len(names):
            raise ValueError(f"lead_index {lead_index} is out of range for {len(names)} leads")
        return lead_index, names[lead_index]
    target = (lead or "MLII").strip().upper()
    for index, name in enumerate(names):
        if name.upper() == target:
            return index, name
    options = ", ".join(f"{index}={name}" for index, name in enumerate(names))
    raise ValueError(f"Lead {lead!r} not found. Available: {options}")


def _aami_beats(annotation: object) -> list[tuple[int, str, int]]:
    """List AAMI-mapped beats from one annotation file.

    Args:
        annotation: WFDB ``.atr`` annotation object.

    Returns:
        ``(sample, symbol, aami)`` for every mappable QRS beat.
    """
    kept: list[tuple[int, str, int]] = []
    for sample, symbol in zip(annotation.sample, annotation.symbol, strict=True):
        aami = symbol_to_aami(symbol or "")
        if aami is None:
            continue
        kept.append((int(sample), str(symbol), aami))
    return kept


def resolve_beat_sample(
    annotation: object,
    timestamp_ms: int,
    *,
    tolerance_samples: int = 2,
) -> tuple[int, str, int]:
    """Match a beat annotation to a wall-clock timestamp.

    Args:
        annotation: WFDB ``.atr`` annotation object.
        timestamp_ms: Beat time in milliseconds.
        tolerance_samples: Maximum sample offset allowed.

    Returns:
        ``(sample, symbol, aami)`` for the nearest beat.

    Raises:
        ValueError: When no suitable beat exists within tolerance.
    """
    target = int(round(timestamp_ms * FS / 1000.0))
    best: tuple[int, str, int] | None = None
    best_distance = tolerance_samples + 1
    for sample, symbol, aami in _aami_beats(annotation):
        distance = abs(sample - target)
        if distance < best_distance:
            best_distance = distance
            best = (sample, symbol, aami)
    if best is None:
        raise ValueError("No AAMI-mapped beats in annotation")
    if best_distance > tolerance_samples:
        raise ValueError(
            f"No beat within {tolerance_samples} samples of timestamp_ms={timestamp_ms} "
            f"(target sample {target})"
        )
    return best


def load_beat_for_viz(
    data_dir: Path,
    *,
    record_name: str,
    timestamp_ms: int,
    lead: str | None = "MLII",
    lead_index: int | None = None,
) -> BeatVizPayload:
    """Load waveform and named tokens for one annotated beat.

    Args:
        data_dir: MIT-BIH root directory.
        record_name: Record id such as ``"208"``.
        timestamp_ms: Beat time in milliseconds.
        lead: Lead name when ``lead_index`` is ``None``.
        lead_index: WFDB channel index; overrides ``lead`` when set.

    Returns:
        A ``BeatVizPayload`` ready for plotting.

    Raises:
        ValueError: When the beat or window cannot be resolved.
    """
    record_path = data_dir / record_name
    record = wfdb.rdrecord(str(record_path))
    channel, lead_name = resolve_lead_index(record, lead=lead, lead_index=lead_index)
    annotation = wfdb.rdann(str(record_path), "atr")
    sample, symbol, aami = resolve_beat_sample(annotation, timestamp_ms)
    kept = _aami_beats(annotation)
    samples = [item[0] for item in kept]
    try:
        beat_pos = samples.index(sample)
    except ValueError as exc:
        raise ValueError(f"Resolved sample {sample} missing from AAMI beat list") from exc
    intervals = rhythm_intervals([record_name] * len(kept), samples)
    highpass, lowpass = design_bandpass(FS)
    filtered = apply_bandpass(record.p_signal[:, channel], highpass, lowpass)
    window = slice_window(filtered, sample + FILTER_DELAY)
    if window is None:
        raise ValueError("Beat window is out of range after filtering")
    r_idx = min(PRE_SAMPLES, len(window) - 1)
    previous = intervals.at[beat_pos, "rr_prev_ms"]
    nxt = intervals.at[beat_pos, "rr_next_ms"]
    baseline = intervals.at[beat_pos, "rr_baseline_ms"]
    tokens = named_beat_tokens(
        window,
        fs=FS,
        r_idx=r_idx,
        rr_prev_ms=None if np.isnan(previous) else float(previous),
        rr_next_ms=None if np.isnan(nxt) else float(nxt),
        rr_baseline_ms=None if np.isnan(baseline) else float(baseline),
    )
    meta = (
        f"MIT-BIH record {record_name} · beat {symbol} (AAMI {AAMI_NAMES[aami]}) · "
        f"lead {lead_name} (index {channel}) · timestamp_ms {int(sample * 1000 / FS)}"
    )
    return BeatVizPayload(window=window, tokens=tokens, meta=meta, lead_name=lead_name)


def token_group(token: str) -> str:
    """Map one token to a plot legend group.

    Args:
        token: Named token such as ``<QRS_WIDE>``.

    Returns:
        ``"morphology"`` or ``"timing"``.
    """
    inner = token.strip("<>").lower()
    if inner.startswith(("beat_", "timing_", "full_", "incomplete_", "interpolated_", "pause_")):
        return "timing"
    if inner.startswith(("qrs_", "r_", "p_", "pr_", "retrograde_", "st_", "t_")):
        return "morphology"
    return "morphology"


# Legend colors for morphology versus timing token groups.
GROUP_COLORS = {
    "morphology": "#55A868",
    "timing": "#C44E52",
}


def render_beat_figure(
    payload: BeatVizPayload,
    out_path: Path,
    *,
    show: bool = False,
) -> None:
    """Save a waveform and token strip figure.

    Args:
        payload: Beat data from ``load_beat_for_viz``.
        out_path: Destination image path.
        show: When ``True``, open an interactive matplotlib window.

    Returns:
        None.
    """
    import matplotlib.pyplot as plt

    window = payload.window
    tokens = list(payload.tokens)
    t_ms = (np.arange(len(window)) - PRE_SAMPLES) / FS * 1000.0
    fid = delineate_beat(window, fs=FS, r_idx=min(PRE_SAMPLES, len(window) - 1))

    fig = plt.figure(figsize=(11, 5.8), dpi=150)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.35, 0.95], hspace=0.32)
    ax_ecg = fig.add_subplot(grid[0, 0])
    ax_tok = fig.add_subplot(grid[1, 0])
    ax_tok.set_axis_off()

    ax_ecg.plot(t_ms, window, color="#1a1a1a", linewidth=1.2, label=f"Filtered {payload.lead_name}")
    ax_ecg.axvline(0.0, color="#C44E52", linestyle="--", linewidth=1.0, alpha=0.85, label="Annotated R")
    qrs_start = (fid.qrs_on - PRE_SAMPLES) / FS * 1000.0
    qrs_end = (fid.qrs_off - PRE_SAMPLES) / FS * 1000.0
    ax_ecg.axvspan(qrs_start, qrs_end, color="#55A868", alpha=0.12, label="Detected QRS")
    for name, idx, color in (
        ("P", fid.p_peak, "#4C72B0"),
        ("R", fid.r_idx, "#C44E52"),
        ("T", fid.t_peak, "#DD8452"),
    ):
        if idx is None:
            continue
        x = (idx - PRE_SAMPLES) / FS * 1000.0
        ax_ecg.axvline(x, color=color, linestyle=":", linewidth=0.9, alpha=0.7)
        y = float(window[idx])
        ax_ecg.scatter([x], [y], s=28, color=color, zorder=5)
        ax_ecg.text(x, y + 0.08, name, ha="center", va="bottom", fontsize=9, color=color)

    ax_ecg.set_xlim(t_ms[0], t_ms[-1])
    ax_ecg.set_xlabel("Time relative to R peak (ms)")
    ax_ecg.set_ylabel("Amplitude (mV)")
    ax_ecg.set_title("Beat waveform → named ECGTok tokens")
    ax_ecg.text(0.01, 0.98, payload.meta, transform=ax_ecg.transAxes, va="top", ha="left", fontsize=9, color="#333333")
    ax_ecg.legend(loc="upper right", fontsize=8, framealpha=0.9)

    token_line = " ".join(tokens)
    ax_tok.text(
        0.0,
        0.92,
        "Model input (space-separated tokens):",
        transform=ax_tok.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    ax_tok.text(
        0.0,
        0.72,
        token_line,
        transform=ax_tok.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        family="monospace",
        wrap=True,
    )

    per_row = 7
    gap = 0.01
    usable = 0.92
    box_w = (usable - gap * (per_row - 1)) / per_row
    box_h = 0.12
    x0 = 0.04
    y_rows = (0.38, 0.22)
    for index, token in enumerate(tokens):
        group = token_group(token)
        color = GROUP_COLORS[group]
        row = index // per_row
        col = index % per_row
        if row >= len(y_rows):
            break
        x = x0 + col * (box_w + gap)
        y0 = y_rows[row]
        ax_tok.add_patch(
            plt.Rectangle(
                (x, y0),
                box_w,
                box_h,
                transform=ax_tok.transAxes,
                facecolor=color,
                alpha=0.18,
                edgecolor=color,
                linewidth=1.0,
            )
        )
        ax_tok.text(
            x + box_w / 2,
            y0 + box_h / 2,
            str(index + 1),
            transform=ax_tok.transAxes,
            ha="center",
            va="center",
            fontsize=7,
            color=color,
            fontweight="bold",
        )

    legend_y = 0.02
    for offset, (label, color) in enumerate(
        (
            ("Morphology (11)", GROUP_COLORS["morphology"]),
            ("Timing / pause (2)", GROUP_COLORS["timing"]),
        )
    ):
        ax_tok.add_patch(
            plt.Rectangle(
                (0.04 + offset * 0.45, legend_y),
                0.025,
                0.05,
                transform=ax_tok.transAxes,
                facecolor=color,
                alpha=0.35,
                edgecolor=color,
            )
        )
        ax_tok.text(0.07 + offset * 0.45, legend_y + 0.025, label, transform=ax_tok.transAxes, va="center", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)