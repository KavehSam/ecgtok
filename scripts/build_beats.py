"""Build named and opaque strings for every AAMI-mapped MIT-BIH beat."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from ecgtok.rhythms import rhythm_at, rhythm_changes
from ecgtok.tokens import named_beat_tokens, opaque_beat_tokens
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


def _lead_index(record: object, lead: str = "MLII") -> int | None:
    """Find the WFDB channel index for a lead name.

    Args:
        record: WFDB record object.
        lead: Lead name to match, case-insensitive.

    Returns:
        Channel index, or ``None`` when the lead is absent.
    """
    names = getattr(record, "sig_name", None) or []
    for index, name in enumerate(names):
        if str(name).strip().upper() == lead:
            return index
    return None


def build_table(data_dir: Path) -> pd.DataFrame:
    """Build the named and opaque token table for every MLII beat.

    Args:
        data_dir: MIT-BIH root directory.

    Returns:
        One row per beat with ``text_named`` and ``text_opaque`` columns.
    """
    highpass, lowpass = design_bandpass(FS)
    rows: list[dict[str, object]] = []
    headers = sorted(data_dir.glob("*.hea"))
    for header in tqdm(headers, desc="records"):
        record_name = header.stem
        if not header.with_suffix(".atr").exists():
            continue
        record = wfdb.rdrecord(str(data_dir / record_name))
        lead = _lead_index(record)
        if lead is None:
            continue
        signal = apply_bandpass(record.p_signal[:, lead], highpass, lowpass)
        annotation = wfdb.rdann(str(data_dir / record_name), "atr")
        changes = rhythm_changes(list(annotation.sample), list(annotation.aux_note))
        kept: list[tuple[int, str, int]] = []
        for sample, symbol in zip(annotation.sample, annotation.symbol, strict=True):
            aami = symbol_to_aami(symbol or "")
            if aami is None:
                continue
            kept.append((int(sample), str(symbol), aami))
        intervals = rhythm_intervals(
            [record_name] * len(kept),
            [sample for sample, _, _ in kept],
        )
        for pos, (sample, symbol, aami) in enumerate(kept):
            window = slice_window(signal, sample + FILTER_DELAY)
            if window is None:
                continue
            previous = intervals.at[pos, "rr_prev_ms"]
            nxt = intervals.at[pos, "rr_next_ms"]
            baseline = intervals.at[pos, "rr_baseline_ms"]
            rows.append(
                {
                    "record_name": record_name,
                    "timestamp_ms": int(sample * 1000 / FS),
                    "symbol": symbol,
                    "aami": aami,
                    "aami_name": AAMI_NAMES[aami],
                    "ventricular": int(aami == 2),
                    "rhythm": rhythm_at(changes, sample),
                    "text_named": " ".join(
                        named_beat_tokens(
                            window,
                            fs=FS,
                            r_idx=min(PRE_SAMPLES, len(window) - 1),
                            rr_prev_ms=None if np.isnan(previous) else float(previous),
                            rr_next_ms=None if np.isnan(nxt) else float(nxt),
                            rr_baseline_ms=None if np.isnan(baseline) else float(baseline),
                        )
                    ),
                    "text_opaque": " ".join(opaque_beat_tokens(window)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Write ``cache/beats.parquet`` from command-line paths.

    Args:
        None. Uses ``sys.argv[1]`` as the data directory and ``sys.argv[2]`` as the output path.

    Returns:
        None.
    """
    source = Path(sys.argv[1] if len(sys.argv) > 1 else "data/mitdb")
    dest = Path(sys.argv[2] if len(sys.argv) > 2 else "cache/beats.parquet")
    table = build_table(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(dest, index=False)
    print(f"wrote {len(table)} beats -> {dest}")


if __name__ == "__main__":
    main()
