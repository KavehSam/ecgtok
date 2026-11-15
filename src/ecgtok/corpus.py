"""Load MIT-BIH MLII beats and the same window the named tokens use."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from ecgtok.rhythms import rhythm_at, rhythm_changes
from ecgtok.windows import (
    AAMI_NAMES,
    FILTER_DELAY,
    FS,
    apply_bandpass,
    design_bandpass,
    slice_window,
    symbol_to_aami,
)


def _lead_index(record: object, lead: str = "MLII") -> int | None:
    names = getattr(record, "sig_name", None) or []
    for index, name in enumerate(names):
        if str(name).strip().upper() == lead:
            return index
    return None


def load_beats(data_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the beat table and a float32 window for every row, in the same order."""
    highpass, lowpass = design_bandpass(FS)
    rows: list[dict[str, object]] = []
    windows: list[np.ndarray] = []
    for header in tqdm(sorted(data_dir.glob("*.hea")), desc="records"):
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
        for sample, symbol, aami in kept:
            window = slice_window(signal, sample + FILTER_DELAY)
            if window is None:
                continue
            rows.append(
                {
                    "record_name": record_name,
                    "timestamp_ms": int(sample * 1000 / FS),
                    "symbol": symbol,
                    "aami": aami,
                    "aami_name": AAMI_NAMES[aami],
                    "ventricular": int(aami == 2),
                    "rhythm": rhythm_at(changes, sample),
                }
            )
            windows.append(np.asarray(window, dtype=np.float32))
    if not rows:
        raise FileNotFoundError(f"no MLII beats under {data_dir}")
    return pd.DataFrame(rows), np.stack(windows, axis=0)
