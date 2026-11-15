#!/usr/bin/env python3
"""Render one MIT-BIH beat and the named ECGTok tokens for that window."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import wfdb

from ecgtok.viz import BeatVizPayload, lead_names, load_beat_for_viz, render_beat_figure

REPO = Path(__file__).resolve().parents[1]


def _list_leads(data_dir: Path, record_name: str) -> None:
    """Print WFDB lead indices for one record.

    Args:
        data_dir: MIT-BIH root directory.
        record_name: Record id such as ``"208"``.

    Returns:
        None. Writes ``index<TAB>name`` lines to stdout.
    """
    record = wfdb.rdrecord(str(data_dir / record_name))
    for index, name in enumerate(lead_names(record)):
        print(f"{index}\t{name}")


def _parse_args() -> argparse.Namespace:
    """Parse visualization CLI flags.

    Returns:
        Parsed ``argparse`` namespace.
    """
    parser = argparse.ArgumentParser(
        description="Visualize one MIT-BIH beat and the ECGTok named token string.",
    )
    parser.add_argument("--data-dir", type=Path, default=REPO / "data" / "mitdb", help="MIT-BIH record directory")
    parser.add_argument("--record", type=str, required=True, help="Record id, for example 208")
    parser.add_argument(
        "--timestamp-ms",
        type=int,
        default=None,
        help="Beat time in milliseconds (matches build_beats timestamp_ms)",
    )
    parser.add_argument("--lead", type=str, default="MLII", help="Lead name, for example MLII or V1")
    parser.add_argument(
        "--lead-index",
        type=int,
        default=None,
        help="WFDB channel index; overrides --lead when set",
    )
    parser.add_argument(
        "--list-leads",
        action="store_true",
        help="Print lead index and name for --record, then exit",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image path (default: docs/figures/<record>_<timestamp_ms>_<lead>.png)",
    )
    parser.add_argument("--show", action="store_true", help="Open an interactive matplotlib window")
    return parser.parse_args()


def _default_out_path(record: str, timestamp_ms: int, lead_name: str) -> Path:
    """Build the default figure path.

    Args:
        record: MIT-BIH record id.
        timestamp_ms: Beat time in milliseconds.
        lead_name: Resolved WFDB lead name.

    Returns:
        Path under ``docs/figures``.
    """
    safe_lead = lead_name.replace(" ", "_")
    return REPO / "docs" / "figures" / f"{record}_{timestamp_ms}_{safe_lead}.png"


def main() -> int:
    """Load one beat and write a figure.

    Returns:
        ``0`` on success, ``1`` on user or data errors.
    """
    args = _parse_args()
    data_dir = args.data_dir
    if not data_dir.is_dir():
        print(f"data-dir not found: {data_dir}", file=sys.stderr)
        print("Run: uv run python scripts/download_mitbih.py data/mitdb", file=sys.stderr)
        return 1
    if args.list_leads:
        _list_leads(data_dir, args.record)
        return 0
    if args.timestamp_ms is None:
        print("--timestamp-ms is required unless --list-leads is set", file=sys.stderr)
        return 1
    try:
        payload: BeatVizPayload = load_beat_for_viz(
            data_dir,
            record_name=args.record,
            timestamp_ms=args.timestamp_ms,
            lead=args.lead,
            lead_index=args.lead_index,
        )
    except (ValueError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1
    out = args.out or _default_out_path(args.record, args.timestamp_ms, payload.lead_name)
    render_beat_figure(payload, out, show=args.show)
    print(f"Wrote {out}")
    print(" ".join(payload.tokens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())