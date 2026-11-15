"""Download the MIT-BIH Arrhythmia Database from PhysioNet."""

from __future__ import annotations

import sys
from pathlib import Path

import wfdb


def download_mitdb(dest: Path) -> None:
    """Fetch mitdb into ``dest``. Files already present are left in place."""
    dest.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("mitdb", str(dest), overwrite=False)


def main() -> None:
    """Download into the directory named on the command line, or ``data/mitdb``."""
    dest = Path(sys.argv[1] if len(sys.argv) > 1 else "data/mitdb")
    download_mitdb(dest)
    print(f"MIT-BIH records are in {dest}")


if __name__ == "__main__":
    main()
