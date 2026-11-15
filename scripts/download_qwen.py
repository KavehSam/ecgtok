"""Download Qwen2.5-1.5B. The weights are not committed."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Download Qwen2.5-1.5B weights from Hugging Face.

    Args:
        None. Uses ``sys.argv[1]`` as the destination, default ``models/Qwen2.5-1.5B``.

    Returns:
        None.
    """
    dest = Path(sys.argv[1] if len(sys.argv) > 1 else "models/Qwen2.5-1.5B")
    dest.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    snapshot_download("Qwen/Qwen2.5-1.5B", local_dir=str(dest))
    print(f"Qwen2.5-1.5B is in {dest}")


if __name__ == "__main__":
    main()
