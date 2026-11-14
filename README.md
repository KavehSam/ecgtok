# ecgtok

Named tokens for MIT-BIH beat classification. Recordings are downloaded from PhysioNet and are not stored in git.

```powershell
uv sync --extra dev
uv run python scripts/download_mitbih.py data/mitdb
```
