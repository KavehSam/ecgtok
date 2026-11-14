# Working in this repository

ecgtok is a citable comparison: named MIT-BIH beat tokens against opaque energy codes, scored by one Qwen2.5-1.5B LoRA setup. Keep the repository to that comparison.

## Layout

- `src/ecgtok/` is the library: split, windowing, delineation, named tokens, opaque codes, packing, and metrics.
- `scripts/` downloads MIT-BIH and Qwen2.5-1.5B, builds the beat table, and trains.
- `docs/taxonomy.md` is the token definition. The README reports both splits and points here.
- `results/mitbih_qwen.json` is the published metric table. It is not a checkpoint.
- Tests live in `tests/`. Use `uv run pytest` and `uv run ruff check src scripts tests`.

## Data and weights

`data/`, `models/`, `cache/`, `runs/`, parquet files, and checkpoints stay gitignored. Download them with the scripts. Do not commit them.

## Protocol

DS1 trains. The last 15% of each DS1 record selects the epoch and, for ventricular versus rest, the threshold. DS2 is scored once from that checkpoint. Report both splits. Named tokens have the higher DS2 AUC on all four tasks. On DS1 validation they win only ventricular versus rest.

Public text stays in US English. Cite a method only when this repository uses it. Commits are authored by KavehSam. Python functions carry type annotations.
