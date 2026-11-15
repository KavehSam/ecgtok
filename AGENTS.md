# Working in this repository

ecgtok compares named MIT-BIH beat tokens with opaque energy codes, fixed-scale codes, and input alignment. One Qwen2.5-1.5B LoRA setup scores all four. Keep new inputs on that same split and those same four tasks.

## Layout

- `src/ecgtok/` is the library: split, windowing, delineation, named tokens, opaque codes, packing, and metrics.
- `scripts/` downloads MIT-BIH and Qwen2.5-1.5B, builds the beat table, and trains.
- `docs/taxonomy.md` is the named-token definition. `docs/adapted.md` is the fixed-scale quantizer and the input alignment. The README reports both splits and points at both.
- `results/mitbih_qwen.json` holds named and opaque metrics. `results/adapted_qwen.json` holds fixed-scale and alignment metrics. Neither file is a checkpoint.
- Tests live in `tests/`. Use `uv run pytest` and `uv run ruff check src scripts tests`.

## Data and weights

`data/`, `models/`, `cache/`, `runs/`, parquet files, and checkpoints stay gitignored. Download them with the scripts. Do not commit them.

## Protocol

DS1 trains. The last 15% of each DS1 record selects the epoch and, for ventricular versus rest, the threshold. DS2 is scored once from that checkpoint. Report both splits, and report AUC and F1. On DS2, named tokens have the highest AUC and the highest F1 on all four tasks. On DS1 validation they lead ventricular versus rest. Input alignment leads the other three DS1-val AUCs. Bold in the README marks the best cell in a row, not a win on every split.

- Every Python function and class has a docstring and type annotations.
- Commits use the name and email already set as `user.name` and `user.email`. Do not change git config.
