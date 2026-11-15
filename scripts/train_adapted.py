"""Fixed-scale codes and input alignment on the four MIT-BIH tasks.

DS1 trains. The last 15% of each DS1 record selects the epoch and, for
ventricular versus rest, the threshold. DS2 is scored once from that
checkpoint. Named-token and opaque-code numbers stay the published run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

import train_qwen
from ecgtok.align import PROTOTYPES, PatchAlign
from ecgtok.corpus import load_beats
from ecgtok.fsq import FRAMES, BeatQuantizer, codes_to_text
from ecgtok.metrics import binary_counts, choose_threshold, class_weights, multiclass_scores
from ecgtok.split import split_ds1_ds2

RECON_EPOCHS = 5
TASKS = train_qwen.TASKS


class WindowRows(Dataset):
    """Dataset of raw beat windows linked by ``beat_index``."""

    def __init__(self, frame: pd.DataFrame, windows: np.ndarray) -> None:
        """Attach split rows to the shared window tensor.

        Args:
            frame: Split slice with ``beat_index`` and ``target`` columns.
            windows: All beat windows stacked as ``float32``.

        Returns:
            None.
        """
        self.frame = frame.reset_index(drop=True)
        self.windows = windows

    def __len__(self) -> int:
        """Return the number of beats in the split.

        Returns:
            Row count of the split frame.
        """
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Fetch one window and label.

        Args:
            index: Row index in the split frame.

        Returns:
            Dict with ``window`` and ``labels`` tensors.
        """
        row = self.frame.iloc[index]
        beat = int(row["beat_index"])
        return {
            "window": torch.tensor(self.windows[beat], dtype=torch.float32),
            "labels": torch.tensor(row["target"]),
        }


def _note(log: Path, line: str) -> None:
    """Append one line to stdout and the adapted training log.

    Args:
        log: Log file path, for example ``runs/adapted.log``.
        line: Text to append.

    Returns:
        None.
    """
    print(line, flush=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _train_quantizer(windows: np.ndarray, device: str, log: Path) -> BeatQuantizer:
    """Fit the FSQ autoencoder on DS1 windows.

    Args:
        windows: Training windows as ``float32`` array.
        device: Torch device string.
        log: Path for reconstruction MSE logs.

    Returns:
        Trained ``BeatQuantizer`` in eval mode.
    """
    model = BeatQuantizer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    data = torch.tensor(windows, dtype=torch.float32)
    for epoch in range(1, RECON_EPOCHS + 1):
        model.train()
        order = torch.randperm(data.shape[0])
        total = 0.0
        seen = 0
        for start in range(0, data.shape[0], 256):
            batch = data[order[start : start + 256]].to(device)
            recon, _ids = model(batch)
            loss = loss_fn(recon, batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"quantizer epoch {epoch} loss is not finite")
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * batch.shape[0]
            seen += batch.shape[0]
        _note(log, f"quantizer epoch {epoch} mse {total / max(seen, 1):.6f}")
    model.eval()
    return model


def _encode(model: BeatQuantizer, windows: np.ndarray, device: str) -> list[str]:
    """Quantize every beat window to FSQ token strings.

    Args:
        model: Trained quantizer.
        windows: All beat windows in database order.
        device: Torch device string.

    Returns:
        One ``text_fsq`` string per window.
    """
    texts: list[str] = []
    data = torch.tensor(windows, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, data.shape[0], 256):
            _recon, ids = model(data[start : start + 256].to(device))
            texts.extend(codes_to_text(ids))
    return texts


def _prototypes(base: nn.Module, count: int, device: str) -> torch.Tensor:
    """Sample frozen Qwen embedding rows for cross-attention keys.

    Args:
        base: Qwen model with an input embedding table.
        count: Number of prototype rows to keep.
        device: Torch device for the returned tensor.

    Returns:
        Tensor of shape ``(count, hidden)``.

    Raises:
        RuntimeError: When fewer than ``count`` rows can be sampled.
    """
    weight = base.get_input_embeddings().weight.detach()
    step = max(int(weight.shape[0]) // count, 1)
    chosen = weight[::step][:count].to(device=device, dtype=torch.float32)
    if chosen.shape[0] != count:
        raise RuntimeError("could not sample enough prototype rows")
    return chosen


def _align_logits(
    base: nn.Module,
    head: nn.Module,
    align: PatchAlign,
    windows: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Forward aligned patches and pool the last frame.

    Args:
        base: LoRA-wrapped Qwen encoder.
        head: Task linear head.
        align: ``PatchAlign`` module.
        windows: Batch of raw beat windows.
        device: Torch device string.

    Returns:
        Logits of shape ``(batch, n_out)``.
    """
    embeds = align(windows.to(device))
    dtype = base.get_input_embeddings().weight.dtype
    mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=device)
    hidden = base(inputs_embeds=embeds.to(dtype=dtype), attention_mask=mask).last_hidden_state
    return head(hidden[:, -1, :].float())


def _score_align(
    base: nn.Module,
    head: nn.Module,
    align: PatchAlign,
    loader: DataLoader[dict[str, torch.Tensor]],
    frame: pd.DataFrame,
    names: list[str],
    binary: bool,
    device: str,
    selection: dict[str, object] | None,
) -> tuple[float, dict[str, object]]:
    """Score one split for the alignment arm.

    Args:
        base: LoRA-wrapped Qwen encoder.
        head: Task linear head.
        align: ``PatchAlign`` module.
        loader: DataLoader for the split.
        frame: Beat frame aligned with ``loader``.
        names: Multiclass label names.
        binary: When ``True``, compute binary metrics.
        device: Torch device string.
        selection: Fixed validation metrics for thresholding, if any.

    Returns:
        ``(score, metrics_dict)`` for the split.
    """
    base.eval()
    head.eval()
    align.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            logits = _align_logits(base, head, align, batch["window"], device)
            if not torch.isfinite(logits).all():
                raise RuntimeError("align evaluation logits are not finite")
            if binary:
                rows.append(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())
            else:
                rows.append(torch.softmax(logits, dim=1).cpu().numpy())
    output = np.concatenate(rows, axis=0)
    truth = frame["target"].to_numpy()
    if binary:
        held = None if selection is None else selection["threshold"]
        threshold = choose_threshold(truth, output) if held is None else float(held)  # type: ignore[arg-type]
        operating = binary_counts(truth, (output >= threshold).astype(int))
        metrics: dict[str, object] = {
            "auc": float(roc_auc_score(truth, output)),
            "threshold": threshold,
            **operating,
        }
        return float(metrics["auc"]), metrics  # type: ignore[arg-type]
    metrics = multiclass_scores(truth.astype(int), output, names)
    return float(metrics["macro_auc"]), metrics  # type: ignore[arg-type]


def _train_align(
    frame: pd.DataFrame,
    windows: np.ndarray,
    task: str,
    device: str,
    log: Path,
) -> dict[str, object]:
    """Train patch alignment and score DS1 validation and DS2 test.

    Args:
        frame: Full beat table with ``beat_index``.
        windows: All windows aligned with ``frame`` rows.
        task: Task name shared with ``train_qwen``.
        device: Torch device string.
        log: Path for training logs.

    Returns:
        Best-epoch metrics for the alignment arm.
    """
    train_qwen._seed(42)
    binary = task == "ventricular"
    train, validation, test, names = train_qwen._prepare(frame, train_qwen.LABEL[task], binary)
    n_out = 1 if binary else len(names)
    tokenizer, base, head, _extra, _row_of, _unk = train_qwen._build([], n_out, device)
    del tokenizer
    hidden = int(base.config.hidden_size)
    align = PatchAlign(hidden).to(device)
    with torch.no_grad():
        align.prototypes.copy_(_prototypes(base, PROTOTYPES, device))
    train_loader = DataLoader(WindowRows(train, windows), batch_size=64, shuffle=True)
    val_loader = DataLoader(WindowRows(validation, windows), batch_size=128)
    test_loader = DataLoader(WindowRows(test, windows), batch_size=128)
    if binary:
        positive = float(train["target"].sum())
        loss_fn: nn.Module = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([(len(train) - positive) / max(positive, 1.0)], device=device)
        )
    else:
        counts = class_weights(train["target"].to_numpy(), len(names))
        weights = torch.tensor(counts, dtype=torch.float32, device=device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
    pieces = [*base.parameters(), *align.parameters(), *head.parameters()]
    trainable = [param for param in pieces if param.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4)
    best_score = -1.0
    best: dict[str, object] = {"classes": names}
    best_blob: dict[str, object] = {}
    for epoch in range(1, train_qwen.EPOCHS + 1):
        base.train()
        head.train()
        align.train()
        for step, batch in enumerate(train_loader, start=1):
            logits = _align_logits(base, head, align, batch["window"], device)
            target = batch["labels"].to(device)
            loss = loss_fn(logits.squeeze(-1), target.float()) if binary else loss_fn(logits, target.long())
            if not torch.isfinite(loss):
                raise RuntimeError(f"{task} align epoch {epoch} step {step} loss is not finite")
            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"{task} align epoch {epoch} step {step} gradient is not finite")
            opt.step()
            if step == 1 or step % 100 == 0 or step == len(train_loader):
                loss_text = f"{float(loss.detach()):.4f}"
                _note(log, f"{task} align epoch {epoch} step {step}/{len(train_loader)} loss {loss_text}")
        score, val_metrics = _score_align(
            base, head, align, val_loader, validation, names, binary, device, None
        )
        _note(log, f"{task} align epoch {epoch} ds1-val {score:.4f}")
        if score > best_score:
            best_score = score
            best = {"epoch": epoch, "classes": names, "ds1_val": val_metrics}
            best_blob = {
                "align": {key: value.detach().cpu().clone() for key, value in align.state_dict().items()},
                "head": {key: value.detach().cpu().clone() for key, value in head.state_dict().items()},
                "lora": {
                    key: value.detach().cpu().clone()
                    for key, value in base.state_dict().items()
                    if "lora_" in key
                },
            }
    saved_align = best_blob["align"]
    saved_head = best_blob["head"]
    saved_lora = best_blob["lora"]
    if not isinstance(saved_align, dict) or not isinstance(saved_head, dict) or not isinstance(saved_lora, dict):
        raise TypeError("missing align snapshot")
    align.load_state_dict({key: value.to(device) for key, value in saved_align.items()})
    head.load_state_dict({key: value.to(device) for key, value in saved_head.items()})
    base.load_state_dict(saved_lora, strict=False)
    selection = best["ds1_val"]
    if not isinstance(selection, dict):
        raise TypeError("missing validation snapshot")
    _, test_metrics = _score_align(
        base, head, align, test_loader, test, names, binary, device, selection
    )
    best["ds2"] = test_metrics
    key = "auc" if binary else "macro_auc"
    _note(log, f"{task} align ds2 {float(test_metrics[key]):.4f}")  # type: ignore[arg-type]
    del base, head, align, opt
    if device == "cuda":
        torch.cuda.empty_cache()
    return best


def _dump(dest: Path, payload: dict[str, object]) -> None:
    """Write JSON results atomically to disk.

    Args:
        dest: Output JSON path.
        payload: Serializable results dict.

    Returns:
        None.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    """Train FSQ and alignment arms on MIT-BIH beats.

    Args:
        None. Uses ``sys.argv[1]`` for the data directory and ``sys.argv[2]`` for the model path.

    Returns:
        None. Writes ``results/adapted_qwen.json``.
    """
    data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/mitdb")
    model_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "models/Qwen2.5-1.5B")
    train_qwen.MODEL = str(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log = Path("runs/adapted.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    frame, windows = load_beats(data_dir)
    frame["beat_index"] = np.arange(len(frame))
    train_frame, _validation, _test = split_ds1_ds2(frame)
    quantizer = _train_quantizer(windows[train_frame["beat_index"].to_numpy()], device, log)
    frame["text_fsq"] = _encode(quantizer, windows, device)
    del quantizer
    if device == "cuda":
        torch.cuda.empty_cache()
    dest = Path("results/adapted_qwen.json")
    payload: dict[str, object] = {
        "epochs": train_qwen.EPOCHS,
        "quantizer_epochs": RECON_EPOCHS,
        "frames": FRAMES,
        "note": (
            "FSQ is a one-lead fixed-scale quantizer trained to reconstruct DS1 beats, "
            "then classified like the named tokens. Input alignment cross-attends 13 patches "
            "to 64 frozen Qwen embedding rows. Neither arm is a full reproduction of HeartLLM or TIME-LLM."
        ),
        "tasks": {},
    }
    tasks = payload["tasks"]
    if not isinstance(tasks, dict):
        raise TypeError("tasks")
    train_qwen.TEXT["fsq"] = "text_fsq"
    for task in TASKS:
        tasks[task] = {"arms": {}}
        _note(log, f"start {task} fsq")
        tasks[task]["arms"]["fsq"] = train_qwen._train_one(frame, task, "fsq", device)
        _dump(dest, payload)
        _note(log, f"start {task} align")
        tasks[task]["arms"]["align"] = _train_align(frame, windows, task, device, log)
        _dump(dest, payload)
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
