"""Qwen2.5-1.5B on named tokens and opaque codes.

DS1 trains. The last 15% of each DS1 record selects the epoch. DS2 is scored
from that checkpoint. A non-finite loss or gradient stops the run.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from ecgtok.metrics import binary_counts, choose_threshold, class_weights, multiclass_scores
from ecgtok.overlay import overlay_special
from ecgtok.pack import pack_ids, resolve_id
from ecgtok.split import split_ds1_ds2

UNK_TOKEN = "<ECG_UNK>"
MAX_LENGTH = 24
EPOCHS = 2
MODEL = "models/Qwen2.5-1.5B"
TASKS = ("ventricular", "aami5", "symbols", "rhythm")
ARMS = ("opaque", "named")
LABEL = {"ventricular": "ventricular", "aami5": "aami_name", "symbols": "symbol", "rhythm": "rhythm"}
TEXT = {"opaque": "text_opaque", "named": "text_named"}


def _seed(seed: int = 42) -> None:
    """Fix Python, NumPy, and CUDA random seeds.

    Args:
        seed: Integer seed shared across libraries.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BeatRows(Dataset):
    """Dataset of one beat string per row, packed for Qwen."""

    def __init__(self, frame: pd.DataFrame, tokenizer: AutoTokenizer, column: str, unk_id: int) -> None:
        """Store the split slice and tokenization settings.

        Args:
            frame: Beat rows with text and ``target`` columns.
            tokenizer: Tokenizer with ECG special tokens registered.
            column: Text column to tokenize, for example ``text_named``.
            unk_id: Token id substituted when a symbol is unknown.

        Returns:
            None.
        """
        self.rows = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.column = column
        self.unk_id = unk_id

    def __len__(self) -> int:
        """Return the number of beats in the split.

        Returns:
            Row count of the underlying frame.
        """
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Pack one beat for the training loop.

        Args:
            index: Row index in the split frame.

        Returns:
            Dict with ``input_ids``, ``attention_mask``, and ``labels`` tensors.
        """
        row = self.rows.iloc[index]
        token_ids = [
            resolve_id(self.tokenizer.convert_tokens_to_ids(token), self.unk_id)
            for token in str(row[self.column]).split()
        ]
        ids, mask = pack_ids(token_ids, int(self.tokenizer.pad_token_id), max_length=MAX_LENGTH)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(row["target"]),
        }


def _specials(frame: pd.DataFrame, column: str) -> list[str]:
    """Collect unique tokens appearing in one text column.

    Args:
        frame: Training split used to size the special vocabulary.
        column: Beat text column, for example ``text_opaque``.

    Returns:
        Sorted list of token strings.
    """
    found = {token for text in frame[column].astype(str) for token in text.split()}
    return sorted(found)


def _build(specials: list[str], n_out: int, device: str) -> tuple[object, ...]:
    """Load Qwen, register specials, and build the classification head.

    Args:
        specials: ECG token strings seen in training.
        n_out: Output dimension, 1 for binary or class count for multiclass.
        device: Torch device string, for example ``"cuda"``.

    Returns:
        ``(tokenizer, base, head, extra, row_of, unk_id)`` tuple.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    vocab = [*specials, UNK_TOKEN] if UNK_TOKEN not in specials else list(specials)
    tokenizer.add_special_tokens({"additional_special_tokens": vocab})
    unk_id = tokenizer.convert_tokens_to_ids(UNK_TOKEN)
    if not isinstance(unk_id, int):
        raise ValueError("unk token was not added")
    base = AutoModel.from_pretrained(MODEL, dtype=torch.float16)
    base.resize_token_embeddings(len(tokenizer))
    base = get_peft_model(
        base,
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"], bias="none"),
    )
    base.gradient_checkpointing_enable()
    base.config.use_cache = False
    base.get_input_embeddings().weight.requires_grad = False
    hidden = int(base.config.hidden_size)
    extra = nn.Parameter(torch.empty(len(vocab), hidden, device=device, dtype=torch.float32))
    nn.init.normal_(extra, mean=0.0, std=0.02)
    row_of = torch.full((len(tokenizer),), -1, dtype=torch.long, device=device)
    for row, token in enumerate(vocab):
        row_of[int(tokenizer.convert_tokens_to_ids(token))] = row
    head = nn.Linear(hidden, n_out).to(device)
    base.to(device)
    return tokenizer, base, head, extra, row_of, unk_id


def _logits(
    base: nn.Module,
    head: nn.Module,
    extra: nn.Parameter,
    row_of: torch.Tensor,
    batch: dict[str, torch.Tensor],
    device: str,
) -> torch.Tensor:
    """Pool the last real token and apply the linear head.

    Args:
        base: LoRA-wrapped Qwen encoder.
        head: Task-specific linear classifier.
        extra: Learned rows for ECG special tokens.
        row_of: Maps vocab ids to rows in ``extra``.
        batch: Batch dict with ``input_ids`` and ``attention_mask``.
        device: Torch device for the forward pass.

    Returns:
        Logits of shape ``(batch, n_out)``.
    """
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    hidden = base(
        inputs_embeds=overlay_special(base.get_input_embeddings().weight, extra, ids, row_of),
        attention_mask=mask,
    ).last_hidden_state
    index = mask.sum(dim=1) - 1
    pooled = hidden[torch.arange(ids.size(0), device=device), index]
    return head(pooled.float())


def _check_finite(value: torch.Tensor, where: str) -> None:
    """Abort training when a tensor contains non-finite values.

    Args:
        value: Loss or gradient norm to inspect.
        where: Message fragment naming the training step.

    Returns:
        None.

    Raises:
        RuntimeError: When any element is NaN or infinite.
    """
    if not torch.isfinite(value).all():
        raise RuntimeError(f"{where} is not finite")


def _prepare(
    frame: pd.DataFrame, label: str, binary: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Split the beat table and attach numeric targets.

    Args:
        frame: Full beat parquet with labels and text columns.
        label: Column used as the supervision target.
        binary: When ``True``, keep scalar ventricular targets.

    Returns:
        ``(train, validation, test, class_names)`` for the task.
    """
    train, validation, test = split_ds1_ds2(frame)
    if binary:
        for part in (train, validation, test):
            part["target"] = part[label].astype(float)
        return train, validation, test, ["rest", "ventricular"]
    names = sorted(train[label].astype(str).unique())
    lookup = {name: index for index, name in enumerate(names)}

    def indexed(part: pd.DataFrame) -> pd.DataFrame:
        """Map string labels to integer class ids.

        Args:
            part: One split slice of the beat frame.

        Returns:
            Rows with an integer ``target`` column.
        """
        kept = part.loc[part[label].isin(lookup)].copy()
        kept["target"] = kept[label].map(lookup).astype(int)
        return kept

    return indexed(train), indexed(validation), indexed(test), names


def _snapshot(base: nn.Module, head: nn.Module, extra: nn.Parameter) -> dict[str, object]:
    """Serialize trainable weights for the best validation epoch.

    Args:
        base: LoRA-wrapped Qwen encoder.
        head: Task linear head.
        extra: Learned special-token embedding rows.

    Returns:
        CPU tensors for ``extra``, ``head``, and LoRA weights.
    """
    return {
        "extra": extra.detach().cpu().clone(),
        "head": {key: value.detach().cpu().clone() for key, value in head.state_dict().items()},
        "lora": {key: value.detach().cpu().clone() for key, value in base.state_dict().items() if "lora_" in key},
    }


def _restore(base: nn.Module, head: nn.Module, extra: nn.Parameter, blob: dict[str, object], device: str) -> None:
    """Reload weights saved by ``_snapshot``.

    Args:
        base: LoRA-wrapped Qwen encoder to update.
        head: Task linear head to update.
        extra: Learned special-token rows to update.
        blob: Snapshot dict from ``_snapshot``.
        device: Torch device for restored tensors.

    Returns:
        None.

    Raises:
        TypeError: When required snapshot keys are missing.
    """
    extra_saved = blob["extra"]
    if not isinstance(extra_saved, torch.Tensor):
        raise TypeError("missing token snapshot")
    extra.data.copy_(extra_saved.to(device))
    head_saved = blob["head"]
    lora_saved = blob["lora"]
    if not isinstance(head_saved, dict) or not isinstance(lora_saved, dict):
        raise TypeError("missing head snapshot")
    head.load_state_dict({key: value.to(device) for key, value in head_saved.items()})
    base.load_state_dict(lora_saved, strict=False)


def _train_one(frame: pd.DataFrame, task: str, arm: str, device: str) -> dict[str, object]:
    """Train and evaluate one task arm on named or opaque strings.

    Args:
        frame: Full beat parquet.
        task: One of ``ventricular``, ``aami5``, ``symbols``, or ``rhythm``.
        arm: ``"named"`` or ``"opaque"`` text column.
        device: Torch device string.

    Returns:
        Metrics dict with ``ds1_val``, ``ds2``, and the chosen epoch.
    """
    _seed(42)
    binary = task == "ventricular"
    train, validation, test, names = _prepare(frame, LABEL[task], binary)
    n_out = 1 if binary else len(names)
    tokenizer, base, head, extra, row_of, unk_id = _build(_specials(train, TEXT[arm]), n_out, device)
    train_loader = DataLoader(BeatRows(train, tokenizer, TEXT[arm], unk_id), batch_size=64, shuffle=True)
    val_loader = DataLoader(BeatRows(validation, tokenizer, TEXT[arm], unk_id), batch_size=128)
    test_loader = DataLoader(BeatRows(test, tokenizer, TEXT[arm], unk_id), batch_size=128)
    if binary:
        positive = float(train["target"].sum())
        loss_fn: nn.Module = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([(len(train) - positive) / max(positive, 1.0)], device=device)
        )
    else:
        counts = class_weights(train["target"].to_numpy(), len(names))
        weights = torch.tensor(counts, dtype=torch.float32, device=device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
    trainable = [param for param in [*base.parameters(), extra, *head.parameters()] if param.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4)
    best_score = -1.0
    best: dict[str, object] = {"classes": names}
    best_blob: dict[str, object] = {}
    log = Path("runs/train.log")
    log.parent.mkdir(parents=True, exist_ok=True)

    def note(line: str) -> None:
        """Append one training log line to stdout and ``runs/train.log``.

        Args:
            line: Text to record.

        Returns:
            None.
        """
        print(line, flush=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    for epoch in range(1, EPOCHS + 1):
        base.train()
        head.train()
        for step, batch in enumerate(train_loader, start=1):
            logits = _logits(base, head, extra, row_of, batch, device)
            target = batch["labels"].to(device)
            loss = loss_fn(logits.squeeze(-1), target.float()) if binary else loss_fn(logits, target.long())
            _check_finite(loss, f"{task} {arm} epoch {epoch} step {step} loss")
            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            _check_finite(grad_norm, f"{task} {arm} epoch {epoch} step {step} gradient")
            opt.step()
            if step == 1 or step % 100 == 0 or step == len(train_loader):
                note(f"{task} {arm} epoch {epoch} step {step}/{len(train_loader)} loss {float(loss.detach()):.4f}")
        score, val_metrics = _evaluate(base, head, extra, row_of, val_loader, validation, names, binary, device, None)
        note(f"{task} {arm} epoch {epoch} ds1-val {score:.4f}")
        if score > best_score:
            best_score = score
            best = {"epoch": epoch, "classes": names, "ds1_val": val_metrics}
            best_blob = _snapshot(base, head, extra)
    _restore(base, head, extra, best_blob, device)
    _, test_metrics = _evaluate(
        base, head, extra, row_of, test_loader, test, names, binary, device, best["ds1_val"]
    )
    best["ds2"] = test_metrics
    note(f"{task} {arm} ds2 { _score_of(test_metrics, binary):.4f}")
    del base, head, extra, opt
    if device == "cuda":
        torch.cuda.empty_cache()
    return best


def _score_of(metrics: dict[str, object], binary: bool) -> float:
    """Read the primary validation metric for model selection.

    Args:
        metrics: Evaluation dict from ``_evaluate``.
        binary: When ``True``, use ``auc`` instead of ``macro_auc``.

    Returns:
        Scalar score used to pick the best epoch.
    """
    key = "auc" if binary else "macro_auc"
    return float(metrics[key])  # type: ignore[arg-type]


def _evaluate(
    base: nn.Module,
    head: nn.Module,
    extra: nn.Parameter,
    row_of: torch.Tensor,
    loader: DataLoader[dict[str, torch.Tensor]],
    frame: pd.DataFrame,
    names: list[str],
    binary: bool,
    device: str,
    selection: dict[str, object] | None,
) -> tuple[float, dict[str, object]]:
    """Score one split after the model is trained or restored.

    Args:
        base: LoRA-wrapped Qwen encoder.
        head: Task linear head.
        extra: Learned special-token embedding rows.
        row_of: Maps vocab ids to rows in ``extra``.
        loader: DataLoader for the split being scored.
        frame: Beat rows aligned with ``loader`` order.
        names: Class names for multiclass metrics.
        binary: When ``True``, compute binary AUC and threshold metrics.
        device: Torch device for inference.
        selection: Prior validation metrics for a fixed threshold, if any.

    Returns:
        ``(score, metrics_dict)`` for the split.
    """
    base.eval()
    head.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            logits = _logits(base, head, extra, row_of, batch, device)
            _check_finite(logits, "evaluation logits")
            if binary:
                rows.append(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())
            else:
                rows.append(torch.softmax(logits, dim=1).cpu().numpy())
    output = np.concatenate(rows, axis=0)
    truth = frame["target"].to_numpy()
    if binary:
        threshold = choose_threshold(truth, output) if selection is None else float(selection["threshold"])  # type: ignore[arg-type]
        operating = binary_counts(truth, (output >= threshold).astype(int))
        metrics: dict[str, object] = {"auc": float(roc_auc_score(truth, output)), "threshold": threshold, **operating}
        return float(metrics["auc"]), metrics  # type: ignore[arg-type]
    metrics = multiclass_scores(truth.astype(int), output, names)
    return float(metrics["macro_auc"]), metrics  # type: ignore[arg-type]


def main() -> None:
    """Run all task and arm combinations on the beat cache.

    Args:
        None. Uses ``sys.argv[1]`` as the parquet path, default ``cache/beats.parquet``.

    Returns:
        None. Writes ``runs/qwen_mitbih.json`` incrementally during training.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frame = pd.read_parquet(Path(sys.argv[1] if len(sys.argv) > 1 else "cache/beats.parquet"))
    dest = Path("runs/qwen_mitbih.json")
    payload: dict[str, object] = {"epochs": EPOCHS, "tasks": {}}
    tasks = payload["tasks"]
    if not isinstance(tasks, dict):
        raise TypeError("tasks")
    for task in TASKS:
        tasks[task] = {"arms": {}}
        for arm in ARMS:
            tasks[task]["arms"][arm] = _train_one(frame, task, arm, device)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        named = tasks[task]["arms"]["named"]["ds1_val"]
        opaque = tasks[task]["arms"]["opaque"]["ds1_val"]
        key = "auc" if task == "ventricular" else "macro_auc"
        tasks[task]["decision"] = (
            "named ds1-val score beat opaque codes"
            if float(named[key]) > float(opaque[key])
            else "named ds1-val score did not beat opaque codes"
        )
        dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"{task} {tasks[task]['decision']}", flush=True)


if __name__ == "__main__":
    main()
