"""Class weights, binary operating points, and macro AUC."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def class_weights(labels: np.ndarray, n_classes: int, max_weight: float = 10.0) -> np.ndarray:
    """Compute inverse-frequency class weights for multiclass training.

    Args:
        labels: Integer class labels.
        n_classes: Number of classes, including empty bins.
        max_weight: Cap applied to every non-zero weight.

    Returns:
        A weight vector of length ``n_classes``; absent classes are zero.
    """
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=n_classes).astype(np.float64)
    present = counts > 0
    weights = np.zeros(n_classes, dtype=np.float64)
    if present.any():
        weights[present] = counts[present].sum() / (int(present.sum()) * counts[present])
    return np.minimum(weights, max_weight)


def binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Summarize a binary decision at one threshold.

    Args:
        y_true: Ground-truth labels in ``{0, 1}``.
        y_pred: Hard predictions in ``{0, 1}``.

    Returns:
        Counts, sensitivity, precision, and F1 in one dictionary.
    """
    truth = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((truth == 1) & (pred == 1)))
    tn = int(np.sum((truth == 0) & (pred == 0)))
    fp = int(np.sum((truth == 0) & (pred == 1)))
    fn = int(np.sum((truth == 1) & (pred == 0)))

    def rate(num: int, den: int) -> float:
        """Compute a rate with zero denominator guard.

        Args:
            num: Numerator count.
            den: Denominator count.

        Returns:
            ``num / den``, or ``0.0`` when ``den`` is zero.
        """
        return 0.0 if den == 0 else float(num / den)

    sensitivity = rate(tp, tp + fn)
    precision = rate(tp, tp + fp)
    f1 = 0.0 if sensitivity + precision == 0 else 2 * precision * sensitivity / (precision + sensitivity)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": sensitivity,
        "precision": precision,
        "f1": f1,
    }


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Pick the probability threshold that maximizes ventricular F1 on validation.

    Args:
        y_true: Ground-truth labels in ``{0, 1}``.
        scores: Predicted positive-class probabilities.

    Returns:
        Threshold between 0.05 and 0.95 in steps of 0.05.
    """
    best_t = 0.5
    best_f1 = -1.0
    for step in range(5, 100, 5):
        threshold = step / 100
        f1 = binary_counts(y_true, (scores >= threshold).astype(int))["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_t = threshold
    return best_t


def multiclass_scores(truth: np.ndarray, probs: np.ndarray, names: list[str]) -> dict[str, object]:
    """Report macro AUC and macro F1 over scorable classes.

    Args:
        truth: Integer class labels.
        probs: Class probability matrix of shape ``(n, len(names))``.
        names: Human-readable name per class column.

    Returns:
        Macro scores, per-class AUC/F1 maps, and the list of classes that were scored.
    """
    target = np.asarray(truth, dtype=int)
    predicted = np.argmax(probs, axis=1)
    per_auc: dict[str, float | None] = {}
    usable: list[int] = []
    for index, name in enumerate(names):
        positive = target == index
        if positive.any() and (~positive).any():
            per_auc[name] = float(roc_auc_score(positive.astype(int), probs[:, index]))
            usable.append(index)
        else:
            per_auc[name] = None
    per_f1 = {
        name: float(f1_score(target == index, predicted == index, zero_division=0))
        for index, name in enumerate(names)
    }
    macro_auc = float(np.mean([per_auc[names[index]] for index in usable])) if usable else 0.0
    macro_f1 = float(np.mean([per_f1[names[index]] for index in usable])) if usable else 0.0
    return {
        "macro_auc": macro_auc,
        "macro_f1": macro_f1,
        "classes_scored": [names[index] for index in usable],
        "per_class_auc": per_auc,
        "per_class_f1": per_f1,
    }