import numpy as np
import torch

from ecgtok.metrics import class_weights, multiclass_scores
from ecgtok.overlay import overlay_special
from ecgtok.pack import pack_ids
from ecgtok.rhythms import rhythm_at, rhythm_changes


def test_pack_keeps_one_id_per_token() -> None:
    ids, mask = pack_ids([1, 2, 3], pad_id=0, max_length=5)
    assert ids == [1, 2, 3, 0, 0]
    assert mask == [1, 1, 1, 0, 0]


def test_overlay_uses_the_learned_row() -> None:
    base = torch.zeros(2, 1, dtype=torch.float16)
    extra = torch.ones(1, 1, dtype=torch.float32)
    mixed = overlay_special(base, extra, torch.tensor([1]), torch.tensor([-1, 0]))
    assert mixed.dtype == torch.float16
    assert mixed.tolist() == [[1.0]]


def test_rare_class_weight_is_capped() -> None:
    weights = class_weights(np.array([0] * 1000 + [1]), n_classes=2, max_weight=10.0)
    assert weights[1] == 10.0


def test_missing_class_is_left_out_of_macro_auc() -> None:
    truth = np.array([0, 1, 0, 1])
    probs = np.array([[0.9, 0.1, 0.0], [0.2, 0.8, 0.0], [0.7, 0.3, 0.0], [0.4, 0.6, 0.0]])
    scores = multiclass_scores(truth, probs, ["N", "V", "Q"])
    assert scores["classes_scored"] == ["N", "V"]
    assert scores["per_class_auc"]["Q"] is None


def test_rhythm_annotation_is_read_from_the_aux_note() -> None:
    changes = rhythm_changes([0, 100], ["(N", "(AFIB"])
    assert rhythm_at(changes, 50) == "N"
    assert rhythm_at(changes, 100) == "AFIB"
