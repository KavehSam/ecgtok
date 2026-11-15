import numpy as np

from ecgtok.clauses import qrs_duration_clause
from ecgtok.tokens import named_beat_tokens, opaque_beat_tokens


def test_duration_bins() -> None:
    assert qrs_duration_clause(70) == "qrs_very_narrow"
    assert qrs_duration_clause(140) == "qrs_wide"
    assert qrs_duration_clause(180) == "qrs_very_wide"


def test_named_tokens_are_single_symbols() -> None:
    window = np.zeros(325)
    window[108] = 1.2
    tokens = named_beat_tokens(
        window, fs=360, r_idx=108, rr_prev_ms=800.0, rr_next_ms=800.0, rr_baseline_ms=800.0
    )
    assert tokens
    assert all(token.startswith("<") and token.endswith(">") for token in tokens)
    assert all(" " not in token for token in tokens)
    assert "<BEAT_ON_TIME>" in tokens


def test_opaque_codes_have_no_clinical_names() -> None:
    tokens = opaque_beat_tokens(np.ones(120))
    assert len(tokens) == 12
    assert tokens[0].startswith("<B00_")
    assert all("QRS" not in token and "ST_" not in token for token in tokens)
