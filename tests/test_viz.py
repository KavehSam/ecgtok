"""Tests for beat visualization helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ecgtok.viz import resolve_beat_sample, resolve_lead_index, token_group


def test_token_group_timing() -> None:
    assert token_group("<BEAT_EARLY>") == "timing"
    assert token_group("<PAUSE_UNCERTAIN>") == "timing"


def test_token_group_morphology() -> None:
    assert token_group("<QRS_WIDE>") == "morphology"
    assert token_group("<T_NEGATIVE>") == "morphology"


def test_resolve_lead_index_by_name() -> None:
    record = SimpleNamespace(sig_name=["MLII", "V1"])
    index, name = resolve_lead_index(record, lead="V1", lead_index=None)
    assert index == 1
    assert name == "V1"


def test_resolve_lead_index_by_number() -> None:
    record = SimpleNamespace(sig_name=["MLII", "V1"])
    index, name = resolve_lead_index(record, lead=None, lead_index=0)
    assert index == 0
    assert name == "MLII"


def test_resolve_beat_sample_matches_timestamp() -> None:
    annotation = SimpleNamespace(
        sample=[1000, 2000, 3000],
        symbol=["N", "V", "N"],
    )
    sample, symbol, aami = resolve_beat_sample(annotation, timestamp_ms=int(2000 * 1000 / 360))
    assert sample == 2000
    assert symbol == "V"
    assert aami == 2


def test_resolve_lead_index_unknown() -> None:
    record = SimpleNamespace(sig_name=["MLII"])
    with pytest.raises(ValueError, match="not found"):
        resolve_lead_index(record, lead="XZ", lead_index=None)