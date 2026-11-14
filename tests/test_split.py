import pandas as pd
import pytest

from ecgtok.split import de_chazal_group, split_ds1_ds2
from ecgtok.windows import symbol_to_aami


def test_paced_records_are_excluded() -> None:
    assert de_chazal_group("102") is None
    assert de_chazal_group("101") == "DS1"
    assert de_chazal_group("100") == "DS2"


def test_validation_is_the_tail_of_each_ds1_record() -> None:
    rows = []
    for record, count in (("101", 10), ("100", 4)):
        for beat in range(count):
            rows.append({"record_name": record, "timestamp_ms": beat * 1000})
    train, validation, test = split_ds1_ds2(pd.DataFrame(rows), val_fraction=0.2)
    assert list(validation["timestamp_ms"]) == [8000, 9000]
    assert train["timestamp_ms"].max() == 7000
    assert len(test) == 4


def test_bad_fraction_is_rejected() -> None:
    with pytest.raises(ValueError):
        split_ds1_ds2(pd.DataFrame({"record_name": ["101"], "timestamp_ms": [0]}), val_fraction=0.9)


def test_aami_symbol_map() -> None:
    assert symbol_to_aami("V") == 2
    assert symbol_to_aami("L") == 0
    assert symbol_to_aami("!") is None
