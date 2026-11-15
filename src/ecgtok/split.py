"""de Chazal DS1/DS2 split. Paced recordings are dropped."""

from __future__ import annotations

import pandas as pd

PACED_RECORDS = frozenset({"102", "104", "107", "217"})
DS1_RECORDS = frozenset(
    {
        "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
        "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
        "223", "230",
    }
)
DS2_RECORDS = frozenset(
    {
        "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
        "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
        "233", "234",
    }
)


def record_key(value: object) -> str:
    """Normalize a record name for split lookup.

    Args:
        value: Record name from a dataframe column.

    Returns:
        The stripped string form of ``value``.
    """
    return str(value).strip()


def de_chazal_group(value: object) -> str | None:
    """Assign a record to DS1, DS2, or exclude paced/unknown records.

    Args:
        value: Record name to classify.

    Returns:
        ``"DS1"``, ``"DS2"``, or ``None`` when the record is paced or unknown.
    """
    key = record_key(value)
    if key in DS1_RECORDS:
        return "DS1"
    if key in DS2_RECORDS:
        return "DS2"
    return None


def split_ds1_ds2(
    frame: pd.DataFrame, val_fraction: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split beats into DS1 train, DS1 validation tail, and DS2 test.

    Args:
        frame: Beat table with ``record_name`` and ``timestamp_ms``.
        val_fraction: Fraction of each DS1 record held out at the end in time.

    Returns:
        ``(train, validation, test)`` dataframes.

    Raises:
        ValueError: When ``val_fraction`` is not in ``(0, 0.5)``.
    """
    if not 0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be between 0 and 0.5")
    grouped = frame.copy()
    grouped["record_key"] = grouped["record_name"].map(record_key)
    grouped["de_chazal"] = grouped["record_key"].map(de_chazal_group)
    usable = grouped.loc[grouped["de_chazal"].notna()].copy()
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    ds1 = usable.loc[usable["de_chazal"] == "DS1"]
    for _, record in ds1.groupby("record_key", sort=False):
        ordered = record.sort_values("timestamp_ms")
        n_val = int(round(len(ordered) * val_fraction))
        if len(ordered) > 1:
            n_val = min(max(n_val, 0), len(ordered) - 1)
        else:
            n_val = 0
        val_parts.append(ordered.iloc[-n_val:] if n_val else ordered.iloc[0:0])
        train_parts.append(ordered.iloc[:-n_val] if n_val else ordered)
    train = pd.concat(train_parts, axis=0).reset_index(drop=True)
    validation = pd.concat(val_parts, axis=0).reset_index(drop=True)
    test = (
        usable.loc[usable["de_chazal"] == "DS2"]
        .sort_values(["record_key", "timestamp_ms"])
        .reset_index(drop=True)
    )
    return train, validation, test