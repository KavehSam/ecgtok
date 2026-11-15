"""Rhythm labels stored in the MIT-BIH annotation aux field."""

from __future__ import annotations


def clean_aux(note: object) -> str:
    """Normalize one MIT-BIH aux note to plain ASCII text.

    Args:
        note: Raw aux note from ``wfdb``, often ``bytes``.

    Returns:
        Trimmed text with trailing nulls removed.
    """
    if note is None:
        return ""
    text = note.decode("ascii", errors="ignore") if isinstance(note, bytes) else str(note)
    return text.replace("\x00", "").strip()


def rhythm_changes(samples: list[int], aux_notes: list[object]) -> list[tuple[int, str]]:
    """Collect rhythm-change markers from parenthetical aux notes.

    Args:
        samples: Annotation sample indices.
        aux_notes: Parallel aux notes from the record annotation file.

    Returns:
        Sorted list of ``(sample, rhythm_code)`` for notes that start with ``(``.
    """
    changes: list[tuple[int, str]] = []
    for sample, note in zip(samples, aux_notes, strict=True):
        text = clean_aux(note)
        if text.startswith("(") and len(text) > 1:
            changes.append((int(sample), text[1:].upper()))
    changes.sort(key=lambda item: item[0])
    return changes


def rhythm_at(changes: list[tuple[int, str]], sample: int, default: str = "N") -> str:
    """Rhythm label in effect at a given sample.

    Args:
        changes: Output of ``rhythm_changes``, sorted by sample.
        sample: Beat sample index to query.
        default: Label used before the first rhythm change.

    Returns:
        The rhythm code active at ``sample``.
    """
    current = default
    for position, name in changes:
        if position <= sample:
            current = name
        else:
            break
    return current