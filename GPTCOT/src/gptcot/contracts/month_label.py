"""Utilities for parsing and normalizing front month labels."""

from __future__ import annotations

from calendar import month_abbr
from datetime import datetime


def parse_month_label(label: str) -> str:
    """
    Convert a label like ``Feb-2015`` into ``YYYY-MM-01``.

    Raises:
        ValueError: if the label cannot be parsed.
    """

    if label is None:
        raise ValueError("FrontMonth_Label is missing.")
    normalized = label.strip()
    try:
        dt = datetime.strptime(normalized, "%b-%Y")
    except ValueError as exc:
        valid = ", ".join(abbr for abbr in month_abbr if abbr)
        raise ValueError(
            f"Unparseable FrontMonth_Label '{label}'. Expected format 'Mon-YYYY' "
            f"using English month abbreviations ({valid})."
        ) from exc
    return f"{dt.year:04d}-{dt.month:02d}-01"
