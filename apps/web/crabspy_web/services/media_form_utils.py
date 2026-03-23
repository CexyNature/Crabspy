"""Parse optional values from HTML form strings."""

from __future__ import annotations


def empty_to_none(s: str | None) -> str | None:
    if s is None:
        return None
    t = s.strip()
    return t if t else None


def parse_optional_int(raw: str | None) -> int | None:
    raw = empty_to_none(raw)
    if raw is None:
        return None
    return int(raw, 10)


def parse_optional_float(raw: str | None) -> float | None:
    raw = empty_to_none(raw)
    if raw is None:
        return None
    return float(raw.replace(",", "."))
