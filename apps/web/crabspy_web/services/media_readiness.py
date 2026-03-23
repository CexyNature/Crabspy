"""Shared rules for when core study metadata satisfies ``ready_for_processing``."""

from __future__ import annotations

from datetime import datetime


def core_metadata_ready_for_processing(
    collected_at: datetime | None,
    sample_code: str | None,
    site_name: str | None,
    location_name: str | None,
) -> bool:
    """True when date collected, sample, site, and location name are all present (non-empty where applicable)."""
    return (
        collected_at is not None
        and bool(sample_code and sample_code.strip())
        and bool(site_name and site_name.strip())
        and bool(location_name and location_name.strip())
    )
