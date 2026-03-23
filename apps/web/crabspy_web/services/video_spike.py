"""Validation helpers for video spike annotations."""

from __future__ import annotations

from crabspy_web.models.media import Media, MediaKind


def media_allows_video_spike(media: Media) -> bool:
    return media.media_kind in (MediaKind.video, MediaKind.unknown)
