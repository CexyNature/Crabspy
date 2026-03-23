"""I/O helpers (video, frames, …)."""

from __future__ import annotations

from crabspy.io.video import VideoMetadata, VideoOpenError, decode_fourcc, open_video

__all__ = [
    "VideoMetadata",
    "VideoOpenError",
    "decode_fourcc",
    "open_video",
]
