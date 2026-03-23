"""Open video files and read stream metadata (OpenCV).

This module replaces ad-hoc ``methods.read_video`` usage for new code: no hard-coded
``video/`` prefix, no ``sys.exit``, optional paths via :class:`pathlib.Path`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2

PathLike = Union[str, Path]


class VideoOpenError(RuntimeError):
    """Failed to open the file or read required video properties."""


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Properties from :class:`cv2.VideoCapture` (best effort; codecs vary by backend)."""

    frame_count: int
    fps: float
    width: int
    height: int
    duration_seconds: float
    fourcc: str


def decode_fourcc(value: float) -> str:
    """Decode OpenCV FOURCC float/int to a four-character string."""
    code = int(round(float(value)))
    return "".join(chr((code >> 8 * i) & 0xFF) for i in range(4))


def _metadata_from_capture(cap: cv2.VideoCapture) -> VideoMetadata:
    fc = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_raw = cap.get(cv2.CAP_PROP_FOURCC)

    frame_count = int(round(fc)) if fc > 0 else 0
    if fps > 0 and fc > 0:
        duration_seconds = float(fc / fps)
    else:
        duration_seconds = 0.0

    return VideoMetadata(
        frame_count=frame_count,
        fps=fps,
        width=w,
        height=h,
        duration_seconds=duration_seconds,
        fourcc=decode_fourcc(fourcc_raw),
    )


def open_video(path: PathLike) -> tuple[cv2.VideoCapture, VideoMetadata]:
    """Open a video file and return a capture object plus metadata.

    The caller **must** call ``capture.release()`` when finished (or use a context
    pattern in future helpers).

    Parameters
    ----------
    path:
        Absolute or relative path to a video file (no implicit ``video/`` prefix).

    Raises
    ------
    VideoOpenError
        If the path is not a file, OpenCV cannot open the stream, or metadata read fails.
    """
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise VideoOpenError(f"Video file not found: {resolved}")

    cap = cv2.VideoCapture(str(resolved))
    if not cap.isOpened():
        cap.release()
        raise VideoOpenError(f"Could not open video (unsupported or corrupt): {resolved}")

    try:
        meta = _metadata_from_capture(cap)
    except Exception as exc:
        cap.release()
        raise VideoOpenError(f"Failed to read video metadata: {resolved}") from exc

    return cap, meta
