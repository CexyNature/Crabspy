"""Tests for crabspy.io.video."""

from __future__ import annotations

import numpy as np
import pytest

import cv2

from crabspy.io.video import VideoOpenError, decode_fourcc, open_video


def test_decode_fourcc_zero() -> None:
    assert decode_fourcc(0) == "\x00\x00\x00\x00"


def test_open_video_not_found(tmp_path) -> None:
    with pytest.raises(VideoOpenError, match="not found"):
        open_video(tmp_path / "missing.mp4")


def test_open_video_reads_metadata(tmp_path) -> None:
    path = tmp_path / "t.avi"
    w, h = 64, 48
    fps = 10.0
    n_frames = 15
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        pytest.skip("VideoWriter (MJPG) not available on this platform")
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()
    if path.stat().st_size == 0:
        pytest.skip("VideoWriter produced empty file")

    cap, meta = open_video(path)
    try:
        assert meta.width == w
        assert meta.height == h
        assert meta.fps == pytest.approx(fps, rel=0.05)
        assert meta.frame_count >= n_frames - 3
        assert meta.duration_seconds > 0
        assert len(meta.fourcc) == 4
    finally:
        cap.release()
