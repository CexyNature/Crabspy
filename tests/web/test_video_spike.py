"""Thin video annotation spike: API + view HTML."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from crabspy_web.app import create_app


def test_video_spike_create_view_delete(tmp_path: Path) -> None:
    (tmp_path / "uploads").mkdir(parents=True)
    (tmp_path / "uploads" / "clip.mp4").write_bytes(b"%fakevideo")
    with TestClient(create_app()) as client:
        client.post(
            "/media/",
            data={"storage_path": "uploads/clip.mp4", "media_kind": "video"},
            follow_redirects=False,
        )
        r = client.get("/media/")
        m = re.search(r'href="/media/([0-9a-f-]{36})/view"', r.text)
        assert m is not None
        mid = m.group(1)

        r_post = client.post(
            f"/media/{mid}/video-spike",
            json={
                "x_norm": 0.25,
                "y_norm": 0.75,
                "time_seconds": 1.5,
                "frame_index": 45,
            },
        )
        assert r_post.status_code == 201
        data = r_post.json()
        assert data["x_norm"] == 0.25
        assert data["y_norm"] == 0.75
        pid = data["id"]

        r_view = client.get(f"/media/{mid}/view")
        assert r_view.status_code == 200
        assert "data-video-spike-root" in r_view.text
        assert pid in r_view.text

        r_del = client.post(f"/media/{mid}/video-spike/{pid}/delete")
        assert r_del.status_code == 200
        assert r_del.json() == {"ok": True}

        r_view2 = client.get(f"/media/{mid}/view")
        assert r_view2.status_code == 200
        assert pid not in r_view2.text


def test_video_spike_rejects_image_media(tmp_path: Path) -> None:
    (tmp_path / "uploads").mkdir(parents=True)
    (tmp_path / "uploads" / "x.jpg").write_bytes(b"x")
    with TestClient(create_app()) as client:
        client.post(
            "/media/",
            data={"storage_path": "uploads/x.jpg", "media_kind": "image"},
            follow_redirects=False,
        )
        r = client.get("/media/")
        m = re.search(r'href="/media/([0-9a-f-]{36})/view"', r.text)
        assert m is not None
        mid = m.group(1)
        r_post = client.post(
            f"/media/{mid}/video-spike",
            json={"x_norm": 0.5, "y_norm": 0.5, "time_seconds": 0.0},
        )
        assert r_post.status_code == 400
