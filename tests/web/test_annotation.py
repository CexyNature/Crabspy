"""Annotations (Phase 3b): API, view HTML, CSV export."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from crabspy_web.app import create_app


def test_annotation_point_create_view_delete(tmp_path: Path) -> None:
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
            f"/media/{mid}/annotations",
            json={
                "kind": "point",
                "points": [{"x_norm": 0.25, "y_norm": 0.75}],
                "time_seconds": 1.5,
                "frame_index": 45,
            },
        )
        assert r_post.status_code == 201
        data = r_post.json()
        assert data["kind"] == "point"
        assert data["points"][0]["x_norm"] == 0.25
        assert data["points"][0]["y_norm"] == 0.75
        aid = data["id"]

        r_view = client.get(f"/media/{mid}/view")
        assert r_view.status_code == 200
        assert "data-annotation-video-root" in r_view.text
        assert aid in r_view.text

        r_del = client.post(f"/media/{mid}/annotations/{aid}/delete")
        assert r_del.status_code == 200
        assert r_del.json() == {"ok": True}

        r_view2 = client.get(f"/media/{mid}/view")
        assert r_view2.status_code == 200
        assert aid not in r_view2.text


def test_annotation_image_point_without_time(tmp_path: Path) -> None:
    (tmp_path / "uploads").mkdir(parents=True)
    (tmp_path / "uploads" / "p.jpg").write_bytes(b"x")
    with TestClient(create_app()) as client:
        client.post(
            "/media/",
            data={"storage_path": "uploads/p.jpg", "media_kind": "image"},
            follow_redirects=False,
        )
        r = client.get("/media/")
        m = re.search(r'href="/media/([0-9a-f-]{36})/view"', r.text)
        assert m is not None
        mid = m.group(1)
        r_post = client.post(
            f"/media/{mid}/annotations",
            json={
                "kind": "point",
                "points": [{"x_norm": 0.4, "y_norm": 0.6}],
            },
        )
        assert r_post.status_code == 201
        assert r_post.json()["kind"] == "point"


def test_annotation_rejects_image_media_with_time(tmp_path: Path) -> None:
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
            f"/media/{mid}/annotations",
            json={
                "kind": "point",
                "points": [{"x_norm": 0.5, "y_norm": 0.5}],
                "time_seconds": 0.0,
            },
        )
        assert r_post.status_code == 400


def test_annotation_polyline_video(tmp_path: Path) -> None:
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
            f"/media/{mid}/annotations",
            json={
                "kind": "polyline",
                "points": [
                    {"x_norm": 0.1, "y_norm": 0.2},
                    {"x_norm": 0.3, "y_norm": 0.4},
                ],
                "time_seconds": 2.0,
                "frame_index": 50,
            },
        )
        assert r_post.status_code == 201
        data = r_post.json()
        assert data["kind"] == "polyline"
        assert len(data["points"]) == 2

        r_view = client.get(f"/media/{mid}/view")
        assert r_view.status_code == 200
        assert "annotation-polyline-wrap" in r_view.text


def test_annotation_polyline_image(tmp_path: Path) -> None:
    (tmp_path / "uploads").mkdir(parents=True)
    (tmp_path / "uploads" / "p.jpg").write_bytes(b"x")
    with TestClient(create_app()) as client:
        client.post(
            "/media/",
            data={"storage_path": "uploads/p.jpg", "media_kind": "image"},
            follow_redirects=False,
        )
        r = client.get("/media/")
        m = re.search(r'href="/media/([0-9a-f-]{36})/view"', r.text)
        assert m is not None
        mid = m.group(1)
        r_post = client.post(
            f"/media/{mid}/annotations",
            json={
                "kind": "polyline",
                "points": [
                    {"x_norm": 0.05, "y_norm": 0.95},
                    {"x_norm": 0.95, "y_norm": 0.05},
                ],
            },
        )
        assert r_post.status_code == 201
        assert r_post.json()["kind"] == "polyline"


def test_media_image_view_includes_annotation_shell(tmp_path: Path) -> None:
    (tmp_path / "uploads").mkdir(parents=True)
    (tmp_path / "uploads" / "photo.jpg").write_bytes(b"x")
    with TestClient(create_app()) as client:
        client.post(
            "/media/",
            data={"storage_path": "uploads/photo.jpg", "media_kind": "image"},
            follow_redirects=False,
        )
        r = client.get("/media/")
        m = re.search(r'href="/media/([0-9a-f-]{36})/view"', r.text)
        assert m is not None
        mid = m.group(1)
        r_view = client.get(f"/media/{mid}/view")
        assert r_view.status_code == 200
        assert 'data-media-image-viewer="1"' in r_view.text
        assert "data-annotation-image-root" in r_view.text
        assert "<img " in r_view.text


def test_export_annotations_csv(tmp_path: Path) -> None:
    (tmp_path / "uploads").mkdir(parents=True)
    (tmp_path / "uploads" / "a.mp4").write_bytes(b"x")
    with TestClient(create_app()) as client:
        client.post(
            "/media/",
            data={"storage_path": "uploads/a.mp4", "media_kind": "video"},
            follow_redirects=False,
        )
        r = client.get("/media/")
        m = re.search(r'href="/media/([0-9a-f-]{36})/view"', r.text)
        mid = m.group(1)
        client.post(
            f"/media/{mid}/annotations",
            json={
                "kind": "point",
                "points": [{"x_norm": 0.1, "y_norm": 0.2}],
                "time_seconds": 0.5,
            },
        )
        r_csv = client.get("/media/export_annotations.csv")
        assert r_csv.status_code == 200
        body = r_csv.text
        assert "annotation_id" in body
        assert "x_norm" in body
        assert mid in body
