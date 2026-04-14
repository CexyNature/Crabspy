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


def test_annotation_video_infers_frame_index_when_frame_rate_set(tmp_path: Path) -> None:
    """Server fills frame_index from time_seconds × media.frame_rate when client omits it."""
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
        client.post(
            f"/media/{mid}",
            data={
                "storage_path": "uploads/clip.mp4",
                "processing_status": "draft",
                "media_kind": "video",
                "collected_at": "",
                "sample_code": "",
                "site_name": "",
                "location_name": "",
                "notes": "",
                "camera_id": "",
                "deployment_time": "",
                "deployment_type": "",
                "latitude": "",
                "longitude": "",
                "original_filename": "",
                "mime_type": "",
                "checksum_sha256": "",
                "width_px": "",
                "height_px": "",
                "duration_seconds": "",
                "frame_rate": "30",
                "active_calibration_id": "",
                "measurement_mode": "homography",
            },
            follow_redirects=False,
        )
        r_post = client.post(
            f"/media/{mid}/annotations",
            json={
                "kind": "point",
                "points": [{"x_norm": 0.25, "y_norm": 0.75}],
                "time_seconds": 2.0,
            },
        )
        assert r_post.status_code == 201
        assert r_post.json()["frame_index"] == 60


def test_annotation_video_infers_frame_index_when_frame_rate_missing_uses_default_fps(
    tmp_path: Path,
) -> None:
    """When media.frame_rate is unset, server and UI use DEFAULT_VIDEO_FPS (30) for frame_index."""
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
                "time_seconds": 1.0,
            },
        )
        assert r_post.status_code == 201
        assert r_post.json()["frame_index"] == 30


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


def test_annotation_label_and_ref_in_response(tmp_path: Path) -> None:
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
        mid = m.group(1)
        r_post = client.post(
            f"/media/{mid}/annotations",
            json={
                "kind": "point",
                "points": [{"x_norm": 0.1, "y_norm": 0.2}],
                "time_seconds": 0.5,
                "label": "snout",
                "ref_width_px": 1920,
                "ref_height_px": 1080,
            },
        )
        assert r_post.status_code == 201
        data = r_post.json()
        assert data["label"] == "snout"
        assert data["ref_width_px"] == 1920
        assert data["ref_height_px"] == 1080


def test_annotation_rejects_partial_ref_dimensions(tmp_path: Path) -> None:
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
        mid = m.group(1)
        r_post = client.post(
            f"/media/{mid}/annotations",
            json={
                "kind": "point",
                "points": [{"x_norm": 0.1, "y_norm": 0.2}],
                "time_seconds": 0.5,
                "ref_width_px": 100,
            },
        )
        assert r_post.status_code == 422


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
        client.post(
            f"/media/{mid}/annotations",
            json={
                "kind": "polyline",
                "points": [
                    {"x_norm": 0.0, "y_norm": 0.0},
                    {"x_norm": 1.0, "y_norm": 0.0},
                ],
                "time_seconds": 0.0,
                "ref_width_px": 100,
                "ref_height_px": 100,
            },
        )
        r_csv = client.get("/media/export_annotations.csv")
        assert r_csv.status_code == 200
        body = r_csv.text
        assert "annotation_id" in body
        assert "x_norm" in body
        assert "path_length_norm" in body
        assert "path_length_px" in body
        assert "path_length_mm" in body
        assert "measurement_mode_used" in body
        assert "media_storage_path" in body
        assert "media_sample_code" in body
        assert "calibration_mm_per_px" in body
        assert "calibration_reference_length_mm" in body
        assert "edge_length_norm" in body
        assert "edge_length_px" in body
        assert mid in body
        assert "1.0" in body
