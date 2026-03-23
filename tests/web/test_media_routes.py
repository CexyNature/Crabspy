"""HTTP tests for media list, create, and CSV export."""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from crabspy_web.app import create_app


def test_media_list_empty() -> None:
    with TestClient(create_app()) as client:
        r = client.get("/media/")
        assert r.status_code == 200
        assert "No media rows yet" in r.text


def test_create_draft_and_list() -> None:
    with TestClient(create_app()) as client:
        r = client.post(
            "/media/",
            data={
                "storage_path": "uploads/demo/clip.mp4",
                "original_filename": "clip.mp4",
                "media_kind": "video",
            },
            follow_redirects=False,
        )
        assert r.status_code == 303

        r2 = client.get("/media/")
        assert r2.status_code == 200
        assert "uploads/demo/clip.mp4" in r2.text
        assert "draft" in r2.text


def test_export_csv_headers() -> None:
    with TestClient(create_app()) as client:
        client.post(
            "/media/",
            data={"storage_path": "uploads/a.jpg", "media_kind": "image"},
            follow_redirects=False,
        )
        r = client.get("/media/export.csv")
        assert r.status_code == 200
        assert "text/csv" in r.headers.get("content-type", "")
        body = r.content.decode("utf-8")
        assert "storage_path" in body
        assert "uploads/a.jpg" in body


def test_settings_database_page() -> None:
    with TestClient(create_app()) as client:
        r = client.get("/settings/database")
        assert r.status_code == 200
        assert "Project database" in r.text
        assert "sqlite" in r.text.lower()


def test_media_csv_import_endpoint() -> None:
    csv_body = (
        "storage_path,collected_at,sample_code,site_name,location_name,notes\n"
        "uploads/t/one.mp4,2025-03-01,SC,S,L,N\n"
    ).encode("utf-8")
    with TestClient(create_app()) as client:
        r = client.post(
            "/media/import",
            files={"file": ("batch.csv", csv_body, "text/csv")},
        )
        assert r.status_code == 200
        assert "Imported:" in r.text
        r2 = client.get("/media/")
        assert r2.status_code == 200
        assert "uploads/t/one.mp4" in r2.text
        assert "ready_for_processing" in r2.text


def test_media_detail_edit_and_delete() -> None:
    with TestClient(create_app()) as client:
        client.post(
            "/media/",
            data={"storage_path": "disk/video/x.mp4", "media_kind": "video"},
            follow_redirects=False,
        )
        r = client.get("/media/")
        m = re.search(r'href="/media/([0-9a-f-]{36})"', r.text)
        assert m is not None
        mid = m.group(1)

        r_detail = client.get(f"/media/{mid}")
        assert r_detail.status_code == 200
        assert "disk/video/x.mp4" in r_detail.text

        r_save = client.post(
            f"/media/{mid}",
            data={
                "storage_path": "disk/video/x.mp4",
                "processing_status": "ready_for_processing",
                "media_kind": "video",
                "collected_at": "2025-01-01",
                "sample_code": "S",
                "site_name": "A",
                "location_name": "B",
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
                "frame_rate": "",
            },
            follow_redirects=False,
        )
        assert r_save.status_code == 303

        r_del = client.post(f"/media/{mid}/delete", follow_redirects=False)
        assert r_del.status_code == 303

        r_list = client.get("/media/")
        assert r_list.status_code == 200
        assert "No media rows yet" in r_list.text


def test_media_edit_ready_without_core_fields_shows_error() -> None:
    with TestClient(create_app()) as client:
        client.post("/media/", data={"storage_path": "a/b.mp4", "media_kind": "video"})
        r = client.get("/media/")
        m = re.search(r'href="/media/([0-9a-f-]{36})"', r.text)
        assert m is not None
        mid = m.group(1)
        r_bad = client.post(
            f"/media/{mid}",
            data={
                "storage_path": "a/b.mp4",
                "processing_status": "ready_for_processing",
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
                "frame_rate": "",
            },
        )
        assert r_bad.status_code == 200
        assert "ready for processing" in r_bad.text.lower()


def test_media_import_duplicate_skipped() -> None:
    csv_body = (
        "path,collected_at,sample_code,site_name,location_name\n"
        "uploads/dup/x.mp4,2025-01-01,A,B,C\n"
    ).encode("utf-8")
    with TestClient(create_app()) as client:
        client.post("/media/import", files={"file": ("a.csv", csv_body, "text/csv")})
        r2 = client.post("/media/import", files={"file": ("b.csv", csv_body, "text/csv")})
        assert r2.status_code == 200
        assert "Skipped" in r2.text or "duplicate" in r2.text.lower()
