"""HTTP tests for media list, create, and CSV export."""

from __future__ import annotations

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
