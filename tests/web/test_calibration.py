"""Calibration: quadrat POST, linking active calibration, polyline length in mm."""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

from fastapi.testclient import TestClient

from crabspy.measurement import (
    corners_from_calibration_json,
    mm_per_px_from_quadrat,
    polyline_length_mm_homography,
    polyline_length_mm_isotropic,
)
from crabspy_web.app import create_app


def _unit_square_corners() -> list[dict[str, float]]:
    return [
        {"x_norm": 0.0, "y_norm": 0.0},
        {"x_norm": 1.0, "y_norm": 0.0},
        {"x_norm": 1.0, "y_norm": 1.0},
        {"x_norm": 0.0, "y_norm": 1.0},
    ]


def test_calibration_create_image_and_polyline_mm(tmp_path: Path) -> None:
    (tmp_path / "uploads").mkdir(parents=True)
    (tmp_path / "uploads" / "ref.jpg").write_bytes(b"x")
    with TestClient(create_app()) as client:
        client.post(
            "/media/",
            data={"storage_path": "uploads/ref.jpg", "media_kind": "image"},
            follow_redirects=False,
        )
        r = client.get("/media/")
        m = re.search(r'href="/media/([0-9a-f-]{36})"', r.text)
        assert m is not None
        mid = m.group(1)

        r_cal = client.post(
            f"/media/{mid}/calibration",
            json={
                "corners": _unit_square_corners(),
                "reference_edge_index": 0,
                "reference_length_mm": 100.0,
                "ref_width_px": 800,
                "ref_height_px": 600,
                "label": "test quadrat",
            },
        )
        assert r_cal.status_code == 201
        cal = r_cal.json()
        assert cal["mm_per_px"] > 0
        assert cal["reference_length_mm"] == 100.0

        r_poly = client.post(
            f"/media/{mid}/annotations",
            json={
                "kind": "polyline",
                "points": [
                    {"x_norm": 0.0, "y_norm": 0.5},
                    {"x_norm": 1.0, "y_norm": 0.5},
                ],
                "ref_width_px": 800,
                "ref_height_px": 600,
            },
        )
        assert r_poly.status_code == 201
        data = r_poly.json()
        assert data["path_length_mm"] is not None
        assert abs(data["path_length_mm"] - 100.0) < 1e-9
        assert data["measurement_mode_used"] == "homography"


def test_active_calibration_linked_from_other_media(tmp_path: Path) -> None:
    (tmp_path / "uploads").mkdir(parents=True)
    (tmp_path / "uploads" / "a.jpg").write_bytes(b"a")
    (tmp_path / "uploads" / "b.jpg").write_bytes(b"b")
    with TestClient(create_app()) as client:
        client.post(
            "/media/",
            data={"storage_path": "uploads/a.jpg", "media_kind": "image"},
            follow_redirects=False,
        )
        client.post(
            "/media/",
            data={"storage_path": "uploads/b.jpg", "media_kind": "image"},
            follow_redirects=False,
        )
        r = client.get("/media/")
        ids = list(dict.fromkeys(re.findall(r'href="/media/([0-9a-f-]{36})"', r.text)))
        assert len(ids) >= 2
        # List is newest-first; first POST was a.jpg, second was b.jpg.
        mid_b, mid_a = ids[0], ids[1]

        r_cal = client.post(
            f"/media/{mid_a}/calibration",
            json={
                "corners": _unit_square_corners(),
                "reference_edge_index": 0,
                "reference_length_mm": 50.0,
                "ref_width_px": 400,
                "ref_height_px": 300,
            },
        )
        assert r_cal.status_code == 201
        cal_id = r_cal.json()["id"]

        form_base = {
            "storage_path": "uploads/b.jpg",
            "processing_status": "draft",
            "media_kind": "image",
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
            "active_calibration_id": cal_id,
        }
        r_save = client.post(f"/media/{mid_b}", data=form_base, follow_redirects=False)
        assert r_save.status_code == 303

        r_poly = client.post(
            f"/media/{mid_b}/annotations",
            json={
                "kind": "polyline",
                "points": [
                    {"x_norm": 0.0, "y_norm": 0.25},
                    {"x_norm": 1.0, "y_norm": 0.25},
                ],
                "ref_width_px": 400,
                "ref_height_px": 300,
            },
        )
        assert r_poly.status_code == 201
        data = r_poly.json()
        assert data["path_length_mm"] is not None
        assert abs(data["path_length_mm"] - 50.0) < 1e-6


def test_media_measurement_mode_switches_mm_math(tmp_path: Path) -> None:
    (tmp_path / "uploads").mkdir(parents=True)
    (tmp_path / "uploads" / "skew.jpg").write_bytes(b"x")
    with TestClient(create_app()) as client:
        client.post(
            "/media/",
            data={"storage_path": "uploads/skew.jpg", "media_kind": "image"},
            follow_redirects=False,
        )
        r = client.get("/media/")
        m = re.search(r'href="/media/([0-9a-f-]{36})"', r.text)
        assert m is not None
        mid = m.group(1)

        corners = [
            {"x_norm": 0.10, "y_norm": 0.10},
            {"x_norm": 0.90, "y_norm": 0.20},
            {"x_norm": 0.82, "y_norm": 0.88},
            {"x_norm": 0.20, "y_norm": 0.80},
        ]
        r_cal = client.post(
            f"/media/{mid}/calibration",
            json={
                "corners": corners,
                "reference_edge_index": 0,
                "reference_length_mm": 100.0,
                "ref_width_px": 800,
                "ref_height_px": 600,
            },
        )
        assert r_cal.status_code == 201
        cal = r_cal.json()

        form_base = {
            "storage_path": "uploads/skew.jpg",
            "processing_status": "draft",
            "media_kind": "image",
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
            "active_calibration_id": cal["id"],
        }

        # 1) Isotropic mode
        r_mode_iso = client.post(
            f"/media/{mid}",
            data={**form_base, "measurement_mode": "isotropic"},
            follow_redirects=False,
        )
        assert r_mode_iso.status_code == 303
        polyline = [
            {"x_norm": 0.15, "y_norm": 0.25},
            {"x_norm": 0.85, "y_norm": 0.70},
        ]
        r_poly_iso = client.post(
            f"/media/{mid}/annotations",
            json={
                "kind": "polyline",
                "points": polyline,
                "ref_width_px": 800,
                "ref_height_px": 600,
            },
        )
        assert r_poly_iso.status_code == 201
        mm_iso_resp = r_poly_iso.json()["path_length_mm"]
        assert mm_iso_resp is not None

        # 2) Homography mode
        r_mode_h = client.post(
            f"/media/{mid}",
            data={**form_base, "measurement_mode": "homography"},
            follow_redirects=False,
        )
        assert r_mode_h.status_code == 303
        r_poly_h = client.post(
            f"/media/{mid}/annotations",
            json={
                "kind": "polyline",
                "points": polyline,
                "ref_width_px": 800,
                "ref_height_px": 600,
            },
        )
        assert r_poly_h.status_code == 201
        mm_h_resp = r_poly_h.json()["path_length_mm"]
        assert mm_h_resp is not None

        corners_t = corners_from_calibration_json(corners)
        pl_t = [(p["x_norm"], p["y_norm"]) for p in polyline]
        mm_px = mm_per_px_from_quadrat(corners_t, 0, 100.0, 800, 600)
        mm_iso = polyline_length_mm_isotropic(pl_t, 800, 600, mm_px)
        mm_h = polyline_length_mm_homography(pl_t, corners_t, 0, 100.0, 800, 600)
        assert abs(mm_iso_resp - round(mm_iso, 1)) < 1e-9
        assert abs(mm_h_resp - round(mm_h, 1)) < 1e-9
        assert abs(mm_h_resp - mm_iso_resp) > 1e-3
        assert r_poly_iso.json()["measurement_mode_used"] == "isotropic"
        assert r_poly_h.json()["measurement_mode_used"] == "homography"


def _first_path_length_mm_from_export(csv_text: str, annotation_id: str) -> float | None:
    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        if row.get("annotation_id") == annotation_id:
            raw = row.get("path_length_mm") or ""
            if raw == "":
                return None
            return float(raw)
    return None


def test_stored_polyline_mm_refreshes_in_csv_after_measurement_mode_change(tmp_path: Path) -> None:
    """Saving media with a new measurement mode updates stored path_length_mm for existing polylines."""
    (tmp_path / "uploads").mkdir(parents=True)
    (tmp_path / "uploads" / "skew.jpg").write_bytes(b"x")
    with TestClient(create_app()) as client:
        client.post(
            "/media/",
            data={"storage_path": "uploads/skew.jpg", "media_kind": "image"},
            follow_redirects=False,
        )
        r = client.get("/media/")
        m = re.search(r'href="/media/([0-9a-f-]{36})"', r.text)
        assert m is not None
        mid = m.group(1)

        corners = [
            {"x_norm": 0.10, "y_norm": 0.10},
            {"x_norm": 0.90, "y_norm": 0.20},
            {"x_norm": 0.82, "y_norm": 0.88},
            {"x_norm": 0.20, "y_norm": 0.80},
        ]
        r_cal = client.post(
            f"/media/{mid}/calibration",
            json={
                "corners": corners,
                "reference_edge_index": 0,
                "reference_length_mm": 100.0,
                "ref_width_px": 800,
                "ref_height_px": 600,
            },
        )
        assert r_cal.status_code == 201
        cal = r_cal.json()

        form_base = {
            "storage_path": "uploads/skew.jpg",
            "processing_status": "draft",
            "media_kind": "image",
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
            "active_calibration_id": cal["id"],
        }

        client.post(
            f"/media/{mid}",
            data={**form_base, "measurement_mode": "isotropic"},
            follow_redirects=False,
        )
        polyline = [
            {"x_norm": 0.15, "y_norm": 0.25},
            {"x_norm": 0.85, "y_norm": 0.70},
        ]
        r_poly = client.post(
            f"/media/{mid}/annotations",
            json={
                "kind": "polyline",
                "points": polyline,
                "ref_width_px": 800,
                "ref_height_px": 600,
            },
        )
        assert r_poly.status_code == 201
        aid = r_poly.json()["id"]

        csv_iso = client.get("/media/export_annotations.csv").text
        mm_iso = _first_path_length_mm_from_export(csv_iso, aid)
        assert mm_iso is not None

        client.post(
            f"/media/{mid}",
            data={**form_base, "measurement_mode": "homography"},
            follow_redirects=False,
        )
        csv_h = client.get("/media/export_annotations.csv").text
        mm_h = _first_path_length_mm_from_export(csv_h, aid)
        assert mm_h is not None
        assert abs(mm_h - mm_iso) > 1e-3
