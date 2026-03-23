"""Unit tests for CSV bulk import parsing."""

from crabspy_web.models.media import MediaProcessingStatus
from crabspy_web.services.media_csv_import import parse_media_import_csv


def test_parse_minimal_path_only() -> None:
    text = "path\ndemo/a.mp4\n"
    rows, errors = parse_media_import_csv(text)
    assert not errors
    assert len(rows) == 1
    assert rows[0].kwargs["storage_path"] == "demo/a.mp4"
    assert rows[0].kwargs["processing_status"] == MediaProcessingStatus.draft


def test_parse_full_row_ready() -> None:
    text = (
        "video_path,collected_at,sample_code,site_name,location_name,notes\n"
        "uploads/x.mp4,2025-01-15,S1,SiteA,Loc1,note here\n"
    )
    rows, errors = parse_media_import_csv(text)
    assert not errors
    assert len(rows) == 1
    k = rows[0].kwargs
    assert k["processing_status"] == MediaProcessingStatus.ready_for_processing
    assert k["sample_code"] == "S1"
    assert k["site_name"] == "SiteA"
    assert k["location_name"] == "Loc1"
    assert k["notes"] == "note here"
    assert k["collected_at"] is not None


def test_parse_row_error_bad_path() -> None:
    text = "path\n../evil.mp4\n"
    rows, errors = parse_media_import_csv(text)
    assert not rows
    assert errors


def test_parse_missing_header() -> None:
    rows, errors = parse_media_import_csv("a,b\n1,2\n")
    assert not rows
    assert any("path" in e[1].lower() for e in errors)


def test_parse_optional_camera_and_coords() -> None:
    text = (
        "path,collected_at,sample_code,site_name,location_name,camera_id,latitude,longitude\n"
        "v/a.mp4,2025-01-01,A,B,C,CAM1,-19.25,146.82\n"
    )
    rows, errors = parse_media_import_csv(text)
    assert not errors
    assert len(rows) == 1
    k = rows[0].kwargs
    assert k["camera_id"] == "CAM1"
    assert k["latitude"] == -19.25
    assert k["longitude"] == 146.82
    assert k["processing_status"] == MediaProcessingStatus.ready_for_processing


def test_parse_invalid_lat_lon() -> None:
    text = (
        "path,collected_at,sample_code,site_name,location_name,latitude\n"
        "v/a.mp4,2025-01-01,A,B,C,not-a-number\n"
    )
    rows, errors = parse_media_import_csv(text)
    assert not rows
    assert errors
