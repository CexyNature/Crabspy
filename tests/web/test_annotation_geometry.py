"""Unit tests for annotation path lengths."""

from __future__ import annotations

from crabspy.measurement import polyline_path_length_norm, polyline_path_length_px

from crabspy_web.services.annotation_geometry import edge_norm_and_px


def test_polyline_path_length_norm_right_angle() -> None:
    coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    assert abs(polyline_path_length_norm(coords) - 2.0) < 1e-9


def test_polyline_path_length_px_matches_norm_when_square_pixels() -> None:
    coords = [(0.0, 0.0), (1.0, 0.0)]
    px = polyline_path_length_px(coords, 100, 100)
    assert abs(px - 100.0) < 1e-9


def test_edge_norm_and_px() -> None:
    n, px = edge_norm_and_px(0.0, 0.0, 0.0, 1.0, 100, 200)
    assert abs(n - 1.0) < 1e-9
    assert px is not None
    assert abs(px - 200.0) < 1e-9
    n2, px2 = edge_norm_and_px(0.0, 0.0, 0.0, 1.0, None, None)
    assert abs(n2 - 1.0) < 1e-9
    assert px2 is None
