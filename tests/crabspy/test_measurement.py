"""Tests for ``crabspy.measurement`` (quadrat + polyline lengths)."""

from __future__ import annotations

import pytest

from crabspy.measurement import (
    edge_length_px,
    mm_per_px_from_quadrat,
    polyline_length_mm_homography,
    polyline_length_mm_isotropic,
    polyline_path_length_norm,
    polyline_path_length_px,
)


def test_polyline_path_length_norm_right_angle() -> None:
    coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    assert abs(polyline_path_length_norm(coords) - 2.0) < 1e-9


def test_polyline_path_length_px_horizontal() -> None:
    coords = [(0.0, 0.0), (1.0, 0.0)]
    px = polyline_path_length_px(coords, 100, 100)
    assert abs(px - 100.0) < 1e-9


def test_mm_per_px_unit_square_top_edge() -> None:
    corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    mm_px = mm_per_px_from_quadrat(corners, 0, 100.0, 800, 600)
    assert abs(mm_px - 100.0 / 800.0) < 1e-9


def test_polyline_length_matches_full_width() -> None:
    corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    mm_px = mm_per_px_from_quadrat(corners, 0, 100.0, 800, 600)
    pl = [(0.0, 0.5), (1.0, 0.5)]
    mm = polyline_length_mm_isotropic(pl, 800, 600, mm_px)
    assert abs(mm - 100.0) < 1e-6


def test_polyline_length_homography_matches_full_width() -> None:
    corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    pl = [(0.0, 0.5), (1.0, 0.5)]
    mm = polyline_length_mm_homography(pl, corners, 0, 100.0, 800, 600)
    assert abs(mm - 100.0) < 1e-6


def test_edge_length_zero_raises() -> None:
    corners = [(0.0, 0.0), (0.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    with pytest.raises(ValueError, match="zero"):
        mm_per_px_from_quadrat(corners, 0, 10.0, 100, 100)
