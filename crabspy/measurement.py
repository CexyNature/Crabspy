"""Physical lengths from normalized image coordinates and a quadrat calibration.

This is the canonical math used by the web UI for carapace polylines. The legacy
``crabspy/measure.py`` script uses OpenCV homography + ``constant.DIM`` in a
desktop workflow; here we use **isotropic scaling** from one known edge length
(see :func:`mm_per_px_from_quadrat`), which matches how calibrations are stored
in the database (four corners, ``reference_edge_index``, ``reference_length_mm``).

Coordinates are **normalized** ``(x, y)`` in ``[0, 1]`` relative to the full
frame; pixel lengths use ``ref_width_px`` × ``ref_height_px`` as the image size
at annotation time.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import cv2
import numpy as np


def polyline_path_length_norm(coords: Sequence[tuple[float, float]]) -> float:
    """Sum of Euclidean segment lengths in normalized 0–1 space."""
    if len(coords) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(coords)):
        total += math.hypot(coords[i][0] - coords[i - 1][0], coords[i][1] - coords[i - 1][1])
    return total


def polyline_path_length_px(
    coords: Sequence[tuple[float, float]],
    ref_width_px: int,
    ref_height_px: int,
) -> float:
    """Path length in pixels: for each segment, ``(Δx·W)² + (Δy·H)²`` under sqrt."""
    w = float(ref_width_px)
    h = float(ref_height_px)
    total = 0.0
    for i in range(1, len(coords)):
        dx = (coords[i][0] - coords[i - 1][0]) * w
        dy = (coords[i][1] - coords[i - 1][1]) * h
        total += math.hypot(dx, dy)
    return total


def edge_length_px(
    corners: Sequence[tuple[float, float]],
    edge_index: int,
    ref_width_px: int,
    ref_height_px: int,
) -> float:
    """Length in pixels of quadrat edge ``edge_index`` (corner ``i`` to ``(i+1)%4``)."""
    i = edge_index % 4
    j = (edge_index + 1) % 4
    x0, y0 = corners[i]
    x1, y1 = corners[j]
    dx = (x1 - x0) * float(ref_width_px)
    dy = (y1 - y0) * float(ref_height_px)
    return math.hypot(dx, dy)


def mm_per_px_from_quadrat(
    corners: Sequence[tuple[float, float]],
    reference_edge_index: int,
    reference_length_mm: float,
    ref_width_px: int,
    ref_height_px: int,
) -> float:
    """Isotropic scale: millimetres per pixel using the known-length reference edge."""
    el = edge_length_px(corners, reference_edge_index, ref_width_px, ref_height_px)
    if el <= 1e-12:
        raise ValueError("Reference edge has zero length in pixels.")
    if reference_length_mm <= 0:
        raise ValueError("reference_length_mm must be positive.")
    return reference_length_mm / el


def polyline_length_mm_isotropic(
    corners_norm: Sequence[tuple[float, float]],
    ref_width_px: int,
    ref_height_px: int,
    mm_per_px: float,
) -> float:
    """Total polyline length in mm (same scale along x and y)."""
    path_px = polyline_path_length_px(corners_norm, ref_width_px, ref_height_px)
    return path_px * mm_per_px


def _pixel_points(coords: Sequence[tuple[float, float]], ref_width_px: int, ref_height_px: int) -> np.ndarray:
    return np.array(
        [[x * float(ref_width_px), y * float(ref_height_px)] for x, y in coords],
        dtype=np.float32,
    )


def polyline_length_mm_homography(
    polyline_norm: Sequence[tuple[float, float]],
    quadrat_corners_norm: Sequence[tuple[float, float]],
    reference_edge_index: int,
    reference_length_mm: float,
    ref_width_px: int,
    ref_height_px: int,
) -> float:
    """Polyline length in mm after quadrat rectification via homography.

    The known-length edge defines destination width in mm. Destination height in mm
    is inferred from the adjacent edge length using the same local mm/px scale.
    """
    if len(polyline_norm) < 2:
        return 0.0
    if len(quadrat_corners_norm) != 4:
        raise ValueError("Exactly four quadrat corners are required.")
    mm_per_px = mm_per_px_from_quadrat(
        quadrat_corners_norm,
        reference_edge_index,
        reference_length_mm,
        ref_width_px,
        ref_height_px,
    )
    i = reference_edge_index % 4
    src = [
        quadrat_corners_norm[i],
        quadrat_corners_norm[(i + 1) % 4],
        quadrat_corners_norm[(i + 2) % 4],
        quadrat_corners_norm[(i + 3) % 4],
    ]
    # Use the adjacent edge to estimate rectangle height in mm.
    adjacent_px = edge_length_px(src, 1, ref_width_px, ref_height_px)
    height_mm = adjacent_px * mm_per_px
    width_mm = float(reference_length_mm)
    if width_mm <= 0 or height_mm <= 1e-12:
        raise ValueError("Invalid quadrat dimensions for homography.")
    src_px = _pixel_points(src, ref_width_px, ref_height_px)
    dst_mm = np.array(
        [[0.0, 0.0], [width_mm, 0.0], [width_mm, height_mm], [0.0, height_mm]],
        dtype=np.float32,
    )
    h_mat = cv2.getPerspectiveTransform(src_px, dst_mm)
    pl_px = _pixel_points(polyline_norm, ref_width_px, ref_height_px).reshape(-1, 1, 2)
    pl_mm = cv2.perspectiveTransform(pl_px, h_mat).reshape(-1, 2)
    total = 0.0
    for idx in range(1, len(pl_mm)):
        dx = float(pl_mm[idx, 0] - pl_mm[idx - 1, 0])
        dy = float(pl_mm[idx, 1] - pl_mm[idx - 1, 1])
        total += math.hypot(dx, dy)
    return total


def corners_from_calibration_json(corners: list[Any]) -> list[tuple[float, float]]:
    """Parse stored calibration corners (dicts or objects with ``x_norm`` / ``y_norm``)."""
    out: list[tuple[float, float]] = []
    for c in corners:
        if isinstance(c, dict):
            out.append((float(c["x_norm"]), float(c["y_norm"])))
        else:
            out.append((float(c.x_norm), float(c.y_norm)))
    return out
