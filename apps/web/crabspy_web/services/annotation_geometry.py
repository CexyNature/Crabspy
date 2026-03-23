"""Polyline length in normalized and pixel space (ref dimensions at annotation time)."""

from __future__ import annotations

import math

from crabspy_web.models.annotation import Annotation, AnnotationKind


def polyline_path_length_norm(coords: list[tuple[float, float]]) -> float:
    """Sum of Euclidean segment lengths in normalized 0–1 space."""
    if len(coords) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(coords)):
        total += math.hypot(coords[i][0] - coords[i - 1][0], coords[i][1] - coords[i - 1][1])
    return total


def polyline_path_length_px(coords: list[tuple[float, float]], ref_width_px: int, ref_height_px: int) -> float:
    """Path length in pixels: segment (Δx·W)² + (Δy·H)² per edge."""
    w = float(ref_width_px)
    h = float(ref_height_px)
    total = 0.0
    for i in range(1, len(coords)):
        dx = (coords[i][0] - coords[i - 1][0]) * w
        dy = (coords[i][1] - coords[i - 1][1]) * h
        total += math.hypot(dx, dy)
    return total


def edge_norm_and_px(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    ref_width_px: int | None,
    ref_height_px: int | None,
) -> tuple[float, float | None]:
    """Edge from (x0,y0) to (x1,y1). Pixel length only if ref dims present."""
    n = math.hypot(x1 - x0, y1 - y0)
    if ref_width_px is None or ref_height_px is None:
        return n, None
    dx = (x1 - x0) * float(ref_width_px)
    dy = (y1 - y0) * float(ref_height_px)
    return n, math.hypot(dx, dy)


def annotation_path_lengths(ann: Annotation, coords: list[tuple[float, float]]) -> tuple[float, float | None]:
    """Total path length for this annotation (0 for a single point)."""
    if ann.kind == AnnotationKind.point:
        return 0.0, None
    if len(coords) < 2:
        return 0.0, None
    pn = polyline_path_length_norm(coords)
    if ann.ref_width_px is not None and ann.ref_height_px is not None:
        return pn, polyline_path_length_px(coords, ann.ref_width_px, ann.ref_height_px)
    return pn, None
