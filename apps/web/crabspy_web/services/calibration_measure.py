"""Resolve mm lengths for annotations using the media's active calibration."""

from __future__ import annotations

import json

from crabspy_web.models.annotation import Annotation, AnnotationKind
from crabspy_web.models.calibration import Calibration
from crabspy_web.models.media import Media, MediaMeasurementMode
from crabspy_web.services.calibration_math import (
    corners_from_json,
    polyline_length_mm_homography,
    polyline_length_mm_isotropic,
)


def path_length_mm_for_polyline(
    ann: Annotation,
    media: Media,
    calibration: Calibration | None,
) -> float | None:
    """Carapace polyline length in mm if calibration + ref dimensions are available."""
    if calibration is None or ann.kind != AnnotationKind.polyline:
        return None
    pts = sorted(ann.points, key=lambda p: p.order_index)
    if len(pts) < 2:
        return None
    coords = [(p.x_norm, p.y_norm) for p in pts]
    rw = ann.ref_width_px or calibration.ref_width_px
    rh = ann.ref_height_px or calibration.ref_height_px
    if rw is None or rh is None:
        return None
    mode = getattr(media, "measurement_mode", MediaMeasurementMode.homography)
    if mode == MediaMeasurementMode.isotropic:
        return polyline_length_mm_isotropic(coords, rw, rh, calibration.mm_per_px)
    raw_corners = corners_from_json(json.loads(calibration.corners_json))
    try:
        return polyline_length_mm_homography(
            coords,
            raw_corners,
            calibration.reference_edge_index,
            calibration.reference_length_mm,
            rw,
            rh,
        )
    except Exception:
        # Fallback to isotropic scale if homography inputs are degenerate.
        return polyline_length_mm_isotropic(coords, rw, rh, calibration.mm_per_px)


def quadrat_overlay_points(calibration: Calibration) -> list[tuple[float, float]]:
    """Normalized corners for drawing (from stored JSON)."""
    raw = json.loads(calibration.corners_json)
    return corners_from_json(raw)
