"""Calibration (quadrat) API payloads."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class CalibrationCornerIn(BaseModel):
    x_norm: float = Field(..., ge=0.0, le=1.0)
    y_norm: float = Field(..., ge=0.0, le=1.0)


class CalibrationCreate(BaseModel):
    corners: list[CalibrationCornerIn]
    reference_edge_index: int = Field(..., ge=0, le=3)
    reference_length_mm: float = Field(..., gt=0.0)
    frame_index: int | None = Field(default=None, ge=0)
    time_seconds: float | None = None
    ref_width_px: int | None = Field(default=None, ge=1)
    ref_height_px: int | None = Field(default=None, ge=1)
    label: str | None = None

    @model_validator(mode="after")
    def validate_corners_and_ref(self) -> CalibrationCreate:
        if len(self.corners) != 4:
            raise ValueError("Exactly four corners are required (clockwise order).")
        rw, rh = self.ref_width_px, self.ref_height_px
        if (rw is None) ^ (rh is None):
            raise ValueError("ref_width_px and ref_height_px must both be set or both omitted.")
        return self


class CalibrationOut(BaseModel):
    id: str
    source_media_id: str
    frame_index: int | None
    time_seconds: float | None
    corners: list[CalibrationCornerIn]
    reference_edge_index: int
    reference_length_mm: float
    ref_width_px: int | None
    ref_height_px: int | None
    mm_per_px: float
    label: str | None
