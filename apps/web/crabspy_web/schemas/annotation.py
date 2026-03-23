"""Pydantic payloads for annotations (Phase 3b)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class AnnotationPointIn(BaseModel):
    x_norm: float = Field(..., ge=0.0, le=1.0)
    y_norm: float = Field(..., ge=0.0, le=1.0)


class AnnotationCreate(BaseModel):
    kind: Literal["point", "polyline"]
    points: list[AnnotationPointIn]
    label: str | None = None
    frame_index: int | None = None
    time_seconds: float | None = None
    ref_width_px: int | None = Field(default=None, ge=1)
    ref_height_px: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_annotation_create(self) -> AnnotationCreate:
        k = self.kind
        n = len(self.points)
        if k == "point" and n != 1:
            raise ValueError("kind 'point' requires exactly one vertex.")
        if k == "polyline" and n < 2:
            raise ValueError("kind 'polyline' requires at least two vertices.")
        rw, rh = self.ref_width_px, self.ref_height_px
        if (rw is None) ^ (rh is None):
            raise ValueError("ref_width_px and ref_height_px must both be set or both omitted.")
        return self


class AnnotationPointOut(BaseModel):
    x_norm: float
    y_norm: float


class AnnotationOut(BaseModel):
    id: str
    kind: str
    points: list[AnnotationPointOut]
    label: str | None
    time_seconds: float | None
    frame_index: int | None
    ref_width_px: int | None = None
    ref_height_px: int | None = None
    path_length_mm: float | None = None
