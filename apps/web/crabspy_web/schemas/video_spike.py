"""Pydantic payloads for the video annotation spike."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VideoSpikeCreate(BaseModel):
    x_norm: float = Field(..., ge=0.0, le=1.0)
    y_norm: float = Field(..., ge=0.0, le=1.0)
    time_seconds: float = Field(..., ge=0.0)
    frame_index: int | None = None


class VideoSpikeOut(BaseModel):
    id: str
    x_norm: float
    y_norm: float
    time_seconds: float
    frame_index: int | None
