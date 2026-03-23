"""Registered media (images / video) with study metadata and technical file fields."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from crabspy_web.models.base import Base


class MediaProcessingStatus(str, enum.Enum):
    """Draft rows may omit some fields; processing must not run until ``ready_for_processing``."""

    draft = "draft"
    ready_for_processing = "ready_for_processing"


class MediaKind(str, enum.Enum):
    image = "image"
    video = "video"
    unknown = "unknown"


class MediaMeasurementMode(str, enum.Enum):
    isotropic = "isotropic"
    homography = "homography"


def _enum_values(enum_cls: type[enum.Enum]) -> list[str]:
    return [member.value for member in enum_cls]


class Media(Base):
    """One media file and its study + technical metadata (see docs/crabspy-rebuild-plan.md)."""

    __tablename__ = "media"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    processing_status: Mapped[MediaProcessingStatus] = mapped_column(
        Enum(MediaProcessingStatus, values_callable=_enum_values, native_enum=False),
        nullable=False,
        default=MediaProcessingStatus.draft,
    )

    media_kind: Mapped[MediaKind] = mapped_column(
        Enum(MediaKind, values_callable=_enum_values, native_enum=False),
        nullable=False,
        default=MediaKind.unknown,
    )
    measurement_mode: Mapped[MediaMeasurementMode] = mapped_column(
        Enum(MediaMeasurementMode, values_callable=_enum_values, native_enum=False),
        nullable=False,
        default=MediaMeasurementMode.homography,
        server_default=MediaMeasurementMode.homography.value,
    )

    # Study metadata (required before processing when status is ready_for_processing — enforced in services).
    collected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    sample_code: Mapped[str | None] = mapped_column(String(255), nullable=True)
    site_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    location_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Optional capture / deployment metadata (nullable; empty values do not block ready_for_processing).
    camera_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    deployment_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deployment_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Technical file information
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    original_filename: Mapped[str | None] = mapped_column(String(512), nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    width_px: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height_px: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    frame_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    checksum_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Use this calibration's mm/px when interpreting carapace polylines on this media (may point to any calibration row).
    active_calibration_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("calibration.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    active_calibration: Mapped["Calibration | None"] = relationship(
        "Calibration",
        foreign_keys=[active_calibration_id],
    )
    calibrations_defined: Mapped[list["Calibration"]] = relationship(
        "Calibration",
        foreign_keys="Calibration.source_media_id",
        back_populates="source_media",
    )
