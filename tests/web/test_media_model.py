"""Roundtrip tests for the ``media`` table (SQLite in-memory; mirrors PostgreSQL-oriented model)."""

from datetime import UTC, datetime

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from crabspy_web.models import Base, Media, MediaKind, MediaProcessingStatus


def test_media_insert_and_query() -> None:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    with factory() as session:
        row = Media(
            storage_path="uploads/example/sample.mp4",
            original_filename="sample.mp4",
            mime_type="video/mp4",
            media_kind=MediaKind.video,
            processing_status=MediaProcessingStatus.draft,
        )
        session.add(row)
        session.commit()
        media_id = row.id

    with factory() as session:
        loaded = session.get(Media, media_id)
        assert loaded is not None
        assert loaded.storage_path.endswith("sample.mp4")
        assert loaded.media_kind == MediaKind.video
        assert loaded.processing_status == MediaProcessingStatus.draft


def test_media_ready_for_processing_metadata() -> None:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    collected = datetime(2025, 6, 1, 12, 0, tzinfo=UTC)
    with factory() as session:
        row = Media(
            storage_path="uploads/site_a/plot1/img.jpg",
            media_kind=MediaKind.image,
            processing_status=MediaProcessingStatus.ready_for_processing,
            collected_at=collected,
            sample_code="S-001",
            site_name="Mudflat A",
            location_name="Quadrat 3",
            notes="Low tide",
            width_px=1920,
            height_px=1080,
        )
        session.add(row)
        session.commit()

    with factory() as session:
        m = session.scalars(select(Media)).first()
        assert m is not None
        assert m.sample_code == "S-001"
        assert m.site_name == "Mudflat A"
        assert m.collected_at is not None
        # SQLite often returns naive datetimes; normalize for comparison.
        ca = m.collected_at if m.collected_at.tzinfo else m.collected_at.replace(tzinfo=UTC)
        assert ca == collected
