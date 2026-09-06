from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, inspect, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    camera_id: Mapped[str] = mapped_column(String(100), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    event_type: Mapped[str] = mapped_column(String(100))
    severity: Mapped[str] = mapped_column(String(20))
    track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    snapshot: Mapped[str | None] = mapped_column(String(500), nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    object_class: Mapped[str | None] = mapped_column(String(100), nullable=True)
    zone_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    bbox_json: Mapped[str | None] = mapped_column(String(500), nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="active")
    plate_text: Mapped[str | None] = mapped_column(String(20), nullable=True)
    ocr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    plate_detection_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)


class EventStore:
    def __init__(self, database_url: str, root: Path) -> None:
        if database_url.startswith("sqlite:///./"):
            db_path = root / database_url.removeprefix("sqlite:///./")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            database_url = "sqlite:///" + str(db_path)
        self.engine = create_engine(database_url, connect_args={"check_same_thread": False})
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)
        self._ensure_event_columns()

    def _ensure_event_columns(self) -> None:
        columns = {column["name"] for column in inspect(self.engine).get_columns("events")}
        additions = {
            "object_class": "VARCHAR(100)",
            "zone_id": "VARCHAR(100)",
            "bbox_json": "VARCHAR(500)",
            "status": "VARCHAR(30) DEFAULT 'active'",
            "plate_text": "VARCHAR(20)",
            "ocr_confidence": "FLOAT",
            "plate_detection_confidence": "FLOAT",
        }
        with self.engine.begin() as connection:
            for name, definition in additions.items():
                if name not in columns:
                    connection.execute(text(f"ALTER TABLE events ADD COLUMN {name} {definition}"))

    def add(self, data: dict[str, Any]) -> dict[str, Any]:
        event = Event(timestamp=datetime.now(timezone.utc), **data)
        with self.sessions() as session:
            session.add(event)
            session.commit()
            session.refresh(event)
            return self.serialize(event)

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.sessions() as session:
            rows = session.scalars(select(Event).order_by(Event.timestamp.desc()).limit(limit)).all()
            return [self.serialize(row) for row in rows]

    def close(self) -> None:
        self.engine.dispose()

    @staticmethod
    def serialize(event: Event) -> dict[str, Any]:
        return {
            "id": event.id, "camera_id": event.camera_id, "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type, "severity": event.severity, "track_id": event.track_id,
            "confidence": event.confidence, "snapshot": event.snapshot, "metadata": event.metadata_json,
            "object_class": event.object_class, "zone_id": event.zone_id,
            "bbox": event.bbox_json, "status": event.status,
            "plate_text": event.plate_text, "ocr_confidence": event.ocr_confidence,
            "plate_detection_confidence": event.plate_detection_confidence,
        }
