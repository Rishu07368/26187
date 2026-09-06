from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, select
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


class EventStore:
    def __init__(self, database_url: str, root: Path) -> None:
        if database_url.startswith("sqlite:///./"):
            db_path = root / database_url.removeprefix("sqlite:///./")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            database_url = "sqlite:///" + str(db_path)
        self.engine = create_engine(database_url, connect_args={"check_same_thread": False})
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)

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

    @staticmethod
    def serialize(event: Event) -> dict[str, Any]:
        return {"id": event.id, "camera_id": event.camera_id, "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type, "severity": event.severity, "track_id": event.track_id,
                "confidence": event.confidence, "snapshot": event.snapshot, "metadata": event.metadata_json}
