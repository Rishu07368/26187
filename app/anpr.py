from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from shapely.geometry import box


PLATE_PATTERN = re.compile(r"^[A-Z0-9]{4,12}$")


def normalize_plate_text(raw_text: str | None) -> str | None:
    if not raw_text:
        return None
    normalized = re.sub(r"[^A-Za-z0-9]", "", str(raw_text)).upper()
    return normalized if PLATE_PATTERN.fullmatch(normalized) else None


def plate_inside_vehicle(plate_bbox: list[float], vehicle_bbox: list[float]) -> bool:
    if len(plate_bbox) != 4 or len(vehicle_bbox) != 4:
        return False
    try:
        plate = box(*[float(value) for value in plate_bbox])
        vehicle = box(*[float(value) for value in vehicle_bbox])
        return vehicle.covers(plate)
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class PlateObservation:
    track_id: int
    vehicle_class: str
    plate_bbox: list[float]
    plate_detection_confidence: float
    raw_text: str | None
    ocr_confidence: float | None


class TemporalPlateVoter:
    def __init__(self, minimum_votes: int = 3) -> None:
        self.minimum_votes = minimum_votes
        self._observations: dict[int, Counter[str]] = {}

    def add(self, track_id: int, text: str | None) -> str | None:
        normalized = normalize_plate_text(text)
        if normalized is None:
            return None
        votes = self._observations.setdefault(track_id, Counter())
        votes[normalized] += 1
        candidate, count = votes.most_common(1)[0]
        return candidate if count >= self.minimum_votes else None

    def forget_missing(self, active_track_ids: set[int]) -> None:
        self._observations = {
            track_id: votes for track_id, votes in self._observations.items()
            if track_id in active_track_ids
        }


class ANPRProcessor:
    """Explicitly unavailable until a verified plate detector and OCR are configured."""

    detector_name = None
    ocr_name = None

    @property
    def available(self) -> bool:
        return False

    def process(self, frame, vehicles: list[dict]) -> list[PlateObservation]:
        return []
