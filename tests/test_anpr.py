from app.anpr import TemporalPlateVoter, normalize_plate_text, plate_inside_vehicle
from app.db import EventStore


def test_plate_normalization_and_invalid_ocr():
    assert normalize_plate_text(" hr-26 ab 1234 ") == "HR26AB1234"
    assert normalize_plate_text("ABC") is None
    assert normalize_plate_text("") is None


def test_temporal_stability_and_track_isolation():
    voter = TemporalPlateVoter(minimum_votes=3)
    assert voter.add(12, "HR26AB1234") is None
    assert voter.add(12, "HR26AB1234") is None
    assert voter.add(12, "HR26AB1234") == "HR26AB1234"
    assert voter.add(13, "HR26AB1234") is None


def test_plate_vehicle_association():
    assert plate_inside_vehicle([20, 40, 80, 60], [0, 0, 100, 100])
    assert not plate_inside_vehicle([120, 40, 180, 60], [0, 0, 100, 100])
    assert not plate_inside_vehicle([1, 2], [0, 0, 100, 100])


def test_anpr_fields_persist_in_existing_event_store(tmp_path):
    store = EventStore("sqlite:///./anpr-test.sqlite", tmp_path)
    event = store.add({
        "camera_id": "cam-1",
        "event_type": "plate_observation",
        "severity": "info",
        "track_id": 7,
        "object_class": "CAR",
        "plate_text": "HR26AB1234",
        "ocr_confidence": 0.91,
        "plate_detection_confidence": 0.88,
        "status": "STABLE",
    })
    assert event["plate_text"] == "HR26AB1234"
    assert event["track_id"] == 7
    assert store.recent(1)[0]["status"] == "STABLE"
    store.close()
