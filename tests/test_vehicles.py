from app.worker import classify_object, split_detections, vehicle_counts


def test_vehicle_class_mapping():
    assert classify_object("car") == ("CAR", True)
    assert classify_object("motorcycle") == ("MOTORCYCLE", True)
    assert classify_object("bus") == ("BUS", True)
    assert classify_object("truck") == ("TRUCK", True)
    assert classify_object("person") == ("PERSON", False)


def test_person_vehicle_separation_and_counts():
    detections = [
        {"class": "person", "track_id": 1},
        {"class": "car", "track_id": 2},
        {"class": "car", "track_id": 3},
        {"class": "truck", "track_id": None},
    ]
    persons, vehicles = split_detections(detections)
    assert len(persons) == 1
    assert len(vehicles) == 3
    assert vehicle_counts(vehicles) == {
        "CAR": 2, "MOTORCYCLE": 0, "BUS": 0, "TRUCK": 1, "BICYCLE": 0,
    }


def test_malformed_and_empty_detections_are_ignored():
    persons, vehicles = split_detections([{}, None, {"class": ""}])
    assert persons == []
    assert vehicles == []
    assert vehicle_counts([])["CAR"] == 0
