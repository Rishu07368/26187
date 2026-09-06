from app.worker import IntrusionState, normalized_polygon
from shapely.geometry import Point


def test_point_inside_and_outside_normalized_polygon():
    polygon = normalized_polygon([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]], 100, 100)
    assert polygon is not None
    assert polygon.covers(Point(50, 50))
    assert not polygon.covers(Point(5, 5))


def test_entry_only_and_reentry():
    state = IntrusionState()
    assert state.update(7, "restricted-zone", False) is False
    assert state.update(7, "restricted-zone", True) is True
    assert state.update(7, "restricted-zone", True) is False
    assert state.update(7, "restricted-zone", False) is False
    assert state.update(7, "restricted-zone", True) is True


def test_tracks_are_independent_and_missing_tracks_are_safe():
    state = IntrusionState()
    assert state.update(1, "zone", True) is True
    assert state.update(2, "zone", True) is True
    state.forget_missing({2})
    assert state.update(1, "zone", True) is True
    assert state.update(2, "zone", True) is False


def test_malformed_polygon_fails_safely():
    assert normalized_polygon([], 100, 100) is None
    assert normalized_polygon([["bad"]], 100, 100) is None
