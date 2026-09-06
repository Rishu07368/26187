from app.face import FaceDetection, associate_faces, face_counts, parse_face_rectangles
import numpy as np


def test_face_rectangle_parsing_and_empty_input():
    assert parse_face_rectangles([(1, 2, 30, 40), ("bad", 2, 3, 4), (1, 2, -1, 4)]) == [
        FaceDetection((1, 2, 30, 40))
    ]
    assert parse_face_rectangles([]) == []
    assert parse_face_rectangles(np.array([[1, 2, 30, 40]])) == [FaceDetection((1, 2, 30, 40))]


def test_face_person_association_and_counts():
    faces = parse_face_rectangles([(20, 20, 30, 30), (200, 200, 20, 20)])
    associated = associate_faces(faces, [{"bbox": [0, 0, 100, 100], "track_id": 4}])
    assert associated[0].track_id == 4
    assert associated[1].track_id is None
    assert face_counts(associated) == {"faces": 2, "associated": 1, "unassociated": 1}
