from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FaceDetection:
    bbox: tuple[int, int, int, int]
    confidence: float | None = None
    track_id: int | None = None


def parse_face_rectangles(rectangles) -> list[FaceDetection]:
    detections: list[FaceDetection] = []
    if rectangles is None:
        return detections
    for rectangle in rectangles:
        try:
            values = [int(value) for value in rectangle[:4]]
            if len(values) != 4 or values[2] <= 0 or values[3] <= 0:
                continue
            detections.append(FaceDetection((values[0], values[1], values[2], values[3])))
        except (TypeError, ValueError, IndexError):
            continue
    return detections


def associate_faces(
    faces: list[FaceDetection], persons: list[dict], minimum_overlap: float = 0.2
) -> list[FaceDetection]:
    associated: list[FaceDetection] = []
    for face in faces:
        fx, fy, fw, fh = face.bbox
        face_area = fw * fh
        best_track = None
        best_overlap = 0.0
        for person in persons:
            bbox = person.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            px1, py1, px2, py2 = bbox
            ix1, iy1 = max(fx, px1), max(fy, py1)
            ix2, iy2 = min(fx + fw, px2), min(fy + fh, py2)
            overlap = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            ratio = overlap / face_area if face_area else 0.0
            if ratio >= minimum_overlap and ratio > best_overlap:
                best_overlap = ratio
                best_track = person.get("track_id")
        associated.append(FaceDetection(face.bbox, face.confidence, best_track))
    return associated


def face_counts(faces: list[FaceDetection]) -> dict[str, int]:
    return {
        "faces": len(faces),
        "associated": sum(face.track_id is not None for face in faces),
        "unassociated": sum(face.track_id is None for face in faces),
    }
