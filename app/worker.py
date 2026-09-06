from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from .camera import LocalVideoSource, RTSPCameraSource, WebcamSource
from .anpr import ANPRProcessor, TemporalPlateVoter, normalize_plate_text
from .config import CameraConfig
from .face import FaceDetection, associate_faces, face_counts, parse_face_rectangles

log = logging.getLogger(__name__)

VEHICLE_CLASS_MAP = {
    "car": "CAR",
    "motorcycle": "MOTORCYCLE",
    "bus": "BUS",
    "truck": "TRUCK",
    "bicycle": "BICYCLE",
}


def classify_object(class_name: str) -> tuple[str, bool]:
    normalized = str(class_name).strip().lower()
    if normalized == "person":
        return "PERSON", False
    vehicle = VEHICLE_CLASS_MAP.get(normalized)
    return (vehicle, True) if vehicle else (str(class_name).upper(), False)


def split_detections(detections: list[dict]) -> tuple[list[dict], list[dict]]:
    persons, vehicles = [], []
    for detection in detections:
        if not isinstance(detection, dict) or not detection.get("class"):
            continue
        _, is_vehicle = classify_object(detection["class"])
        (vehicles if is_vehicle else persons).append(detection)
    return persons, vehicles


def vehicle_counts(detections: list[dict]) -> dict[str, int]:
    counts = {label: 0 for label in ("CAR", "MOTORCYCLE", "BUS", "TRUCK", "BICYCLE")}
    for detection in detections:
        label, is_vehicle = classify_object(detection.get("class", ""))
        if is_vehicle and label in counts:
            counts[label] += 1
    return counts


def normalized_polygon(points: list[list[float]], width: int, height: int) -> Polygon | None:
    if len(points) < 3 or width <= 0 or height <= 0:
        return None
    try:
        scaled = [(float(x) * width, float(y) * height) for x, y in points]
        polygon = Polygon(scaled)
        return polygon if polygon.is_valid and not polygon.is_empty else None
    except (TypeError, ValueError):
        return None


class IntrusionState:
    def __init__(self) -> None:
        self._inside: dict[tuple[int, str], bool] = {}

    def update(self, track_id: int, zone_id: str, inside: bool) -> bool:
        key = (track_id, zone_id)
        previous = self._inside.get(key, False)
        self._inside[key] = inside
        return inside and not previous

    def forget_missing(self, active_track_ids: set[int]) -> None:
        self._inside = {key: state for key, state in self._inside.items()
                        if key[0] in active_track_ids}


class CameraWorker:
    def __init__(self, config: CameraConfig, model_name: str, snapshot_dir: Path,
                 event_callback: Callable[[dict], None]) -> None:
        self.config = config
        self.snapshot_dir = snapshot_dir
        self.event_callback = event_callback
        source_type = config.source_type.lower()
        if source_type == "rtsp":
            self.source = RTSPCameraSource(config.rtsp_url or "")
        elif source_type == "webcam":
            index = int(config.source if config.source is not None else 0)
            self.source = WebcamSource(index)
        else:
            self.source = LocalVideoSource(config.file_path or "")
        self.model_name = model_name
        self.stop_event = threading.Event()
        self.capture_thread = threading.Thread(target=self._capture_loop, name=f"capture-{config.id}", daemon=True)
        self.inference_thread = threading.Thread(target=self._inference_loop, name=f"inference-{config.id}", daemon=True)
        self.face_thread = threading.Thread(target=self._face_loop, name=f"face-{config.id}", daemon=True)
        self._frame_lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._annotated: bytes | None = None
        self._model = None
        self._model_error: str | None = None
        self._face_detector = None
        self._face_error: str | None = None
        self.face_detections: list[FaceDetection] = []
        self.face_detection_fps = 0.0
        self._last_face_detection_time = 0.0
        self._last_face_capture_count = 0
        self.frames_processed = 0
        self.processing_fps = 0.0
        self.active_tracks = 0
        self.active_vehicle_tracks = 0
        self.vehicle_counts: dict[str, int] = {
            label: 0 for label in ("CAR", "MOTORCYCLE", "BUS", "TRUCK", "BICYCLE")
        }
        self.detection_count = 0
        self.latest_detections: list[dict] = []
        self._last_process_time = 0.0
        self._last_processed_capture_count = 0
        self._track_history: dict[int, list[tuple[float, float]]] = defaultdict(list)
        self._last_events: dict[tuple[str, int], float] = {}
        self._intrusion_state = IntrusionState()
        self._zone_states: dict[str, str] = {}
        self._zone_occupied: dict[str, bool] = {}
        self._intrusion_overlay: str | None = None
        self._anpr = ANPRProcessor()
        self._plate_voter = TemporalPlateVoter()
        self.anpr_observations: list[dict] = []
        self._anpr_frame_interval = 5

    def start(self) -> None:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.capture_thread.start()
        self.inference_thread.start()
        self.face_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.source.close()
        self.capture_thread.join(timeout=3)
        self.inference_thread.join(timeout=3)
        self.face_thread.join(timeout=3)

    def _capture_loop(self) -> None:
        backoff = 1.0
        while not self.stop_event.is_set():
            frame = self.source.read()
            if frame is not None:
                with self._frame_lock:
                    self._frame = frame
                backoff = 1.0
                continue
            self.source.connection_state = "RECONNECTING" if self.config.source_type.lower() == "rtsp" else "OFFLINE"
            self.stop_event.wait(backoff)
            backoff = min(backoff * 2, 30.0)

    def _load_model(self):
        if self._model is not None or self._model_error:
            return self._model
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_name)
        except Exception as exc:
            self._model_error = str(exc)
            log.exception("YOLO load failed for %s", self.config.id)
        return self._model

    def _inference_loop(self) -> None:
        while not self.stop_event.is_set():
            with self._frame_lock:
                frame = None if self._frame is None else self._frame.copy()
                capture_count = self.source.frames_captured
            if frame is None or capture_count <= self._last_processed_capture_count:
                self.stop_event.wait(0.05)
                continue
            self._last_processed_capture_count = capture_count
            self._process(frame)

    def _load_face_detector(self):
        if self._face_detector is not None or self._face_error:
            return self._face_detector
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(cascade_path)
            if detector.empty():
                raise RuntimeError(f"Unable to load face cascade: {cascade_path}")
            self._face_detector = detector
        except Exception as exc:
            self._face_error = str(exc)
            log.exception("Face detector load failed for %s", self.config.id)
        return self._face_detector

    def _face_loop(self) -> None:
        while not self.stop_event.is_set():
            with self._frame_lock:
                frame = None if self._frame is None else self._frame.copy()
                capture_count = self.source.frames_captured
            if frame is None or capture_count <= self._last_face_capture_count:
                self.stop_event.wait(0.01)
                continue
            self._last_face_capture_count = capture_count
            detector = self._load_face_detector()
            if detector is None:
                self.face_detections = []
                self.stop_event.wait(0.1)
                continue
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rectangles = detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                persons, _ = split_detections(self.latest_detections)
                detections = associate_faces(parse_face_rectangles(rectangles), persons)
                self.face_detections = detections
                now = time.monotonic()
                if self._last_face_detection_time:
                    elapsed = now - self._last_face_detection_time
                    if elapsed > 0:
                        self.face_detection_fps = round(1 / elapsed, 2)
                self._last_face_detection_time = now
            except Exception:
                self._face_error = "Face detection failed"
                log.exception("Face detection failed for %s", self.config.id)

    def _process(self, frame: np.ndarray) -> None:
        annotated = frame.copy()
        self._intrusion_overlay = None
        zones = self._zones_for_frame(frame.shape[1], frame.shape[0])
        for zone in zones:
            cv2.polylines(annotated, [zone["pixel_points"]], True, (0, 165, 255), 3)
            cv2.putText(annotated, zone["id"], tuple(zone["pixel_points"][0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        model = self._load_model()
        boxes = []
        if model is not None:
            try:
                result = model.track(frame, persist=True, tracker="bytetrack.yaml",
                                     conf=self.config.confidence, verbose=False)[0]
                if result.boxes is not None:
                    for box in result.boxes:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        cls = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())
                        track_id = int(box.id[0].item()) if box.id is not None else None
                        boxes.append((cls, confidence, *(int(value) for value in xyxy), track_id))
                        name = result.names.get(cls, str(cls))
                        display_name, is_vehicle = classify_object(name)
                        label = f"{display_name} {confidence:.2f}" + (f" ID:{track_id}" if track_id is not None else "")
                        color = (255, 180, 0) if is_vehicle else (0, 220, 80)
                        cv2.rectangle(annotated, tuple(xyxy[:2]), tuple(xyxy[2:]), color, 2)
                        cv2.putText(annotated, label, (xyxy[0], max(20, xyxy[1] - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except Exception:
                log.exception("Inference failed for %s", self.config.id)
        else:
            cv2.putText(annotated, "YOLO unavailable", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        current_detections = [
            {"class": result_name, "confidence": confidence, "track_id": track_id,
             "bbox": [x1, y1, x2, y2]}
            for (result_name, confidence, track_id), (_, _, x1, y1, x2, y2, _) in
            zip(self._detection_labels(boxes), boxes)
        ]
        for face in self.face_detections:
            x, y, width, height = face.bbox
            label = "FACE"
            if face.track_id is not None:
                label += f" #{face.track_id}"
            cv2.rectangle(annotated, (x, y), (x + width, y + height), (255, 180, 0), 2)
            cv2.putText(annotated, label, (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 2)
        self._analytics(boxes, annotated, zones)
        for state_key, state in self._zone_states.items():
            label_position = (20, 35 + 25 * list(self._zone_states).index(state_key))
            cv2.putText(annotated, f"{state_key} {state}", label_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 0, 255) if state == "INSIDE" else (0, 165, 255), 2)
        if self._intrusion_overlay:
            cv2.rectangle(annotated, (10, 70), (330, 125), (0, 0, 255), -1)
            cv2.putText(annotated, self._intrusion_overlay, (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            with self._frame_lock:
                self._annotated = encoded.tobytes()
        now = time.monotonic()
        if self._last_process_time:
            interval = now - self._last_process_time
            if interval > 0:
                self.processing_fps = round(1 / interval, 2)
        self._last_process_time = now
        self.frames_processed += 1
        self.detection_count = len(boxes)
        self.active_tracks = sum(1 for box in boxes if box[-1] is not None)
        self.latest_detections = current_detections
        _, vehicle_detections = split_detections(self.latest_detections)
        self.vehicle_counts = vehicle_counts(vehicle_detections)
        self.active_vehicle_tracks = sum(
            1 for detection in vehicle_detections if detection["track_id"] is not None
        )
        self._plate_voter.forget_missing({
            detection["track_id"] for detection in vehicle_detections
            if detection["track_id"] is not None
        })
        if self.frames_processed % self._anpr_frame_interval == 0:
            self.anpr_observations = [
                {
                    "track_id": observation.track_id,
                    "vehicle_class": observation.vehicle_class,
                    "plate": normalize_plate_text(observation.raw_text),
                    "ocr_confidence": observation.ocr_confidence,
                    "status": "OBSERVED",
                }
                for observation in self._anpr.process(frame, vehicle_detections)
            ]

    def _detection_labels(self, boxes) -> list[tuple[str, float, int | None]]:
        model = self._model
        names = getattr(model, "names", {}) if model is not None else {}
        return [(
            classify_object(names.get(cls, str(cls)))[0],
            confidence,
            track_id,
        )
                for cls, confidence, _, _, _, _, track_id in boxes]

    def _zones_for_frame(self, width: int, height: int) -> list[dict]:
        configured = self.config.zones
        if not configured and self.config.zone:
            configured = [{"id": "restricted-zone", "type": "polygon", "points": self.config.zone}]
        zones = []
        for item in configured:
            if item.get("type", "polygon") != "polygon":
                continue
            points = item.get("points", [])
            polygon = normalized_polygon(points, width, height)
            if polygon is not None:
                zones.append({"id": str(item.get("id", "restricted-zone")), "polygon": polygon,
                              "pixel_points": np.array(polygon.exterior.coords[:-1], dtype=np.int32)})
        return zones

    def _analytics(self, boxes, frame: np.ndarray, zones: list[dict]) -> None:
        now = time.monotonic()
        active_track_ids = {box[-1] for box in boxes if box[-1] is not None}
        self._intrusion_state.forget_missing(active_track_ids)
        self._zone_states = {}
        candidates: dict[str, tuple[str, float, int, tuple[int, int, int, int]]] = {}
        for cls, confidence, x1, y1, x2, y2, track_id in boxes:
            if track_id is None:
                continue
            raw_class = self._detection_labels([(cls, confidence, x1, y1, x2, y2, track_id)])[0][0]
            object_class, _ = classify_object(raw_class)
            if object_class != "PERSON":
                continue
            foot_point = ((x1 + x2) / 2, y2)
            for zone in zones:
                inside = bool(zone["polygon"].covers(Point(foot_point)))
                state_key = f"{zone['id']}:{track_id}"
                self._zone_states[state_key] = "INSIDE" if inside else "OUTSIDE"
                self._intrusion_state.update(track_id, zone["id"], inside)
                if inside and zone["id"] not in candidates:
                    candidates[zone["id"]] = (object_class, confidence, track_id, (x1, y1, x2, y2))
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            history = self._track_history[track_id]
            history.append(center)
            del history[:-30]
            if len(history) >= 15:
                distance = ((history[-1][0] - history[-15][0]) ** 2 +
                            (history[-1][1] - history[-15][1]) ** 2) ** 0.5
                if distance > 120:
                    self._emit_once("suspicious_high_speed_movement", track_id, confidence, frame, now)
            if len(history) >= 150:
                distance = ((history[-1][0] - history[-150][0]) ** 2 +
                            (history[-1][1] - history[-150][1]) ** 2) ** 0.5
                if distance < 20:
                    self._emit_once("loitering", track_id, confidence, frame, now)
        for zone_id, candidate in candidates.items():
            if not self._zone_occupied.get(zone_id, False):
                object_class, confidence, track_id, bbox = candidate
                self._intrusion_overlay = f"INTRUSION Track #{track_id}"
                self._emit_intrusion(zone_id, object_class, confidence, track_id, bbox, frame)
            self._zone_occupied[zone_id] = True
        configured_zone_ids = {zone["id"] for zone in zones}
        for zone_id in configured_zone_ids - candidates.keys():
            self._zone_occupied[zone_id] = False

    def _emit_intrusion(self, zone_id: str, object_class: str, confidence: float,
                        track_id: int, bbox: tuple[int, int, int, int], frame: np.ndarray) -> None:
        filename = f"{self.config.id}_{int(time.time())}_intrusion.jpg"
        cv2.imwrite(str(self.snapshot_dir / filename), frame)
        self.event_callback({
            "camera_id": self.config.id, "event_type": "intrusion", "severity": "high",
            "track_id": track_id, "confidence": confidence, "snapshot": filename,
            "metadata_json": json.dumps({"source_type": self.config.source_type}),
            "object_class": object_class, "zone_id": zone_id,
            "bbox_json": json.dumps(list(bbox)), "status": "active",
        })

    def _emit_once(self, event_type: str, track_id: int, confidence: float, frame: np.ndarray, now: float) -> None:
        key = (event_type, track_id)
        if now - self._last_events.get(key, 0) < 10:
            return
        self._last_events[key] = now
        filename = f"{self.config.id}_{int(time.time())}_{event_type}.jpg"
        cv2.imwrite(str(self.snapshot_dir / filename), frame)
        self.event_callback({"camera_id": self.config.id, "event_type": event_type, "severity": "high",
                             "track_id": track_id, "confidence": confidence, "snapshot": filename,
                             "metadata_json": json.dumps({"source_type": self.config.source_type})})

    def frame(self) -> bytes | None:
        with self._frame_lock:
            return self._annotated

    def health(self) -> dict:
        return {"camera_id": self.config.id, "name": self.config.name, "source_type": self.config.source_type,
                "connected": self.source.connected, "connection_state": self.source.connection_state,
                "resolution": self.source.resolution, "capture_fps": round(self.source.capture_fps, 2),
                "processing_fps": self.processing_fps, "last_frame_timestamp": self.source.last_frame_timestamp,
                "frames_captured": self.source.frames_captured, "frames_processed": self.frames_processed,
                "active_tracks": self.active_tracks, "detection_count": self.detection_count,
                "detections": self.latest_detections,
                "vehicle_counts": self.vehicle_counts,
                "active_vehicle_tracks": self.active_vehicle_tracks,
                "faces_detected": len(self.face_detections),
                "faces_associated": face_counts(self.face_detections)["associated"],
                "faces_unassociated": face_counts(self.face_detections)["unassociated"],
                "face_detection_fps": self.face_detection_fps,
                "face_detector": "OpenCV Haar Cascade" if self._face_detector is not None else None,
                "anpr_available": self._anpr.available,
                "anpr_detector": self._anpr.detector_name,
                "anpr_ocr": self._anpr.ocr_name,
                "anpr_observations": self.anpr_observations,
                "zone_states": self._zone_states, "intrusion_overlay": self._intrusion_overlay,
                "source": getattr(self.config, "source", None),
                "reconnect_attempts": getattr(self.source, "reconnect_attempts", 0),
                "latest_error": self.source.latest_error, "model_error": self._model_error,
                "face_error": self._face_error}
