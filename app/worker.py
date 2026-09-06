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
from .config import CameraConfig

log = logging.getLogger(__name__)


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
        self._frame_lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._annotated: bytes | None = None
        self._model = None
        self._model_error: str | None = None
        self._face_detector = None
        self._face_error: str | None = None
        self.frames_processed = 0
        self.processing_fps = 0.0
        self.active_tracks = 0
        self.detection_count = 0
        self.latest_detections: list[dict] = []
        self._last_process_time = 0.0
        self._last_processed_capture_count = 0
        self._track_history: dict[int, list[tuple[float, float]]] = defaultdict(list)
        self._last_events: dict[tuple[str, int], float] = {}

    def start(self) -> None:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.capture_thread.start()
        self.inference_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.source.close()
        self.capture_thread.join(timeout=3)
        self.inference_thread.join(timeout=3)

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
            import mediapipe as mp
            self._face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
        except Exception as exc:
            self._face_error = str(exc)
            log.exception("Face detector load failed for %s", self.config.id)
        return self._face_detector

    def _process(self, frame: np.ndarray) -> None:
        annotated = frame.copy()
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
                        boxes.append((cls, confidence, *xyxy, track_id))
                        name = result.names.get(cls, str(cls))
                        label = f"{name} {confidence:.2f}" + (f" ID:{track_id}" if track_id is not None else "")
                        cv2.rectangle(annotated, tuple(xyxy[:2]), tuple(xyxy[2:]), (0, 220, 80), 2)
                        cv2.putText(annotated, label, (xyxy[0], max(20, xyxy[1] - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 80), 2)
            except Exception:
                log.exception("Inference failed for %s", self.config.id)
        else:
            cv2.putText(annotated, "YOLO unavailable", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        face_detector = self._load_face_detector()
        if face_detector is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_result = face_detector.process(rgb)
                for detection in face_result.detections or []:
                    box = detection.location_data.relative_bounding_box
                    height, width = frame.shape[:2]
                    x1, y1 = max(0, int(box.xmin * width)), max(0, int(box.ymin * height))
                    x2, y2 = min(width, x1 + int(box.width * width)), min(height, y1 + int(box.height * height))
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 0), 2)
                    cv2.putText(annotated, "face", (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 2)
            except Exception:
                log.exception("Face detection failed for %s", self.config.id)
        self._analytics(boxes, annotated)
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
        self.latest_detections = [
            {"class": result_name, "confidence": confidence, "track_id": track_id}
            for result_name, confidence, track_id in self._detection_labels(boxes)
        ]

    def _detection_labels(self, boxes) -> list[tuple[str, float, int | None]]:
        model = self._model
        names = getattr(model, "names", {}) if model is not None else {}
        return [(names.get(cls, str(cls)), confidence, track_id)
                for cls, confidence, _, _, _, _, track_id in boxes]

    def _analytics(self, boxes, frame: np.ndarray) -> None:
        zone = Polygon(self.config.zone) if len(self.config.zone) >= 3 else None
        now = time.monotonic()
        for _, confidence, x1, y1, x2, y2, track_id in boxes:
            if track_id is None:
                continue
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            history = self._track_history[track_id]
            history.append(center)
            del history[:-30]
            if zone and zone.contains(Point(center)):
                self._emit_once("restricted_zone_intrusion", track_id, confidence, frame, now)
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
                "source": getattr(self.config, "source", None),
                "latest_error": self.source.latest_error, "model_error": self._model_error,
                "face_error": self._face_error}
