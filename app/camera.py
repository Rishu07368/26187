from __future__ import annotations

import threading
import time
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone

import cv2
import numpy as np


class CameraSource(ABC):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self.connected = False
        self.connection_state = "OFFLINE"
        self.last_frame_timestamp: str | None = None
        self.frames_captured = 0
        self.capture_fps = 0.0
        self.resolution: list[int] | None = None
        self.latest_error: str | None = None

    @abstractmethod
    def open(self) -> bool: ...

    @abstractmethod
    def read(self) -> np.ndarray | None: ...

    @abstractmethod
    def close(self) -> None: ...

    def _record(self, frame: np.ndarray) -> None:
        with self._lock:
            self._latest = frame
            self.frames_captured += 1
            self.last_frame_timestamp = datetime.now(timezone.utc).isoformat()
            self.resolution = [int(frame.shape[1]), int(frame.shape[0])]


class RTSPCameraSource(CameraSource):
    def __init__(self, url: str, open_timeout_ms: int = 5000, read_timeout_ms: int = 5000) -> None:
        super().__init__()
        self.url = url
        self.open_timeout_ms = open_timeout_ms
        self.read_timeout_ms = read_timeout_ms
        self.capture: cv2.VideoCapture | None = None
        self._last_capture_time = 0.0
        self._fps_samples: list[float] = []
        self.reconnect_attempts = 0

    def _connect(self) -> bool:
        self.connection_state = "RECONNECTING"
        self.reconnect_attempts += 1
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            "rtsp_transport;tcp|stimeout;5000000",
        )
        cap = cv2.VideoCapture()
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.open_timeout_ms)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.read_timeout_ms)
        if not cap.open(self.url, cv2.CAP_FFMPEG):
            cap.release()
            self.latest_error = "Unable to open RTSP stream"
            self.connected = False
            return False
        self.capture = cap
        self.connected = True
        self.connection_state = "CONNECTED"
        self.latest_error = None
        return True

    def open(self) -> bool:
        return self._connect()

    def read(self) -> np.ndarray | None:
        if self.capture is None and not self._connect():
            return None
        assert self.capture is not None
        ok, frame = self.capture.read()
        now = time.monotonic()
        if not ok or frame is None:
            self.latest_error = "RTSP frame read failed"
            self.connected = False
            self.connection_state = "RECONNECTING"
            self.capture.release()
            self.capture = None
            return None
        if self._last_capture_time:
            delta = now - self._last_capture_time
            if delta > 0:
                self._fps_samples.append(1 / delta)
                self._fps_samples = self._fps_samples[-30:]
                self.capture_fps = sum(self._fps_samples) / len(self._fps_samples)
        self._last_capture_time = now
        self._record(frame)
        return frame

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.connected = False
        self.connection_state = "OFFLINE"


class WebcamSource(CameraSource):
    def __init__(self, index: int = 0, target_fps: float = 30.0) -> None:
        super().__init__()
        self.index = index
        self.target_fps = target_fps
        self.capture: cv2.VideoCapture | None = None
        self._next_read = 0.0
        self._last_capture_time = 0.0
        self._fps_samples: list[float] = []

    def _connect(self) -> bool:
        self.connection_state = "RECONNECTING"
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            cap.release()
            self.connected = False
            self.connection_state = "OFFLINE"
            self.latest_error = f"Unable to open webcam index {self.index}"
            return False
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture = cap
        self.connected = True
        self.connection_state = "CONNECTED"
        self.latest_error = None
        return True

    def open(self) -> bool:
        return self._connect()

    def read(self) -> np.ndarray | None:
        interval = 1.0 / self.target_fps
        now = time.monotonic()
        if self._next_read > now:
            time.sleep(self._next_read - now)
        self._next_read = time.monotonic() + interval
        if self.capture is None and not self._connect():
            return None
        assert self.capture is not None
        ok, frame = self.capture.read()
        now = time.monotonic()
        if not ok or frame is None:
            self.connected = False
            self.connection_state = "RECONNECTING"
            self.latest_error = "Webcam frame read failed"
            self.capture.release()
            self.capture = None
            return None
        if self._last_capture_time:
            delta = now - self._last_capture_time
            if delta > 0:
                self._fps_samples.append(1 / delta)
                self._fps_samples = self._fps_samples[-30:]
                self.capture_fps = round(sum(self._fps_samples) / len(self._fps_samples), 2)
        self._last_capture_time = now
        self.connected = True
        self.connection_state = "CONNECTED"
        self.latest_error = None
        self._record(frame)
        return frame

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.connected = False
        self.connection_state = "OFFLINE"


class LocalVideoSource(CameraSource):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        self.capture: cv2.VideoCapture | None = None

    def open(self) -> bool:
        if self.capture is None:
            self.capture = cv2.VideoCapture(self.path)
        if not self.capture.isOpened():
            self.latest_error = f"File unavailable: {self.path}"
            self.connection_state = "OFFLINE"
            return False
        self.connected = True
        self.connection_state = "CONNECTED"
        self.latest_error = None
        return True

    def read(self) -> np.ndarray | None:
        started = time.monotonic()
        if not self.open():
            return None
        ok, frame = self.capture.read()
        if not ok:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()
            if not ok or frame is None:
                self.connected = False
                self.connection_state = "OFFLINE"
                self.latest_error = "File demo has no decodable frames"
                return None
        self.connected = True
        self.connection_state = "CONNECTED"
        self.latest_error = None
        elapsed = time.monotonic() - started
        if elapsed > 0:
            self.capture_fps = round(1 / elapsed, 2)
        self._record(frame)
        return frame

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.connected = False
        self.connection_state = "OFFLINE"
