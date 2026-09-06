from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class CameraConfig:
    id: str
    name: str
    source_type: str
    source: int | str | None = None
    rtsp_url: str | None = None
    file_path: str | None = None
    enabled: bool = True
    confidence: float = 0.35
    zone: list[list[int]] = field(default_factory=list)
    zones: list[dict] = field(default_factory=list)


@dataclass
class Settings:
    root: Path
    host: str
    port: int
    cameras_config: Path
    yolo_model: str
    snapshot_dir: Path
    database_url: str
    cameras: list[CameraConfig]


def load_settings(root: Path | None = None) -> Settings:
    root = root or Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")
    config_path = Path(os.getenv("CAMERAS_CONFIG", "cameras.json"))
    if not config_path.is_absolute():
        config_path = root / config_path
    cameras = []
    if config_path.exists():
        raw_cameras = json.loads(config_path.read_text(encoding="utf-8"))
        cameras = []
        for item in raw_cameras:
            if item.get("file_path") and not Path(item["file_path"]).is_absolute():
                item["file_path"] = str(root / item["file_path"])
            cameras.append(CameraConfig(**item))
    if os.getenv("CAMERA_MODE", "local").lower() == "rtsp":
        rtsp_url = os.getenv("RTSP_URL", "").strip()
        cameras = [
            CameraConfig(
                id=camera.id,
                name=camera.name,
                source_type=camera.source_type,
                source=camera.source,
                rtsp_url=rtsp_url if camera.source_type.lower() == "rtsp" and rtsp_url else camera.rtsp_url,
                file_path=camera.file_path,
                enabled=bool(rtsp_url) and camera.source_type.lower() == "rtsp",
                confidence=camera.confidence,
                zone=camera.zone,
                zones=camera.zones,
            )
            for camera in cameras
        ]
    snapshot_dir = Path(os.getenv("SNAPSHOT_DIR", "data/snapshots"))
    if not snapshot_dir.is_absolute():
        snapshot_dir = root / snapshot_dir
    return Settings(root, os.getenv("HOST", "127.0.0.1"), int(os.getenv("PORT", "8765")),
                    config_path, os.getenv("YOLO_MODEL", "yolov8n.pt"), snapshot_dir,
                    os.getenv("DATABASE_URL", "sqlite:///./data/events.db"), cameras)
