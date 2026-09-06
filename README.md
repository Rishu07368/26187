# SIH26187 Border Surveillance

This application ingests configured IP CCTV RTSP streams in persistent camera
workers, runs YOLOv8 with ByteTrack, generates backend-owned alerts, and
distributes one annotated MJPEG stream to all viewers.

## Run

```powershell
cd C:\Users\srish\PycharmProjects\PythonProject\26187
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

Open http://127.0.0.1:8765. The included `cameras.json` enables the real
webcam source and keeps the placeholder RTSP camera disabled. To configure a
real CCTV camera, replace the placeholder RTSP URL using a secured local
configuration and set `enabled` to `true`. RTSP uses FFmpeg TCP transport,
short open/read timeouts, bounded reconnect backoff, and latest-frame
semantics. RTSP credentials are never returned by the API or sent to the
browser. Local files are supported only as `FILE / DEMO` fallback sources.

## Render deployment

`render.yaml` uses Python 3.12, preloads the existing `yolov8n.pt` model during
the build, and starts Uvicorn on `0.0.0.0:$PORT`. Render is configured for
`CAMERA_MODE=rtsp`, so it does not assume a laptop webcam exists in the cloud.
Set the secret Render environment variable `RTSP_URL` to a reachable camera
URL; credentials must not be committed. Local development continues to use
the webcam configuration and port 8765 by default.

## Endpoints

- `GET /health`
- `GET /cameras`
- `GET /cameras/{camera_id}/health`
- `GET /stream/{camera_id}`
- `GET /events`
- `WS /alerts/live`

## ANPR status

The ANPR data model, OCR normalization, vehicle/plate association helpers,
temporal voting, and dashboard empty state are present. The current isolated
environment has no verified license-plate detector or OCR engine installed, so
the runtime processor is explicitly disabled and does not emit plate text.
Physical ANPR validation must not be claimed until a legitimate detector and
OCR engine are configured and validated against a real vehicle.
