from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .config import load_settings
from .db import EventStore
from .worker import CameraWorker

settings = load_settings()
store = EventStore(settings.database_url, settings.root)
workers: dict[str, CameraWorker] = {}
subscribers: set[asyncio.Queue] = set()
event_loop: asyncio.AbstractEventLoop | None = None


def publish_event(data: dict) -> None:
    event = store.add(data)
    if event_loop is not None:
        for queue in tuple(subscribers):
            event_loop.call_soon_threadsafe(queue.put_nowait, event)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global event_loop
    event_loop = asyncio.get_running_loop()
    for config in settings.cameras:
        if config.enabled:
            worker = CameraWorker(config, settings.yolo_model, settings.snapshot_dir, publish_event)
            workers[config.id] = worker
            worker.start()
    yield
    for worker in workers.values():
        worker.stop()
    workers.clear()
    event_loop = None


app = FastAPI(title="SIH26187 Border Surveillance", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return (settings.root / "frontend" / "index.html").read_text(encoding="utf-8")


@app.get("/health")
async def health():
    return {"status": "ok", "port": settings.port, "cameras": len(workers)}


@app.get("/cameras")
async def cameras():
    return [worker.health() for worker in workers.values()]


@app.get("/cameras/{camera_id}/health")
async def camera_health(camera_id: str):
    worker = workers.get(camera_id)
    if worker is None:
        return JSONResponse({"error": "camera not found"}, status_code=404)
    return worker.health()


@app.get("/events")
async def events(limit: int = 100):
    return store.recent(max(1, min(limit, 500)))


@app.get("/stream/{camera_id}")
async def stream(camera_id: str):
    worker = workers.get(camera_id)
    if worker is None:
        return JSONResponse({"error": "camera not found"}, status_code=404)

    async def generate():
        while True:
            frame = worker.frame()
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                       str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
            await asyncio.sleep(0.05)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/alerts/live")
async def alerts_live(websocket: WebSocket):
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue()
    subscribers.add(queue)
    try:
        while True:
            await websocket.send_json(await queue.get())
    except WebSocketDisconnect:
        pass
    finally:
        subscribers.discard(queue)
