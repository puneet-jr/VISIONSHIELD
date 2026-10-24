"""FastAPI Routes and Streaming Controllers
Serves Dashboard UI, Zero-App Mobile Camera Streamer, MJPEG Video Feed,
REST APIs for zones/events, and Real-Time WebSocket channels.
"""
from fastapi import APIRouter, Request, Response, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pathlib import Path
from io import BytesIO
import secrets
import qrcode
from pydantic import BaseModel, Field, field_validator
from fastapi import HTTPException, Query
from app.config import config
import cv2
import numpy as np
import asyncio
import time
import socket
import json
import logging
from typing import List, Dict, Any

from app.ingestion.video_stream import video_manager
from app.engine.pipeline import pipeline
from app.engine.incident_agent import incident_dispatcher
from app.core.event_hub import event_hub
from app.db.database import db

logger = logging.getLogger("VisionShield.Routes")
router = APIRouter()

def get_local_ip() -> str:
    """Detect laptop local Wi-Fi IP address for the phone QR code."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

# --- Page Routes ---

@router.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Serve Tactical Security Command Center Dashboard."""
    with open(Path(__file__).parent / "templates/dashboard.html", "r", encoding="utf-8") as f:
        html = f.read()
    local_ip = get_local_ip()
    html = html.replace("{{LOCAL_IP}}", local_ip)
    return HTMLResponse(content=html)

@router.get("/phone", response_class=HTMLResponse)
async def serve_phone_camera(request: Request):
    """Serve Zero-App Mobile Browser Camera Streamer."""
    with open(Path(__file__).parent / "templates/mobile_cam.html", "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html)

# --- Video Ingestion & MJPEG Streaming ---

async def generate_mjpeg():
    """Generator yielding multipart JPEG frames for the live dashboard player."""
    last_id = -1
    while True:
        with video_manager.lock:
            jpeg, frame_id = video_manager.jpeg, video_manager.jpeg_id
        if jpeg and frame_id != last_id:
            last_id = frame_id
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        await asyncio.sleep(0.02)

@router.get("/video_feed")
async def video_feed():
    """MJPEG Live Annotated Surveillance Stream."""
    return StreamingResponse(
        generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.post("/api/upload_frame")
async def upload_frame(file: UploadFile = File(...)):
    """Receives camera frames uploaded directly by mobile browser."""
    contents = await file.read(512_001)
    if len(contents) > 512_000:
        raise HTTPException(413, "Frame too large")
    np_arr = np.frombuffer(contents, np.uint8)
    frame = await asyncio.to_thread(cv2.imdecode, np_arr, cv2.IMREAD_COLOR)
    if frame is not None:
        await asyncio.to_thread(video_manager.push_phone_frame, frame)
        return {"status": "ok"}
    raise HTTPException(422, "Invalid image")

# --- REST APIs ---

@router.get("/api/info")
async def get_system_info():
    return {
        "local_ip": get_local_ip(),
        "fps": round(video_manager.fps, 1) if time.time() - video_manager.last_frame_time < 5 else 0,
        "inference_ms": round(pipeline.inference_ms, 1),
        "processing_fps": round(pipeline.processing_fps, 1),
        "preview_fps": round(video_manager.preview_fps, 1),
        "detection_counts": pipeline.last_detection_counts,
        "source_path": video_manager.source_path,
        "camera_online": video_manager.camera_online and time.time() - video_manager.last_frame_time < 5,
        "detector_ready": pipeline.detector.model is not None,
        "detector_error": pipeline.detector.error,
        "backend": getattr(pipeline.detector, "backend", "torch"),
        "enabled_classes": pipeline.detector.enabled_labels,
        "input_size": config.INPUT_SIZE,
        "model": Path(config.YOLO_MODEL).name,
        "zone_dwell_seconds": config.ZONE_DWELL_SECONDS,
        "knife_confidence": config.KNIFE_CONFIDENCE,
        "report_mode": "Gemini" if incident_dispatcher._gemini_client else "Offline template",
        "remote_ready": config.PUBLIC_BASE_URL.startswith("https://"),
        "source_type": video_manager.source_type,
        "active_tracks": sum(t.disappeared_count == 0 for t in list(pipeline.tracker.tracks.values())) if video_manager.camera_online and time.time() - video_manager.last_frame_time < 5 else 0,
        "zones": current_zones()
    }

@router.post("/api/source")
async def change_source(payload: Dict[str, Any]):
    source = str(payload.get("source", "0"))
    if source not in {"0", "1", "phone_upload"}:
        raise HTTPException(422, "Choose webcam 0, webcam 1, or phone_upload")
    await asyncio.to_thread(video_manager.start, source)
    return {"status": "ok", "source": source}

@router.get("/api/zones")
async def get_zones():
    return {"zones": current_zones()}

@router.post("/api/zones")
async def update_zones(payload: Dict[str, Any]):
    zones = payload.get("zones", [])
    if len(zones) > 20:
        raise HTTPException(422, "At most 20 zones")
    validated = []
    for zone in zones:
        try:
            validated.append(ZonePayload.model_validate(zone).model_dump())
        except Exception as exc:
            raise HTTPException(422, "Invalid polygon: use 3-20 normalized points") from exc
    def apply():
        with pipeline.lock:
            if pipeline.geofence.source != video_manager.source_path:
                pipeline.geofence.select_source(video_manager.source_path)
            pipeline.geofence.replace_zones(validated)
    await asyncio.to_thread(apply)
    return {"status": "ok", "zones_count": len(validated)}

class ZonePayload(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=100)
    points: list[tuple[float, float]] = Field(min_length=3, max_length=20)
    severity: str = Field(default="HIGH", pattern="^(HIGH|MEDIUM|LOW)$")
    color: str = Field(default="#ef4444", pattern="^#[0-9a-fA-F]{6}$")

    @field_validator("points")
    @classmethod
    def normalized_points(cls, points):
        if any(not 0 <= v <= 1 for point in points for v in point):
            raise ValueError("Coordinates must be normalized")
        if len(set(points)) != len(points):
            raise ValueError("Duplicate vertices")
        area = abs(sum(points[i][0]*points[(i+1)%len(points)][1] - points[(i+1)%len(points)][0]*points[i][1] for i in range(len(points)))) / 2
        if area < 0.001:
            raise ValueError("Polygon is too small or degenerate")
        def orientation(a, b, c):
            return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if j == i+1 or (i == 0 and j == n-1):
                    continue
                a, b, c, d = points[i], points[(i+1)%n], points[j], points[(j+1)%n]
                if orientation(a,b,c)*orientation(a,b,d) <= 0 and orientation(c,d,a)*orientation(c,d,b) <= 0:
                    raise ValueError("Polygon edges cross")
        return points

@router.get("/api/events")
def get_events(limit: int = Query(50, ge=1, le=200)):
    return {"events": db.get_recent_events(limit)}

@router.get("/api/incidents")
def get_incidents(limit: int = Query(10, ge=1, le=50)):
    return {"incidents": db.get_recent_incidents(limit)}

# --- WebSocket Channel ---

@router.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """Sub-millisecond WebSocket channel for telemetry, alerts, and AI dossiers."""
    if not authorized(websocket):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    queue = event_hub.subscribe()
    
    # Send initial snapshot of recent events
    recent = event_hub.get_recent_events()
    for ev in recent:
        await websocket.send_text(json.dumps(ev))

    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=20)
                await websocket.send_json(event)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "HEARTBEAT"})
    except WebSocketDisconnect:
        event_hub.unsubscribe(queue)
    except Exception as e:
        logger.warning(f"WebSocket client error: {e}")
        event_hub.unsubscribe(queue)


def authorized(connection):
    # Local dashboard access only; a public Host always requires a session token.
    import ipaddress
    host = connection.url.hostname or ""
    local = host in {"localhost", "testserver"}
    try:
        address = ipaddress.ip_address(host)
        local = address.is_loopback or address.is_private
    except ValueError:
        pass
    return local or secrets.compare_digest(connection.cookies.get("vs_access", ""), config.ACCESS_TOKEN)


def phone_url(request):
    base = config.PUBLIC_BASE_URL or str(request.base_url).rstrip("/")
    return base + "/phone#token=" + config.ACCESS_TOKEN


@router.post("/api/pair")
async def pair(request: Request):
    payload = await request.json()
    if not secrets.compare_digest(str(payload.get("token", "")), config.ACCESS_TOKEN):
        raise HTTPException(403, "Invalid pairing link. Scan the QR code again.")
    response = JSONResponse({"status": "paired"})
    response.set_cookie("vs_access", config.ACCESS_TOKEN, httponly=True, secure=bool(config.PUBLIC_BASE_URL), samesite="strict")
    return response


@router.get("/api/connection")
def connection_info(request: Request):
    return {"phone_url": phone_url(request), "remote_ready": bool(config.PUBLIC_BASE_URL)}


@router.get("/api/qr")
def pairing_qr(request: Request):
    output = BytesIO()
    qrcode.make(phone_url(request)).save(output, format="PNG")
    return Response(output.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})


@router.websocket("/ws/camera")
async def camera_socket(websocket: WebSocket):
    if not authorized(websocket):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    # A single sender owns capture at a time. No unbounded queue of stale frames.
    if camera_lock.locked():
        await websocket.close(code=1013, reason="Another phone is streaming")
        return
    async with camera_lock:
        await asyncio.to_thread(video_manager.start, "phone_upload")
        stream_generation = video_manager.generation
        try:
            while True:
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=30)
                if video_manager.generation != stream_generation:
                    await websocket.close(code=4000, reason="Camera source changed")
                    return
                if len(data) > 512_000:
                    await websocket.close(code=1009)
                    return
                frame = await asyncio.to_thread(cv2.imdecode, np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if frame is None or max(frame.shape[:2]) > 1920:
                    await websocket.send_json({"error": "Invalid or oversized image"})
                    continue
                accepted = await asyncio.to_thread(video_manager.push_phone_frame, frame, stream_generation)
                if not accepted:
                    await websocket.close(code=4000, reason="Camera source changed")
                    return
                await websocket.send_json({"ok": True})
        except (WebSocketDisconnect, asyncio.TimeoutError):
            pass
        finally:
            if video_manager.generation == stream_generation:
                video_manager.camera_online = False

camera_lock = asyncio.Lock()


@router.get("/api/telemetry")
async def live_telemetry():
    snapshot = pipeline.snapshot
    if (snapshot.get("generation") != video_manager.generation or
        not video_manager.camera_online or time.time() - snapshot.get("captured_at", 0) > 1):
        return {"tracks": [], "zones": current_zones(), "confirmed_zones": [], "stale": True}
    return {**snapshot, "stale": False, "age_ms": round((time.time()-snapshot["captured_at"])*1000)}


def current_zones():
    if pipeline.geofence.source == video_manager.source_path:
        return pipeline.geofence.get_zones_dict()
    return pipeline.geofence.saved.get(video_manager.source_path, [])
