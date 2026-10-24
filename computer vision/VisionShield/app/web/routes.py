"""FastAPI Routes and Streaming Controllers
Serves Dashboard UI, Zero-App Mobile Camera Streamer, MJPEG Video Feed,
REST APIs for zones/events, and Real-Time WebSocket channels.
"""

from fastapi import (
    APIRouter,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
    UploadFile,
    File,
)
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pathlib import Path
from io import BytesIO
import secrets
import qrcode
from pydantic import BaseModel, Field, field_validator, model_validator
from fastapi import HTTPException, Query
from app.config import config
import cv2
import numpy as np
import asyncio
import time
import socket
import json
import logging
from typing import Dict, Any, Literal
from datetime import datetime
from uuid import uuid4
import re
import base64
from app.core.security import role, origin_allowed, pair_allowed
from app.core.observability import metrics
from app.core.cameras import camera_registry
from app.core.streams import streams

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


async def generate_mjpeg(request=None):
    """Generator yielding multipart JPEG frames for the live dashboard player."""
    last_id = -1
    while True:
        if request is not None and await request.is_disconnected():
            return
        with video_manager.lock:
            jpeg, frame_id = video_manager.jpeg, video_manager.jpeg_id
        if jpeg and frame_id != last_id:
            last_id = frame_id
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        await asyncio.sleep(0.02)


@router.get("/video_feed")
async def video_feed(request: Request):
    """MJPEG Live Annotated Surveillance Stream."""
    return StreamingResponse(
        generate_mjpeg(request), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.post("/api/upload_frame")
async def upload_frame(request: Request, file: UploadFile = File(...)):
    """Receives camera frames uploaded directly by mobile browser."""
    contents = await file.read(512_001)
    if len(contents) > 512_000:
        raise HTTPException(413, "Frame too large")
    np_arr = np.frombuffer(contents, np.uint8)
    frame = await asyncio.to_thread(cv2.imdecode, np_arr, cv2.IMREAD_COLOR)
    if frame is not None:
        context = streams.for_device(request.cookies.get("vs_device", ""))
        if role(request) == "camera" and (not context or not context.busy):
            raise HTTPException(409, "Start this paired phone camera before uploading frames")
        target = context.video if context else video_manager
        await asyncio.to_thread(target.push_phone_frame, frame, target.generation)
        return {"status": "ok"}
    raise HTTPException(422, "Invalid image")


# --- REST APIs ---


@router.get("/api/info")
async def get_system_info():
    return {
        "local_ip": get_local_ip(),
        "fps": round(video_manager.fps, 1)
        if time.time() - video_manager.last_frame_time < 5
        else 0,
        "inference_ms": round(pipeline.inference_ms, 1),
        "processing_fps": round(pipeline.processing_fps, 1),
        "preview_fps": round(video_manager.preview_fps, 1),
        "detection_counts": pipeline.last_detection_counts,
        "source_path": video_manager.source_path,
        "camera_id": video_manager.camera_id,
        "camera": camera_registry.get(video_manager.camera_id),
        "pose": pipeline.pose.information(),
        "zone_immediate_warning": config.ZONE_IMMEDIATE_WARNING,
        "frame_dimensions": list(video_manager.latest_frame.shape[:2][::-1])
        if video_manager.latest_frame is not None
        else None,
        "source_error": video_manager.source_error,
        "recorded": video_manager.recorded,
        "camera_online": video_manager.camera_online
        and time.time() - video_manager.last_frame_time < 5,
        "detector_ready": pipeline.detector.model is not None,
        "detector_error": pipeline.detector.error,
        "backend": getattr(pipeline.detector, "backend", "torch"),
        "enabled_classes": pipeline.detector.enabled_labels,
        "input_size": config.INPUT_SIZE,
        "model": Path(config.YOLO_MODEL).name,
        "zone_dwell_seconds": config.ZONE_DWELL_SECONDS,
        "knife_confidence": config.KNIFE_CONFIDENCE,
        "report_mode": "Disabled"
        if not config.ENABLE_AI_DOSSIER
        else "Gemini"
        if incident_dispatcher._gemini_client
        else "Offline template",
        "remote_ready": config.PUBLIC_BASE_URL.startswith("https://"),
        "source_type": video_manager.source_type,
        "active_tracks": sum(
            t.disappeared_count == 0 for t in list(pipeline.tracker.tracks.values())
        )
        if video_manager.camera_online and time.time() - video_manager.last_frame_time < 5
        else 0,
        "zones": current_zones(),
    }


@router.post("/api/source")
async def change_source(payload: Dict[str, Any]):
    source = str(payload.get("source", "0"))
    if source not in {"0", "1", "phone_upload"}:
        try:
            video_manager.recording_path(source)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    try:
        await asyncio.to_thread(video_manager.start, source)
        pipeline.zone_source = None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
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
            source = pipeline.zone_source or video_manager.source_path
            if pipeline.geofence.source != source:
                pipeline.geofence.select_source(source)
            pipeline.geofence.replace_zones(validated)
            pipeline.crowd.history.clear()
            pipeline.crowd.pending.clear()
            pipeline.crowd.zone_metrics = []

    await asyncio.to_thread(apply)
    return {"status": "ok", "zones_count": len(validated)}


class ZonePayload(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=100)
    points: list[tuple[float, float]] = Field(min_length=3, max_length=20)
    severity: str = Field(default="HIGH", pattern="^(HIGH|MEDIUM|LOW)$")
    color: str = Field(default="#7dd3fc", pattern="^#[0-9a-fA-F]{6}$")
    purpose: Literal[
        "restricted", "object_monitor", "crowd_monitor", "river_edge", "pedestrian_only"
    ] = "restricted"
    dwell_seconds: float = Field(default=1.5, ge=0.5, le=300)
    max_capacity: int | None = Field(default=None, ge=1, le=100000)
    area_m2: float | None = Field(default=None, gt=0, le=10000000, allow_inf_nan=False)
    density_threshold: float | None = Field(default=None, gt=0, le=20, allow_inf_nan=False)
    minimum_spacing_m: float | None = Field(default=None, gt=0, le=100, allow_inf_nan=False)
    ground_width_m: float | None = Field(default=None, gt=0, le=10000, allow_inf_nan=False)
    ground_height_m: float | None = Field(default=None, gt=0, le=10000, allow_inf_nan=False)

    @model_validator(mode="after")
    def calibrated_policy(self):
        if bool(self.ground_width_m) != bool(self.ground_height_m):
            raise ValueError("Supply both measured ground dimensions")
        if self.ground_width_m:
            if len(self.points) != 4 or not cv2.isContourConvex(
                np.array(self.points, dtype=np.float32)
            ):
                raise ValueError(
                    "Ground calibration requires four ordered corners of a measured rectangle"
                )
            if (
                self.area_m2
                and abs(self.area_m2 - self.ground_width_m * self.ground_height_m)
                > 0.01 * self.area_m2
            ):
                raise ValueError("Area and measured rectangle dimensions disagree")
        if self.minimum_spacing_m and not self.ground_width_m:
            raise ValueError("Metre spacing requires a measured ground quadrilateral")
        if self.density_threshold and not (self.area_m2 or self.ground_width_m):
            raise ValueError("Density requires measured zone area")
        return self

    @field_validator("points")
    @classmethod
    def normalized_points(cls, points):
        if any(not 0 <= v <= 1 for point in points for v in point):
            raise ValueError("Coordinates must be normalized")
        if len(set(points)) != len(points):
            raise ValueError("Duplicate vertices")
        area = (
            abs(
                sum(
                    points[i][0] * points[(i + 1) % len(points)][1]
                    - points[(i + 1) % len(points)][0] * points[i][1]
                    for i in range(len(points))
                )
            )
            / 2
        )
        if area < 0.001:
            raise ValueError("Polygon is too small or degenerate")

        def orientation(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                if j == i + 1 or (i == 0 and j == n - 1):
                    continue
                a, b, c, d = (
                    points[i],
                    points[(i + 1) % n],
                    points[j],
                    points[(j + 1) % n],
                )
                if (
                    orientation(a, b, c) * orientation(a, b, d) <= 0
                    and orientation(c, d, a) * orientation(c, d, b) <= 0
                ):
                    raise ValueError("Polygon edges cross")
        return points


@router.get("/api/events")
def get_events(
    limit: int = Query(50, ge=1, le=200),
    camera: str | None = None,
    event_type: str | None = None,
    severity: Literal["HIGH", "MEDIUM", "LOW"] | None = None,
    status: Literal["new", "acknowledged", "false_positive"] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
):
    if start and end and start > end:
        raise HTTPException(422, "start must be <= end")
    return {
        "events": db.get_recent_events(
            limit,
            camera,
            event_type,
            severity,
            status,
            start.timestamp() if start else None,
            end.timestamp() if end else None,
        )
    }


@router.get("/api/events/{event_id}")
def event_details(event_id: str):
    event = db.get_event(event_id)
    if not event:
        raise HTTPException(404, "Event not found or expired")
    return event


@router.get("/api/events/{event_id}/evidence")
def event_evidence(event_id: str):
    event = db.get_event(event_id, include_evidence=True)
    if not event or not event.get("evidence_base64"):
        raise HTTPException(404, "Evidence unavailable, disabled or expired")
    return Response(
        base64.b64decode(event["evidence_base64"]),
        media_type="image/jpeg",
        headers={"Cache-Control": "private, no-store"},
    )


class ReviewPayload(BaseModel):
    status: Literal["new", "acknowledged", "false_positive"]


@router.patch("/api/events/{event_id}")
def review_event(event_id: str, payload: ReviewPayload, request: Request):
    if role(request) == "guard" and payload.status != "acknowledged":
        raise HTTPException(403, "Guards may acknowledge events only")
    if not db.review_event(event_id, payload.status):
        raise HTTPException(404, "Event not found")
    return db.get_event(event_id)


@router.delete("/api/events/{event_id}")
def delete_event(event_id: str):
    if not db.delete_event(event_id):
        raise HTTPException(404, "Event not found")
    return {"status": "deleted"}


@router.get("/api/incidents")
def get_incidents(limit: int = Query(10, ge=1, le=50)):
    return {"incidents": db.get_recent_incidents(limit)}


# --- WebSocket Channel ---


async def watch_disconnect(websocket):
    try:
        while True:
            if (await websocket.receive())["type"] == "websocket.disconnect":
                return
    except (WebSocketDisconnect, RuntimeError):
        return


@router.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """Sub-millisecond WebSocket channel for telemetry, alerts, and AI dossiers."""
    if not authorized(websocket):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    queue = event_hub.subscribe()
    disconnected = asyncio.create_task(watch_disconnect(websocket))
    pending_event = None

    # Send initial snapshot of recent events
    recent = event_hub.get_recent_events()
    try:
        for ev in recent:
            await websocket.send_text(json.dumps({**ev, "replay": True}))
        pending_event = asyncio.create_task(queue.get())
        while not disconnected.done():
            done, _ = await asyncio.wait(
                {pending_event, disconnected}, timeout=20, return_when=asyncio.FIRST_COMPLETED
            )
            if disconnected in done:
                break
            if pending_event in done:
                await websocket.send_json(pending_event.result())
                pending_event = asyncio.create_task(queue.get())
            else:
                await websocket.send_json({"type": "HEARTBEAT"})
    except WebSocketDisconnect:
        event_hub.unsubscribe(queue)
    except Exception as e:
        logger.warning(f"WebSocket client error: {e}")
        event_hub.unsubscribe(queue)
    finally:
        event_hub.unsubscribe(queue)
        tasks = [disconnected] + ([pending_event] if pending_event else [])
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def authorized(connection):
    return role(connection) in {"operator", "guard"} and origin_allowed(connection)


def phone_url(request, access="camera"):
    base = config.PUBLIC_BASE_URL or str(request.base_url).rstrip("/")
    if not base.startswith('https://'):
        raise HTTPException(503,'Secure phone link is starting or unavailable. Run python run.py (HTTPS tunnel is enabled by default), then retry Connect phone. Localhost links cannot connect other devices.')
    if access == "guard":
        return base + "/guard#token=" + config.GUARD_TOKEN
    return base + (
        "/login#token=" + config.ACCESS_TOKEN
        if access == "operator"
        else "/phone#token=" + config.CAMERA_TOKEN
    )


@router.post("/api/pair")
async def pair(request: Request):
    if not origin_allowed(request):
        raise HTTPException(403, "Origin is not allowed")
    if not pair_allowed(request.client.host if request.client else "unknown"):
        raise HTTPException(429, "Too many pairing attempts; retry in one minute")
    try:
        payload = await request.json()
    except ValueError as exc:
        raise HTTPException(422, "Expected a JSON token") from exc
    token = payload.get("token") if isinstance(payload, dict) else None
    if not isinstance(token, str) or not 1 <= len(token) <= 200:
        raise HTTPException(422, "Invalid token format")
    access = (
        "operator"
        if secrets.compare_digest(token, config.ACCESS_TOKEN)
        else "camera"
        if secrets.compare_digest(token, config.CAMERA_TOKEN)
        else "guard"
        if secrets.compare_digest(token, config.GUARD_TOKEN)
        else None
    )
    if not access:
        raise HTTPException(403, "Invalid pairing link. Scan a new QR code.")
    response = JSONResponse({"status": "paired", "role": access})
    response.set_cookie(
        "vs_access",
        token,
        httponly=True,
        secure=bool(config.PUBLIC_BASE_URL) or request.url.scheme == "https",
        samesite="strict",
    )
    device = request.cookies.get("vs_device", "")
    if not re.fullmatch("[0-9a-f]{32}", device):
        device = uuid4().hex
    response.set_cookie(
        "vs_device",
        device,
        httponly=True,
        secure=bool(config.PUBLIC_BASE_URL) or request.url.scheme == "https",
        samesite="strict",
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@router.get("/api/connection")
def connection_info(request: Request, access: Literal["camera", "operator", "guard"] = "camera"):
    return {
        "phone_url": phone_url(request, access),
        "remote_ready": bool(config.PUBLIC_BASE_URL),
        "access_role": access,
    }


@router.get("/api/qr")
def pairing_qr(request: Request, access: Literal["camera", "operator", "guard"] = "camera"):
    output = BytesIO()
    qrcode.make(phone_url(request, access)).save(output, format="PNG")
    return Response(
        output.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"}
    )


@router.get("/login", response_class=HTMLResponse)
def login_page():
    return HTMLResponse("""<!doctype html><html><meta name="viewport" content="width=device-width"><title>VisionShield pairing</title>
    <body style="background:#101923;color:#e0edf5;font:18px system-ui;padding:40px"><h1>Operator pairing</h1><p id="status">Pairing this browser...</p>
    <script>const token=new URLSearchParams(location.hash.slice(1)).get('token');history.replaceState(null,'',location.pathname);
    fetch('/api/pair',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token})}).then(r=>{if(!r.ok)throw Error('Pairing failed. Ask the operator for a new dashboard link.');location.replace('/')}).catch(e=>document.getElementById('status').textContent=e.message);</script></body></html>""")


@router.websocket("/ws/camera")
async def camera_socket(websocket: WebSocket):
    if role(websocket) not in {"camera", "operator"} or not origin_allowed(websocket):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    device = websocket.cookies.get("vs_device", "")
    if not re.fullmatch("[0-9a-f]{32}", device):
        device = uuid4().hex
    try:
        context = streams.claim(device)
    except ValueError as exc:
        await websocket.close(code=1013, reason=str(exc))
        return
    try:
        await asyncio.to_thread(context.video.start, "phone_upload")
    except Exception:
        streams.release(context, context.video.generation)
        raise
    context.video.camera_id = "phone:" + device
    stream_generation = context.video.generation
    try:
        while True:
            data = await asyncio.wait_for(websocket.receive_bytes(), timeout=30)
            if context.video.generation != stream_generation:
                await websocket.close(code=4000, reason="Camera source changed")
                return
            if len(data) > 512_000:
                await websocket.close(code=1009)
                return
            captured_at = None
            if data[:4] == b"VSC1":
                import struct

                if len(data) < 12:
                    continue
                captured_at = struct.unpack(">d", data[4:12])[0]
                if not time.time() - 30 <= captured_at <= time.time() + 1:
                    captured_at = None
                data = data[12:]
            frame = await asyncio.to_thread(
                cv2.imdecode, np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR
            )
            if frame is None or max(frame.shape[:2]) > 1920:
                await websocket.send_json({"error": "Invalid or oversized image"})
                continue
            accepted = await asyncio.to_thread(
                context.video.push_phone_frame,
                frame,
                stream_generation,
                captured_at,
            )
            if not accepted:
                await websocket.close(code=4000, reason="Camera source changed")
                return
            await websocket.send_json({"ok": True})
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    finally:
        streams.release(context, stream_generation)


camera_lock = asyncio.Lock()


@router.get("/api/telemetry")
async def live_telemetry():
    snapshot = pipeline.snapshot
    if (
        snapshot.get("generation") != video_manager.generation
        or not video_manager.camera_online
        or time.time() - snapshot.get("captured_at", 0) > 1
    ):
        return {
            "tracks": [],
            "zones": current_zones(),
            "confirmed_zones": [],
            "stale": True,
        }
    return {
        **snapshot,
        "stale": False,
        "age_ms": round((time.time() - snapshot["captured_at"]) * 1000),
    }


def current_zones():
    source = pipeline.zone_source or video_manager.source_path
    if pipeline.geofence.source == source:
        return pipeline.geofence.get_zones_dict()
    return pipeline.geofence.saved.get(source, [])


@router.get("/health/live")
def liveness():
    return {"status": "alive", "version": "3.0.0"}


@router.get("/health/ready")
def readiness():
    ready = pipeline.detector.model is not None and pipeline.detector.error is None
    return JSONResponse(
        {
            "status": "ready" if ready else "awaiting_source_or_model",
            "camera_online": video_manager.camera_online,
        },
        status_code=200 if ready else 503,
    )


@router.get("/api/time")
def server_time():
    return {"server_time_ms": time.time() * 1000}


@router.get("/api/camera/status")
def camera_status(request: Request):
    context = streams.for_device(request.cookies.get("vs_device", ""))
    video = context.video if context else video_manager
    return {
        "camera_online": video.camera_online,
        "source_error": video.source_error,
        "camera_id": video.camera_id,
    }


class CameraLocation(BaseModel):
    latitude: float = Field(ge=-90, le=90, allow_inf_nan=False)
    longitude: float = Field(ge=-180, le=180, allow_inf_nan=False)
    accuracy_m: float | None = Field(default=None, ge=0, le=100000, allow_inf_nan=False)


class CameraSettings(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    location: CameraLocation | None = None
    metres_per_pixel: float | None = Field(default=None, gt=0, le=100, allow_inf_nan=False)
    reference_width: int | None = Field(default=None, ge=32, le=7680)
    reference_height: int | None = Field(default=None, ge=32, le=7680)


@router.get("/api/camera/settings")
def camera_settings():
    return camera_registry.get(video_manager.camera_id)


@router.put("/api/camera/settings")
def update_camera_settings(payload: CameraSettings):
    if payload.metres_per_pixel and not (payload.reference_width and payload.reference_height):
        raise HTTPException(422, "Scale needs the reference frame dimensions")
    values = payload.model_dump()
    if values["location"] is not None:
        values["location"].update(source="operator_entered", timestamp=time.time())
    return camera_registry.update(video_manager.camera_id, values)


@router.post("/api/camera/location")
def camera_gps(request: Request, payload: CameraLocation):
    context = streams.for_device(request.cookies.get("vs_device", ""))
    video = context.video if context else video_manager
    if video.source_type != "phone_upload" or not video.camera_online:
        raise HTTPException(409, "Start the phone camera before sharing its GPS")
    device = request.cookies.get("vs_device", "")
    if role(request) == "camera" and video.camera_id != "phone:" + device:
        raise HTTPException(403, "GPS must belong to the active paired camera")
    camera_registry.update(
        video.camera_id,
        {
            "location": {
                **payload.model_dump(),
                "source": "camera_browser_gps",
                "timestamp": time.time(),
            }
        },
        persist=False,
    )
    return {
        "status": "camera_location_updated",
        "retention": "session memory; event evidence may include this camera location",
    }


class PersonAnnotation(BaseModel):
    label: Literal["man", "woman", "unspecified"]


@router.patch("/api/person-annotations/{person_id}")
def annotate_person(person_id: str, payload: PersonAnnotation):
    with pipeline.lock:
        if person_id not in pipeline.identities.people:
            raise HTTPException(404, "Anonymous person is no longer in this session")
        annotation = {
            "label": None if payload.label == "unspecified" else payload.label,
            "source": "operator_entered; not visually inferred",
        }
        pipeline.person_annotations[person_id] = annotation
    return annotation


@router.get("/api/metrics")
def live_metrics():
    return metrics.snapshot(video_manager, pipeline)


@router.get("/api/summary")
def dashboard_summary():
    return db.event_summary()


class DisplayMetric(BaseModel):
    client_id: str = Field(min_length=1, max_length=80, pattern="^[A-Za-z0-9-]+$")
    displayed_fps: float = Field(ge=0, le=120)
    display_latency_ms: float = Field(ge=0, le=60000)
    dropped_frames: int = Field(ge=0, le=10000000)
    clock_uncertainty_ms: float = Field(ge=0, le=60000)


@router.post("/api/client-metrics")
def display_metrics(payload: DisplayMetric):
    values = payload.model_dump()
    client_id = values.pop("client_id")
    metrics.display(client_id, values)
    return {"status": "ok"}


@router.get("/api/insights")
def insights():
    from app.core.model_registry import model_information

    return {
        "model": model_information(pipeline.detector),
        "pose": pipeline.pose.information(),
        "pipeline": dict(pipeline.stage_times),
        "metrics": metrics.snapshot(video_manager, pipeline),
        "identities": pipeline.identities.public_history()[-50:],
        "source": {
            "camera_id": video_manager.camera_id,
            "type": video_manager.source_type,
            "recorded": video_manager.recorded,
        },
        "rules": {
            "kind": "deterministic review policy, not a learned threat classifier",
            "knife_confidence": config.KNIFE_CONFIDENCE,
            "knife_min_observations": config.KNIFE_MIN_OBSERVATIONS,
            "knife_seconds": config.KNIFE_CONFIRM_SECONDS,
            "bag_seconds": config.ABANDONED_DWELL_SECONDS,
            "bag_requires_monitor_zone": config.REQUIRE_BAG_MONITOR_ZONE,
            "zone_seconds": config.ZONE_DWELL_SECONDS,
            "zone_immediate_warning": config.ZONE_IMMEDIATE_WARNING,
            "zone_classes": [config.PERSON_CLASS, *config.ANIMAL_CLASSES],
            "knife_immediate_visual_warning": config.KNIFE_IMMEDIATE_WARNING,
            "knife_requires_person": config.KNIFE_REQUIRE_PERSON,
            "duplicate_cooldown_seconds": config.ALERT_COOLDOWN_SECONDS,
        },
    }


@router.get("/api/identities")
def identity_history():
    return {
        "session_history": pipeline.identities.public_history(),
        "persisted_segments": db.get_identity_segments(),
        "scope": "anonymous camera/session associations; no face recognition; no stored descriptors",
    }


@router.get("/api/recordings")
def recordings():
    root = Path(config.RECORDINGS_DIR)
    return {
        "recordings": [
            {"source": "recording:" + p.name, "name": p.name, "recorded": True}
            for p in sorted(root.glob("*"))
            if p.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}
        ]
    }


@router.get("/api/diagnostics")
async def diagnostics(request: Request):
    import httpx
    from urllib.parse import urlsplit

    output = {
        "binding": config.HOST,
        "role": role(request),
        "secure_public_origin": config.PUBLIC_BASE_URL or None,
        "api": "reachable",
        "websocket_paths": ["/ws/events", "/ws/camera", "/ws/preview"],
        "cors": {
            "mode": "same-origin by default",
            "allowed_origins": config.ALLOWED_ORIGINS,
        },
        "stream": {
            "online": video_manager.camera_online,
            "error": video_manager.source_error,
        },
    }
    if not config.PUBLIC_BASE_URL:
        output["remote"] = {
            "code": "REMOTE_NOT_CONFIGURED",
            "detail": "Use --remote or configure a stable public HTTPS reverse proxy. 0.0.0.0 alone is not internet routing.",
        }
        return output
    host = urlsplit(config.PUBLIC_BASE_URL).hostname
    try:
        await asyncio.wait_for(asyncio.to_thread(socket.getaddrinfo, host, 443), 5)
    except (OSError, asyncio.TimeoutError):
        output["remote"] = {
            "code": "DNS_FAILURE",
            "detail": "Public hostname could not be resolved",
        }
        return output
    try:
        async with httpx.AsyncClient(timeout=5, trust_env=False, follow_redirects=False) as client:
            response = await client.get(config.PUBLIC_BASE_URL + "/health/live")
        code = (
            "OK"
            if response.status_code == 200
            else "AUTH_FAILURE"
            if response.status_code in {401, 403}
            else "BACKEND_UNREACHABLE"
        )
        output["remote"] = {"code": code, "http_status": response.status_code}
    except httpx.TransportError:
        output["remote"] = {
            "code": "BACKEND_UNREACHABLE",
            "detail": "DNS resolved but HTTPS backend/tunnel is unavailable",
        }
    return output


@router.websocket("/ws/preview")
async def preview_socket(websocket: WebSocket):
    if not authorized(websocket):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    last = -1
    disconnected = asyncio.create_task(watch_disconnect(websocket))
    try:
        while not disconnected.done():
            with video_manager.lock:
                jpeg, identifier, metadata = (
                    video_manager.jpeg,
                    video_manager.jpeg_id,
                    dict(video_manager.preview_metadata),
                )
            if jpeg and identifier != last:
                await websocket.send_json({"type": "FRAME", "sequence": identifier, **metadata})
                await websocket.send_bytes(jpeg)
                last = identifier
            await asyncio.sleep(0.01)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        disconnected.cancel()
        await asyncio.gather(disconnected, return_exceptions=True)
