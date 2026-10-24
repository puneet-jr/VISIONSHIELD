"""Multi-camera crowd overview and operator-scoped zone configuration."""

import asyncio
import json
import time
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from app.config import ROOT
from app.core.security import role, origin_allowed
from app.core.cameras import camera_registry
from app.core.streams import streams
from app.web.routes import ZonePayload, watch_disconnect

router = APIRouter()


@router.get("/guard", response_class=HTMLResponse)
def guard_page():
    return HTMLResponse(
        (Path(__file__).parent / "templates/guard.html").read_text(encoding="utf-8")
    )


@router.get("/api/streams")
def stream_overview(request: Request):
    result = []
    for context in streams.all():
        video = context.video
        engine = context.pipeline
        snapshot = engine.snapshot
        online = video.camera_online and time.time() - video.last_frame_time < 5
        fresh = (
            online
            and snapshot.get("generation") == video.generation
            and time.time() - snapshot.get("captured_at", 0) < 2
        )
        zones = (
            engine.geofence.get_zones_dict()
            if engine.geofence.source == (engine.zone_source or video.source_path)
            else engine.geofence.saved.get(engine.zone_source or video.source_path, [])
        )
        result.append(
            {
                "id": context.key,
                "camera_id": video.camera_id,
                "camera": camera_registry.get(video.camera_id),
                "online": online,
                "stale": not fresh,
                "source_type": video.source_type,
                "recorded": video.recorded,
                "source_error": video.source_error,
                "width": snapshot.get("width", 640),
                "height": snapshot.get("height", 480),
                "captured_at": snapshot.get("captured_at"),
                "counts": snapshot.get("counts", {}) if fresh else {},
                "tracks": snapshot.get("tracks", []) if fresh else [],
                "head_map": snapshot.get("head_map") if fresh else None,
                "zones": zones,
                "crowd": snapshot.get("crowd", {"zones": []}) if fresh else {"zones": []},
                "inference_ms": engine.inference_ms,
                "processing_fps": engine.processing_fps if fresh else 0,
                "pose_enabled": engine.pose.enabled,
            }
        )
    return {
        "streams": result,
        "can_configure": role(request) == "operator",
        "count_scope": "Visible detections per camera; overlapping cameras are not summed as unique attendance.",
    }


class ZoneCollection(BaseModel):
    zones: list[ZonePayload] = Field(max_length=20)


@router.post("/api/streams/{key}/zones")
async def configure_stream(key: str, payload: ZoneCollection):
    context = streams.get(key)
    if not context:
        raise HTTPException(404, "Unknown camera stream")

    def apply():
        with context.pipeline.lock:
            engine = context.pipeline
            engine.geofence.select_source(engine.zone_source or context.video.source_path)
            engine.geofence.replace_zones([zone.model_dump() for zone in payload.zones])
            engine.crowd.history.clear()
            engine.crowd.pending.clear()
            engine.crowd.zone_metrics = []

    await asyncio.to_thread(apply)
    return {"status": "saved", "zones_count": len(payload.zones)}


@router.get("/api/datasets")
def dataset_status():
    path = ROOT / "artifacts/crowd-dataset-catalog.json"
    return (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {"sources": [], "trained": False}
    )


@router.websocket("/ws/streams/{key}/preview")
async def stream_preview(websocket: WebSocket, key: str):
    if role(websocket) not in {"operator", "guard"} or not origin_allowed(websocket):
        await websocket.close(code=1008)
        return
    context = streams.get(key)
    if not context:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    last = -1
    disconnected = asyncio.create_task(watch_disconnect(websocket))
    try:
        while not disconnected.done():
            with context.video.lock:
                jpeg = context.video.jpeg
                identifier = context.video.jpeg_id
            if jpeg and identifier != last:
                await websocket.send_bytes(jpeg)
                last = identifier
            await asyncio.sleep(
                0.04
            )  # Bounded shared latest JPEG; no decode or inference per viewer.
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        disconnected.cancel()
        await asyncio.gather(disconnected, return_exceptions=True)
