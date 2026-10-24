"""VisionShield FastAPI Application Entry Point
High-performance asynchronous server orchestrating vision ingestion,
processing pipeline, WebSockets, and tactical dashboard.
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import asyncio
import threading
import time
import logging
import os

from app.config import config
from app.core.event_hub import event_hub
from app.ingestion.video_stream import video_manager
from app.engine.pipeline import pipeline
from app.web.routes import router
from app.db.database import db
from app.core.observability import metrics
from app.core.logging_config import configure_logging

configure_logging()
logger = logging.getLogger("VisionShield.Main")

# Global background processing loop
_pipeline_thread = None
_pipeline_running = False


def _surveillance_worker():
    global _pipeline_running
    logger.info("Background Computer Vision processing worker started.")
    while _pipeline_running:
        started = time.perf_counter()
        try:
            with pipeline.lock:
                pipeline.run_cycle()
        except Exception as e:
            logger.error(f"Error in pipeline run cycle: {e}")
        time.sleep(max(0.005, 1 / config.TARGET_FPS - (time.perf_counter() - started)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline_thread, _pipeline_running
    logger.info("Initializing VisionShield Autonomous Surveillance Engine...")

    # 1. Start Video Stream Ingestion (Default Webcam 0 or synthetic fallback)
    event_hub.bind(asyncio.get_running_loop())
    await asyncio.to_thread(video_manager.start, os.getenv("START_SOURCE", "phone_upload"))

    # 2. Start Background Vision Inference Thread
    _pipeline_running = True
    _pipeline_thread = threading.Thread(target=_surveillance_worker, daemon=True)
    _pipeline_thread.start()

    async def maintenance():
        last_prune = time.monotonic()
        while True:
            metrics.sample()
            if time.monotonic() - last_prune > 3600:
                await asyncio.to_thread(db.prune)
                last_prune = time.monotonic()
            await asyncio.sleep(1)

    maintenance_task = asyncio.create_task(maintenance())
    await asyncio.to_thread(db.prune)
    logger.info(
        "VisionShield HTTP/preview services online; detector loads on the first camera frame."
    )
    try:
        yield
    finally:
        maintenance_task.cancel()
        try:
            await maintenance_task
        except asyncio.CancelledError:
            pass

    logger.info("Shutting down VisionShield Engine...")
    _pipeline_running = False
    if _pipeline_thread:
        await asyncio.to_thread(_pipeline_thread.join, 10.0)
    await asyncio.to_thread(video_manager.stop)
    pipeline.executor.shutdown(wait=False, cancel_futures=True)
    event_hub.bind(None)
    pipeline.identities.clear()
    logger.info("VisionShield Engine cleanly stopped.")


app = FastAPI(
    title="VisionShield AI",
    description="Real-Time Multi-Object Threat Intelligence Platform",
    version="3.0.0",
    lifespan=lifespan,
)

if config.ALLOWED_ORIGINS:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Content-Type", "Authorization"],
    )

# Mount static assets
static_dir = os.path.join(os.path.dirname(__file__), "web", "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Include Web and API routes
app.include_router(router)


@app.middleware("http")
async def access_control(request, call_next):
    from app.core.security import role, origin_allowed
    from fastapi.responses import JSONResponse

    public = {"/phone", "/login", "/api/pair", "/static/js/mobile.js", "/health/live"}
    camera_paths = {"/api/upload_frame", "/api/time", "/api/camera/status"}
    access = role(request)
    if request.url.path not in public:
        if access is None:
            return JSONResponse(
                {
                    "detail": "Pair this browser using an operator-issued link.",
                    "code": "AUTH_REQUIRED",
                },
                status_code=401,
            )
        if access == "camera" and request.url.path not in camera_paths:
            return JSONResponse(
                {
                    "detail": "This camera session cannot access operator data or controls.",
                    "code": "ROLE_FORBIDDEN",
                },
                status_code=403,
            )
    if request.method not in {"GET", "HEAD", "OPTIONS"} and not origin_allowed(request):
        return JSONResponse(
            {"detail": "Origin is not allowed", "code": "ORIGIN_FORBIDDEN"},
            status_code=403,
        )
    response = await call_next(request)
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = (
        "no-store"
        if request.url.path.startswith("/api/")
        else response.headers.get("Cache-Control", "no-cache")
    )
    return response
