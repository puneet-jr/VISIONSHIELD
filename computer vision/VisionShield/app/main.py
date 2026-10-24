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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
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
    await asyncio.to_thread(video_manager.start, "0")

    # 2. Start Background Vision Inference Thread
    _pipeline_running = True
    _pipeline_thread = threading.Thread(target=_surveillance_worker, daemon=True)
    _pipeline_thread.start()

    logger.info("VisionShield Engine is ONLINE and ready.")
    yield

    logger.info("Shutting down VisionShield Engine...")
    _pipeline_running = False
    if _pipeline_thread:
        await asyncio.to_thread(_pipeline_thread.join, 10.0)
    await asyncio.to_thread(video_manager.stop)
    pipeline.executor.shutdown(wait=False, cancel_futures=True)
    logger.info("VisionShield Engine cleanly stopped.")

app = FastAPI(
    title="VisionShield AI",
    description="Real-Time Multi-Object Threat Intelligence Platform",
    version="2.0.0",
    lifespan=lifespan
)

# Mount static assets
static_dir = os.path.join(os.path.dirname(__file__), "web", "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Include Web and API routes
app.include_router(router)


@app.middleware("http")
async def access_control(request, call_next):
    from app.web.routes import authorized
    from fastapi.responses import JSONResponse
    if request.url.path not in {"/phone", "/api/pair", "/static/js/mobile.js"} and not authorized(request):
        return JSONResponse({"detail": "Pair this device using the local dashboard QR code."}, status_code=401)
    response = await call_next(request)
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response
