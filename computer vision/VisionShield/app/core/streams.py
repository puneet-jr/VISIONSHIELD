"""Bounded independent camera contexts sharing one serial inference worker/model."""

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from app.config import config
from app.ingestion.video_stream import VideoStreamManager, video_manager
from app.engine.pipeline import SurveillancePipeline, pipeline
from app.engine.geofence import GeofenceAnalyzer


@dataclass
class StreamContext:
    key: str
    video: VideoStreamManager
    pipeline: SurveillancePipeline
    device: str = ""
    busy: bool = False
    last_used: float = 0


class StreamRegistry:
    def __init__(self):
        self.lock = threading.RLock()
        self.default = StreamContext("main", video_manager, pipeline)
        self.contexts = {"main": self.default}

    def all(self):
        with self.lock:
            return list(self.contexts.values())

    def get(self, key):
        with self.lock:
            return self.contexts.get(key)

    def for_device(self, device):
        with self.lock:
            return next(
                (ctx for ctx in self.contexts.values() if ctx.device == device and device), None
            )

    def claim(self, device):
        with self.lock:
            context = self.for_device(device)
            if context and context.busy:
                raise ValueError("This paired device already has a camera sender")
            if context and context.video.source_type != "phone_upload":
                context = None
            if context is None:
                if (
                    not self.default.busy
                    and self.default.video.source_type == "phone_upload"
                    and not self.default.video.camera_online
                ):
                    context = self.default
                else:
                    for key, old in list(self.contexts.items()):
                        if (
                            key != "main"
                            and not old.busy
                            and time.monotonic() - old.last_used > 120
                        ):
                            self._close(old)
                            self.contexts.pop(key)
                    if len(self.contexts) >= config.MAX_CAMERA_STREAMS:
                        raise ValueError(
                            "Camera limit reached; disconnect an unused camera and wait two minutes"
                        )
                    video = VideoStreamManager()
                    engine = SurveillancePipeline(
                        video=video,
                        detector=pipeline.detector,
                        zone_source="phone:" + device,
                        enable_pose=False,
                        enable_refinement=False,
                    )
                    engine.geofence = GeofenceAnalyzer(
                        str(Path(config.ZONES_PATH).with_name("zones-" + device + ".json"))
                    )
                    context = StreamContext(device, video, engine)
                    self.contexts[device] = context
            context.device = device
            context.busy = True
            context.last_used = time.monotonic()
            context.pipeline.zone_source = "phone:" + device
            return context

    def release(self, context, generation):
        with self.lock:
            context.busy = False
            context.last_used = time.monotonic()
            if context.video.generation == generation:
                context.video.camera_online = False

    @staticmethod
    def _close(context):
        context.video.stop()
        context.pipeline.pose.close()
        context.pipeline.executor.shutdown(wait=False, cancel_futures=True)
        context.pipeline.identities.clear()

    def close(self):
        for context in self.all():
            self._close(context)


streams = StreamRegistry()
