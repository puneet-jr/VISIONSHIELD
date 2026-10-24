"""Bounded process/display metrics. Capture-to-display uses estimated clock alignment."""

from collections import deque
import time
import psutil


class Metrics:
    def __init__(self):
        self.process = psutil.Process()
        self.samples = deque(maxlen=120)
        self.clients = {}

    def sample(self):
        self.samples.append(
            {
                "timestamp": time.time(),
                "cpu_percent_one_core_100": self.process.cpu_percent(None),
                "rss_mb": round(self.process.memory_info().rss / 1024**2, 2),
            }
        )
        self.clients = {k: v for k, v in self.clients.items() if time.time() - v["updated_at"] < 15}

    def display(self, client_id, payload):
        self.clients[client_id] = {**payload, "updated_at": time.time()}
        if len(self.clients) > 32:
            del self.clients[next(iter(self.clients))]

    def snapshot(self, video, pipeline):
        denominator = pipeline.processed_frames + pipeline.skipped_frames
        return {
            "resource_samples": list(self.samples)[-10:],
            "display_clients": list(self.clients.values()),
            "capture_frames": video.source_frames,
            "preview_frames_encoded": video.jpeg_id - video.jpeg_start_id,
            "inference_frames": pipeline.processed_frames,
            "inference_dropped_frames": pipeline.skipped_frames,
            "inference_drop_percent": round(100 * pipeline.skipped_frames / max(1, denominator), 2),
            "preview_dropped_frames": video.preview_skipped,
            "preview_fps": round(video.preview_fps, 1),
            "inference_fps": round(pipeline.processing_fps, 1),
            "inference_ms": round(pipeline.inference_ms, 1),
            "suppressed_alerts": pipeline.assessment.suppressed,
            "stage_times_ms": dict(pipeline.stage_times),
            "latency_scope": "display estimates are client-reported capture-to-image-load with midpoint clock synchronization; hardware sensor delay is unmeasured",
        }


metrics = Metrics()
