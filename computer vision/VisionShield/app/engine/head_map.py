"""Actual head detections for an image-space crowd map; no face/identity inference."""

from types import SimpleNamespace
import time
import logging
from app.config import config, ROOT
from app.engine.accelerated_detector import AcceleratedYOLO


class HeadMapDetector:
    def __init__(self):
        self.model = None
        self.error = None

    def detect(self, frame):
        if self.model is None:
            path = ROOT / "data/models/v8s-head.pt"
            if not path.with_suffix(".onnx.json").exists():
                self.error = "Head model unavailable; run scripts/setup_head_model.py"
                return None
            try:
                options = SimpleNamespace(
                    CPU_THREADS=config.CPU_THREADS,
                    INPUT_SIZE=640,
                    CONFIDENCE_THRESHOLD=config.HEAD_CONFIDENCE,
                    IOU_THRESHOLD=0.45,
                    MAX_DETECTIONS=1500,
                )
                self.model = AcceleratedYOLO(path, options)
            except Exception as exc:
                self.error = str(exc)
                logging.getLogger(__name__).exception("Head model unavailable")
                return None
        started = time.perf_counter()
        try:
            # Four overlapping views enlarge distant heads. Ownership boundaries
            # assign every image location to one tile, preventing double counts.
            h, w = frame.shape[:2]
            detections = []
            for row in range(2 if config.HEAD_MAP_TILED else 1):
                for column in range(2 if config.HEAD_MAP_TILED else 1):
                    if not config.HEAD_MAP_TILED:
                        detections = self.model.detect(frame, [0])
                        break
                    x0 = max(0, round((column * 0.5 - 0.12) * w))
                    x1 = min(w, round(((column + 1) * 0.5 + 0.12) * w))
                    y0 = max(0, round((row * 0.5 - 0.12) * h))
                    y1 = min(h, round(((row + 1) * 0.5 + 0.12) * h))
                    for detection in self.model.detect(frame[y0:y1, x0:x1], [0]):
                        box = detection["box"]
                        cx = (box[0] + box[2]) / 2 + x0
                        cy = (box[1] + box[3]) / 2 + y0
                        if (
                            column * w / 2 <= cx < (column + 1) * w / 2
                            and row * h / 2 <= cy < (row + 1) * h / 2
                        ):
                            detections.append(
                                {
                                    "box": [box[0] + x0, box[1] + y0, box[2] + x0, box[3] + y0],
                                    "confidence": detection["confidence"],
                                }
                            )
            self.error = None
            return {
                "points": [
                    [
                        round((d["box"][0] + d["box"][2]) / 2, 1),
                        round((d["box"][1] + d["box"][3]) / 2, 1),
                        d["confidence"],
                    ]
                    for d in detections
                ],
                "count": len(detections),
                "inference_ms": round((time.perf_counter() - started) * 1000, 1),
                "method": "learned_head_detector",
                "model": "v8s-head, publisher pretrained; local ONNX",
                "coordinates": "image pixels, head centres; not ground positions or identity",
                "tiling": "2x2 overlapping crops with disjoint centre ownership"
                if config.HEAD_MAP_TILED
                else "full frame",
                "limit": 1500,
                "limit_reached": len(detections) >= 1500,
            }
        except Exception as exc:
            self.error = str(exc)
            return None


head_detector = HeadMapDetector()
