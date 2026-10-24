"""Actual pose inference smoke check, not posture accuracy evaluation."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
import time
from hashlib import sha256
from collections import Counter
import cv2
import httpx
from app.config import config
from app.engine.posture import PoseAnalyzer


def main():
    source = ROOT / "data/pose-smoke-bus.jpg"
    if not source.exists():
        response = httpx.get(
            "https://ultralytics.com/images/bus.jpg", follow_redirects=True, timeout=30
        )
        response.raise_for_status()
        source.write_bytes(response.content)
    images = [source, *sorted((ROOT / "datasets/coco_val_sample").glob("*.jpg"))[:10]]
    analyzer = PoseAnalyzer()
    rows = []
    counts = Counter()
    try:
        for image in images:
            frame = cv2.imread(str(image))
            poses, _, _, elapsed = analyzer.infer(frame, time.time(), 1)
            counts.update(pose["posture"] for pose in poses)
            rows.append(
                {
                    "image": image.name,
                    "postures": [pose["posture"] for pose in poses],
                    "confidence": [pose["confidence"] for pose in poses],
                    "pose_inference_ms": round(elapsed, 1),
                }
            )
    finally:
        analyzer.close()
    path = Path(config.POSE_MODEL)
    report = {
        "checkpoint": path.name,
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "source": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt",
        "training": "official vendor COCO-keypoints pretrained; no local pose training",
        "license": "Ultralytics AGPL-3.0 or commercial terms; check sample image rights",
        "posture_method": "17 COCO keypoints + confidence-gated hip/knee angle rules; no trained sitting/standing action classifier",
        "input_size": 320,
        "images": rows,
        "predicted_posture_counts": dict(counts),
        "scope": "Actual model inference smoke check. No posture ground truth or posture accuracy measured. Temporal display smoothing is tested separately.",
    }
    (ROOT / "artifacts/pose-verification.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
