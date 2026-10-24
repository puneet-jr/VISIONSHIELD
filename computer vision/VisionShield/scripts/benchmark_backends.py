"""Offline side-by-side comparison; never starts a server or camera."""

from pathlib import Path
import sys
import json
import time
import statistics
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import config
from app.engine.detector import ObjectDetector
from app.engine.tracker import box_iou

images = sorted((ROOT / "datasets/coco_val_sample").glob("*.jpg"))
assert images, "Prepare the validation sample first"
config.INFERENCE_BACKEND = "torch"
torch = ObjectDetector()
frame = cv2.imread(str(images[0]))
torch.detect(frame)
config.INFERENCE_BACKEND = "onnx"
onnx = ObjectDetector()
onnx.detect(frame)
measurements = {"torch": [], "onnx-cpu": []}
rows = []
for path in images:
    frame = cv2.imread(str(path))
    outputs = {}
    for name, model in [("torch", torch), ("onnx-cpu", onnx)]:
        started = time.perf_counter()
        outputs[name] = model.detect(frame)
        measurements[name].append((time.perf_counter() - started) * 1000)
    a, b = outputs["torch"], outputs["onnx-cpu"]
    unmatched = list(b)
    matched = 0
    for detection in a:
        candidates = [
            (box_iou(detection["box"], other["box"]), i)
            for i, other in enumerate(unmatched)
            if other["label"] == detection["label"]
        ]
        overlap, index = max(candidates, default=(0, -1))
        if overlap > 0.98:
            matched += 1
            unmatched.pop(index)
    rows.append(
        {
            "image": path.name,
            "torch_boxes": len(a),
            "onnx_boxes": len(b),
            "matched_at_iou_098": matched,
        }
    )
report = {
    "input_size": config.INPUT_SIZE,
    "same_weights": True,
    "measurements_ms": measurements,
    "median_ms": {k: round(statistics.median(v), 1) for k, v in measurements.items()},
    "prediction_comparison": rows,
}
(ROOT / "artifacts/backend-benchmark.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print(
    json.dumps(
        {
            "median_ms": report["median_ms"],
            "matched_boxes": sum(r["matched_at_iou_098"] for r in rows),
            "torch_boxes": sum(r["torch_boxes"] for r in rows),
            "onnx_boxes": sum(r["onnx_boxes"] for r in rows),
        },
        indent=2,
    )
)
