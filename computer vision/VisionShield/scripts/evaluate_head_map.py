"""Frozen head detector evaluation on the same 24 Mall frames as the body baseline."""

from pathlib import Path
import sys, io, json, zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from scipy.io import loadmat
from app.engine.head_map import head_detector


def main():
    samples = []
    with zipfile.ZipFile(ROOT / "datasets/crowd/mall/source.zip") as data:
        annotations = loadmat(
            io.BytesIO(data.read("mall_dataset/mall_gt.mat")), simplify_cells=True
        )["frame"]
        for index in np.linspace(0, len(annotations) - 1, 24, dtype=int):
            image = cv2.imdecode(
                np.frombuffer(data.read(f"mall_dataset/frames/seq_{index + 1:06d}.jpg"), np.uint8),
                cv2.IMREAD_COLOR,
            )
            result = head_detector.detect(image)
            if result is None:
                raise RuntimeError(head_detector.error)
            expected = len(np.asarray(annotations[index]["loc"]).reshape(-1, 2))
            samples.append(
                {
                    "frame": int(index + 1),
                    "annotated_heads": expected,
                    "detected_heads": result["count"],
                    "error": result["count"] - expected,
                    "inference_ms": result["inference_ms"],
                }
            )
    errors = np.array([s["error"] for s in samples])
    blank = head_detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    report = {
        "dataset": "Mall, same fixed 24/2000 frames as body detector baseline",
        "model": "v8s-head publisher pretrained, local ONNX",
        "confidence": 0.35,
        "trained_locally": False,
        "sample_size": 24,
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "bias": float(errors.mean()),
        "body_baseline_mae": 18.875,
        "blank_false_detections": blank["count"],
        "samples": samples,
        "limitations": [
            "Small single-scene diagnostic, not held-out multi-scene validation.",
            "Publisher training dataset identities/splits undisclosed; Mall training overlap cannot be excluded.",
            "Not stampede prediction; head centres are image coordinates, not feet on a ground plane.",
        ],
    }
    (ROOT / "artifacts/head-counting-evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
