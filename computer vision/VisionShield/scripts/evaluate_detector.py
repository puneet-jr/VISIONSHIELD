"""Reproducible small-sample detector check. Not a full COCO or danger benchmark."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import config
from app.engine.detector import ObjectDetector


def iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(union, 1e-9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=config.YOLO_MODEL)
    parser.add_argument("--size", type=int, default=config.INPUT_SIZE)
    parser.add_argument("--output", default="artifacts/knife-validation.json")
    args = parser.parse_args()
    config.INPUT_SIZE = args.size
    detector = ObjectDetector(model_name=args.model)
    source = ROOT / "datasets/coco_val_sample"
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    totals = {str(c): {"tp": 0, "fp": 0, "fn": 0} for c in [0.4, 0.55]}
    for entry in manifest:
        image = cv2.imread(str(source / entry["file"]))
        h, w = image.shape[:2]
        truth = []
        for line in (source / Path(entry["file"]).with_suffix(".txt")).read_text().splitlines():
            cls, cx, cy, bw, bh = map(float, line.split())
            if int(cls) == 43:
                truth.append(
                    [
                        (cx - bw / 2) * w,
                        (cy - bh / 2) * h,
                        (cx + bw / 2) * w,
                        (cy + bh / 2) * h,
                    ]
                )
        started = time.perf_counter()
        predictions = [d for d in detector.detect(image) if d["label"] == "knife"]
        duration = (time.perf_counter() - started) * 1000
        row = {
            "image": entry["file"],
            "ground_truth_knives": len(truth),
            "predictions": [{"confidence": d["confidence"], "box": d["box"]} for d in predictions],
            "elapsed_ms": round(duration, 1),
        }
        for threshold in [0.4, 0.55]:
            matched = set()
            tp = fp = 0
            for detection in sorted(predictions, key=lambda d: -d["confidence"]):
                if detection["confidence"] < threshold:
                    continue
                candidates = [
                    (iou(detection["box"], box), i)
                    for i, box in enumerate(truth)
                    if i not in matched
                ]
                best = max(candidates, default=(0, -1))
                if best[0] >= 0.5:
                    matched.add(best[1])
                    tp += 1
                else:
                    fp += 1
            counts = {"tp": tp, "fp": fp, "fn": len(truth) - len(matched)}
            row[str(threshold)] = counts
            for key in counts:
                totals[str(threshold)][key] += counts[key]
        rows.append(row)
        print(
            entry["file"],
            "knives annotated:",
            len(truth),
            "detected:",
            len(predictions),
            flush=True,
        )
    for counts in totals.values():
        counts["precision"] = round(counts["tp"] / max(1, counts["tp"] + counts["fp"]), 3)
        counts["recall"] = round(counts["tp"] / max(1, counts["tp"] + counts["fn"]), 3)
    model_path = Path(args.model)
    report = {
        "model": model_path.name,
        "sha256": sha256(model_path.read_bytes()).hexdigest(),
        "input_size": args.size,
        "split": "COCO val2017",
        "sample_design": "first 10 knife-positive and first 10 knife-negative image IDs, not representative; no custom training",
        "iou_threshold": 0.5,
        "threshold_results": totals,
        "rows": rows,
        "limitations": "20 selected images do not establish deployment accuracy, hand contact, dangerous intent, or video alert precision. Temporal/person gates are tested separately.",
    }
    target = ROOT / args.output
    target.parent.mkdir(exist_ok=True, parents=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if detector.error:
        raise RuntimeError(detector.error)
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
