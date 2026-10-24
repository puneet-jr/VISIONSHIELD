"""Fixed-threshold deployed detector check on the locally held-out COCO smoke test."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
from collections import Counter
import cv2
from app.config import config
from app.engine.detector import ObjectDetector
from app.engine.tracker import box_iou
from app.research.datasets import audit


def main():
    root = ROOT / "datasets/security_smoke"
    manifest = json.loads((root / "manifest.json").read_text())
    check = audit(root)
    if not check["valid"]:
        raise ValueError(check["errors"])
    detector = ObjectDetector()
    counts = {}
    confusion = Counter()
    results = []
    names = manifest["names"]

    def count(label):
        return counts.setdefault(label, Counter(tp=0, fp=0, fn=0))

    for record in manifest["records"]:
        if record["split"] != "test":
            continue
        frame = cv2.imread(str(root / record["image"]))
        h, w = frame.shape[:2]
        gt = []
        for line in (root / record["label"]).read_text().splitlines():
            if not line.strip():
                continue
            cls, x, y, bw, bh = map(float, line.split())
            label = names[str(int(cls))] if isinstance(names, dict) else names[int(cls)]
            gt.append(
                {
                    "label": label,
                    "box": [
                        (x - bw / 2) * w,
                        (y - bh / 2) * h,
                        (x + bw / 2) * w,
                        (y + bh / 2) * h,
                    ],
                }
            )
        predictions = detector.detect(frame)
        if detector.error:
            raise RuntimeError(detector.error)
        # Confidence-sorted one-to-one IoU matching at a fixed deployed threshold.
        matched = set()
        for prediction in sorted(predictions, key=lambda p: -p["confidence"]):
            candidates = [
                (box_iou(prediction["box"], g["box"]), index)
                for index, g in enumerate(gt)
                if index not in matched
            ]
            overlap, index = max(candidates, default=(0, -1))
            if overlap >= 0.5:
                matched.add(index)
                truth = gt[index]["label"]
                predicted = prediction["label"]
                confusion[f"{truth} -> {predicted}"] += 1
                if truth == predicted:
                    count(truth)["tp"] += 1
                else:
                    count(truth)["fn"] += 1
                    count(predicted)["fp"] += 1
            else:
                count(prediction["label"])["fp"] += 1
                confusion[f"BACKGROUND -> {prediction['label']}"] += 1
        for index, g in enumerate(gt):
            if index not in matched:
                count(g["label"])["fn"] += 1
                confusion[f"{g['label']} -> BACKGROUND"] += 1
        results.append(
            {
                "image": record["image"],
                "ground_truth_boxes": len(gt),
                "predicted_boxes": len(predictions),
            }
        )
    for c in counts.values():
        c["precision"] = c["tp"] / max(1, c["tp"] + c["fp"])
        c["recall"] = c["tp"] / max(1, c["tp"] + c["fn"])
        c["F1"] = 2 * c["tp"] / max(1, 2 * c["tp"] + c["fp"] + c["fn"])
    report = {
        "scope": "Four COCO smoke test images held out from local fine-tuning only; not unbiased deployment evidence",
        "model": Path(config.YOLO_MODEL).name,
        "backend": detector.backend,
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
        "match_IoU": 0.5,
        "input_size": config.INPUT_SIZE,
        "per_class": counts,
        "confusion_sparse": dict(confusion),
        "images": results,
        "limitations": "Tiny selected sample. Default vendor model was trained on COCO; this is a smoke check. No threat classification accuracy is implied.",
    }
    (ROOT / "artifacts/detector-heldout.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
