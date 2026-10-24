"""Evaluate frozen detector counting on a fixed Mall sample; never train on evaluation frames."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import io, json, zipfile
import cv2
import numpy as np
from scipy.io import loadmat
from app.engine.detector import ObjectDetector


def main():
    archive = ROOT / "datasets/crowd/mall/source.zip"
    detector = ObjectDetector()
    with zipfile.ZipFile(archive) as data:
        annotations = loadmat(
            io.BytesIO(data.read("mall_dataset/mall_gt.mat")), simplify_cells=True
        )
        print(
            {
                k: (type(v).__name__, np.shape(v))
                for k, v in annotations.items()
                if not k.startswith("_")
            },
            flush=True,
        )
        ground = annotations["frame"]
        selected = np.linspace(0, len(ground) - 1, 24, dtype=int)
        samples = []
        recording = ROOT / "data/recordings/Mall-counting-research.avi"
        recording.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(recording), cv2.VideoWriter_fourcc(*"MJPG"), 2, (640, 480))
        for index in selected:
            frame = cv2.imdecode(
                np.frombuffer(data.read(f"mall_dataset/frames/seq_{index + 1:06d}.jpg"), np.uint8),
                cv2.IMREAD_COLOR,
            )
            points = np.asarray(ground[index]["loc"]).reshape(-1, 2)
            detections = detector.detect(frame)
            predicted = sum(d["label"] == "person" for d in detections)
            samples.append(
                {
                    "frame": int(index + 1),
                    "annotated_heads": len(points),
                    "visible_person_detections": predicted,
                    "error": predicted - len(points),
                }
            )
            # Deliberately labelled sampled slideshow: frame order does not imply original time intervals.
            cv2.putText(
                frame,
                "RECORDED MALL SAMPLED FRAMES - NOT LIVE",
                (8, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                1,
            )
            for _ in range(4):
                writer.write(frame)
        writer.release()
    errors = np.array([r["error"] for r in samples])
    report = {
        "dataset": "Mall, fixed evenly spaced 24/2000 frames",
        "source": "https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html",
        "license": "research/noncommercial; cite original authors",
        "model": Path(detector.config.YOLO_MODEL).name
        if hasattr(detector, "config")
        else "yolo11s.pt",
        "backend": detector.backend,
        "trained_on_this_dataset": False,
        "sample_size": len(samples),
        "count_mae": float(np.abs(errors).mean()),
        "count_rmse": float(np.sqrt(np.mean(errors**2))),
        "count_bias": float(errors.mean()),
        "samples": samples,
        "limitations": [
            "Sparse single-scene sample, not independent multi-scene benchmark.",
            "Head-point counts compared with whole-image person detections; head and full-body definitions differ.",
            "Occlusion and small people cause undercounting; unsuitable for guaranteed attendance or life-safety capacity measurement.",
            "Slideshow uses sampled frames at artificial timing; not valid for motion or buildup evaluation.",
        ],
        "recording": str(recording.relative_to(ROOT)),
    }
    target = ROOT / "artifacts/crowd-counting-evaluation.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
