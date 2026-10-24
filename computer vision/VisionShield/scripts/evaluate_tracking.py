"""Deterministic SIMULATED tracking fixtures; not a MOTChallenge accuracy claim."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
from types import SimpleNamespace
from collections import Counter, defaultdict
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from app.engine.tracker import ObjectTracker, box_iou
from app.engine.identity import IdentityManager


def detection(x):
    return {
        "label": "person",
        "class_id": 0,
        "box": [x, 30, x + 40, 150],
        "centroid": (x + 20, 90),
        "confidence": 0.9,
        "aspect_ratio": 3.0,
    }


def fixtures():
    result = {}
    for name in [
        "ordinary",
        "occlusion",
        "leave_return",
        "crossing",
        "similar_clothing",
        "simultaneous",
        "tracker_reset",
    ]:
        frames = []
        for i in range(90):
            people = [("A", int(30 + i * 3), (20, 30, 220))]
            if name in {"crossing", "similar_clothing", "simultaneous"}:
                people.append(
                    (
                        "B",
                        int(430 - i * 3),
                        (20, 30, 220) if name == "similar_clothing" else (220, 40, 20),
                    )
                )
            if name == "occlusion" and 30 <= i < 38:
                people = []
            if name == "leave_return" and 25 <= i < 65:
                people = []
            image = np.full((240, 640, 3), 35, dtype=np.uint8)
            for _, x, color in people:
                image[30:150, x : x + 40] = color
            frames.append((image, people))
        result[name] = frames
    return result


def match_frame(people, predictions):
    if not people or not predictions:
        return []
    overlaps = np.array(
        [[box_iou(detection(x)["box"], box) for box, _ in predictions] for _, x, _ in people]
    )
    a, b = linear_sum_assignment(-overlaps)
    return [(people[i][0], predictions[j][1]) for i, j in zip(a, b) if overlaps[i, j] > 0.5]


def summarize(observations, total_gt, total_predictions):
    pairs = Counter((gt, pred) for gt, pred in observations)
    ground = sorted({gt for gt, _ in observations})
    predictions = sorted({p for _, p in observations})
    matrix = np.array([[pairs[(g, p)] for p in predictions] for g in ground], dtype=float)
    a, b = linear_sum_assignment(-matrix) if matrix.size else ([], [])
    idtp = int(sum(matrix[i, j] for i, j in zip(a, b)))
    idfp = total_predictions - idtp
    idfn = total_gt - idtp
    histories = defaultdict(list)
    for gt, pred in observations:
        histories[gt].append(pred)
    switches = sum(sum(x != y for x, y in zip(ids, ids[1:])) for ids in histories.values())
    return {
        "IDTP": idtp,
        "IDFP": idfp,
        "IDFN": idfn,
        "IDF1": round(2 * idtp / max(1, 2 * idtp + idfp + idfn), 4),
        "identity_switches": switches,
        "identity_fragments": sum(max(0, len(set(ids)) - 1) for ids in histories.values()),
        "ground_truth_boxes": total_gt,
        "matched_predictions": len(observations),
        "total_predictions": total_predictions,
    }


def main():
    data = fixtures()
    output = {}
    for name, frames in data.items():
        tracker = ObjectTracker()
        identity = IdentityManager()
        generation = 1
        byte = BYTETracker(
            SimpleNamespace(
                track_high_thresh=0.5,
                track_low_thresh=0.1,
                new_track_thresh=0.6,
                track_buffer=30,
                match_thresh=0.8,
                fuse_score=True,
            )
        )
        observations = {"custom_track": [], "custom_person": [], "bytetrack": []}
        total = 0
        total_custom = 0
        total_byte = 0
        for index, (frame, people) in enumerate(frames):
            if name == "tracker_reset" and index == 45:
                tracker = ObjectTracker()
                generation += 1
            detections = [detection(x) for _, x, _ in people]
            timestamp = 100 + index / 10
            tracks = [
                t for t in tracker.update(detections, timestamp, frame) if t.disappeared_count == 0
            ]
            identity.associate(tracks, "simulated-camera", generation, timestamp)
            identity.drain_history()
            array = np.array(
                [[*d["box"], d["confidence"], 0] for d in detections], dtype=np.float32
            ).reshape(-1, 6)
            byte_tracks = byte.update(Boxes(array, frame.shape[:2]), frame)
            predicted = [(list(row[:4]), str(int(row[4]))) for row in byte_tracks]
            total += len(people)
            total_custom += len(tracks)
            total_byte += len(predicted)
            observations["custom_track"].extend(
                match_frame(people, [(t.box, f"{generation}:{t.track_id}") for t in tracks])
            )
            observations["custom_person"].extend(
                match_frame(people, [(t.box, t.person_id) for t in tracks])
            )
            observations["bytetrack"].extend(match_frame(people, predicted))
        output[name] = {
            key: summarize(values, total, total_byte if key == "bytetrack" else total_custom)
            for key, values in observations.items()
        }
    # A selectable recorded smoke source; its rectangles are NOT expected to be
    # detected by YOLO. Ground-truth detections are used only in this evaluator.
    root = ROOT / "data/recordings"
    root.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(root / "SIMULATED-transport.avi"),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10,
        (640, 240),
    )
    if not writer.isOpened():
        raise RuntimeError("Cannot create recorded smoke source")
    for index, (frame, _) in enumerate(data["crossing"]):
        cv2.putText(
            frame,
            "SIMULATED transport fixture - not detector accuracy",
            (10, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
        )
        writer.write(frame)
    writer.release()
    report = {
        "scope": "SIMULATED perfect input detections on coloured rectangles; no real-world accuracy claim",
        "metrics_definition": "IDF1 uses one-to-one global identity-count assignment. Switches count changes at matched observations; identity_fragments counts distinct IDs minus one (not official MOT fragmentation).",
        "tracker_selection": "Retain custom motion/colour + global Hungarian assignment. ByteTrack is evaluated as an alternative, not silently enabled.",
        "limitations": [
            "No learned ReID or facial recognition.",
            "Similar clothing can remain ambiguous.",
            "Synthetic detections omit detector errors and cannot establish deployment tracking quality.",
            "MOT17/MOT20 were not downloaded or evaluated.",
        ],
        "scenarios": output,
    }
    (ROOT / "artifacts/tracking-evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
