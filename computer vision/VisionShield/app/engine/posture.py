"""Low-rate, bounded pose enrichment using official COCO-keypoint weights.
Sitting/standing are geometric estimates, not separately trained action labels.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import defaultdict, deque
import time
import numpy as np
from app.config import config
from app.engine.tracker import box_iou


def angle(a, b, c):
    u = np.asarray(a) - b
    v = np.asarray(c) - b
    denominator = np.linalg.norm(u) * np.linalg.norm(v)
    if denominator < 1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / denominator, -1, 1))))


def classify_posture(keypoints):
    points = np.asarray(keypoints, dtype=float)
    if points.shape != (17, 3) or not np.isfinite(points).all():
        return "unknown", 0.0
    votes = []
    scores = []
    for shoulder, hip, knee, ankle in [(5, 11, 13, 15), (6, 12, 14, 16)]:
        joints = points[[shoulder, hip, knee, ankle]]
        if min(joints[:, 2]) < 0.4:
            continue
        s, h, k, a = joints[:, :2]
        knee_angle = angle(h, k, a)
        hip_angle = angle(s, h, k)
        if knee_angle is None or hip_angle is None:
            continue
        torso = max(1.0, np.linalg.norm(s - h))
        if abs(s[1] - h[1]) < 0.45 * torso:
            continue  # horizontal/foreshortened torso
        if knee_angle >= 155 and hip_angle >= 150 and a[1] > h[1]:
            votes.append("standing")
        elif knee_angle <= 145 and hip_angle <= 140 and abs(k[1] - h[1]) < 0.9 * torso:
            votes.append("sitting")
        else:
            votes.append("unknown")
        scores.append(float(min(joints[:, 2])))
    if not votes or len(set(votes)) != 1:
        return "unknown", 0.0
    return votes[0], round(min(scores), 3)


class PoseAnalyzer:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.model = None
        self.error = None
        self.future = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pose")
        self.last_submit = 0.0
        self.cache = []
        self.cache_time = 0.0
        self.generation = None
        self.inference_ms = 0.0
        self.history = defaultdict(lambda: deque(maxlen=3))

    def infer(self, frame, timestamp, generation):
        if self.model is None:
            if not Path(config.POSE_MODEL).is_file():
                raise FileNotFoundError("Pose checkpoint missing: run setup.ps1")
            from ultralytics import YOLO
            import torch

            torch.set_num_threads(config.CPU_THREADS)
            self.model = YOLO(config.POSE_MODEL)
        started = time.perf_counter()
        results = self.model.predict(
            frame, imgsz=320, conf=0.35, device=config.DEVICE, verbose=False, max_det=12
        )
        rows = []
        for result in results:
            if result.keypoints is None or result.boxes is None:
                continue
            for box, points in zip(
                result.boxes.xyxy.cpu().numpy(), result.keypoints.data.cpu().numpy()
            ):
                label, confidence = classify_posture(points)
                rows.append(
                    {
                        "box": box.tolist(),
                        "posture": label,
                        "confidence": confidence,
                        "keypoints": points.tolist(),
                    }
                )
        return rows, timestamp, generation, (time.perf_counter() - started) * 1000

    def apply(self, tracks, frame, timestamp, generation):
        people = [track for track in tracks if track.label == "person"]
        for track in people:
            track.posture = {
                "label": "unknown",
                "confidence": 0.0,
                "method": "COCO keypoints + joint angles",
                "limitation": "Occlusion, camera angle and bent/crouching poses can be ambiguous.",
            }
        if not self.enabled:
            return
        if generation != self.generation:
            self.cache = []
            self.cache_time = 0.0
            self.history.clear()
            self.generation = generation
        fresh = False
        if self.future and self.future.done():
            try:
                rows, source_time, source_generation, elapsed = self.future.result()
                if source_generation == generation:
                    self.cache = rows
                    self.cache_time = source_time
                    self.inference_ms = elapsed
                    fresh = True
                self.error = None
            except Exception as exc:
                self.error = str(exc)
            self.future = None
        age = timestamp - self.cache_time
        if 0 <= age <= config.POSE_MAX_AGE_SECONDS:
            edges = sorted(
                [
                    (box_iou(t.box, p["box"]), i, j)
                    for i, t in enumerate(people)
                    for j, p in enumerate(self.cache)
                ],
                reverse=True,
            )
            used_people = set()
            used_poses = set()
            for overlap, i, j in edges:
                if overlap < 0.35 or i in used_people or j in used_poses:
                    continue
                used_people.add(i)
                used_poses.add(j)
                t = people[i]
                p = self.cache[j]
                if fresh:
                    self.history[t.track_id].append(p["posture"])
                history = self.history[t.track_id]
                stable = (
                    len(history) >= 2 and history[-1] == history[-2] and p["posture"] != "unknown"
                )
                t.posture.update(
                    label=p["posture"] if stable else "unknown",
                    candidate=p["posture"],
                    confidence=p["confidence"],
                    age_seconds=round(age, 2),
                )
                if age <= 0.75 and overlap >= 0.5:
                    t.posture["keypoints"] = p["keypoints"]
        self.history = defaultdict(
            lambda: deque(maxlen=3), {t.track_id: self.history[t.track_id] for t in people}
        )
        if (
            people
            and not self.error
            and self.future is None
            and time.monotonic() - self.last_submit >= config.POSE_INTERVAL_SECONDS
        ):
            self.last_submit = time.monotonic()
            self.future = self.executor.submit(self.infer, frame.copy(), timestamp, generation)

    def information(self):
        return {
            "enabled": self.enabled,
            "model": Path(config.POSE_MODEL).name,
            "loaded": self.model is not None,
            "error": self.error,
            "inference_ms": round(self.inference_ms, 1),
            "input_size": 320,
            "training": "official vendor COCO-keypoints pretrained; posture joint-angle rules are not trained actions",
        }

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
