"""Observed crowd risk indicators, not validated pressure/stampede prediction.
Metres are reported only from an operator-supplied quadrilateral ground plane.
"""

from collections import deque
import numpy as np
import cv2
from scipy.spatial.distance import cdist
from app.config import config


class CrowdDynamicsAnalyzer:
    def __init__(self):
        self.history = {}
        self.pending = {}
        self.zone_metrics = []

    def process(self, tracks, zones, width, height, timestamp):
        people = [t for t in tracks if t.label == "person" and t.disappeared_count == 0]
        vehicles = [
            t
            for t in tracks
            if t.label in {"car", "motorcycle", "bus", "truck", "bicycle", "train"}
            and t.disappeared_count == 0
        ]
        events = []
        metrics = []
        active_conditions = set()
        current_zone_ids = set(zones)
        self.history = {
            key: value for key, value in self.history.items() if key in current_zone_ids
        }
        for zone in zones.values():
            if zone.purpose not in {"crowd_monitor", "river_edge", "pedestrian_only"}:
                continue
            inside = [
                t
                for t in people
                if zone.contains_point((t.box[0] + t.box[2]) / 2, t.box[3], width, height)
            ]
            count = len(inside)
            positions = np.array(
                [[(t.box[0] + t.box[2]) / 2, t.box[3]] for t in inside], dtype=float
            ).reshape(-1, 2)
            area = zone.area_m2
            ground = None
            calibrated = False
            if zone.ground_width_m and zone.ground_height_m and len(zone.points) == 4:
                source = np.array(
                    [[x * width, y * height] for x, y in zone.points], dtype=np.float32
                )
                target = np.array(
                    [
                        [0, 0],
                        [zone.ground_width_m, 0],
                        [zone.ground_width_m, zone.ground_height_m],
                        [0, zone.ground_height_m],
                    ],
                    dtype=np.float32,
                )
                transform = cv2.getPerspectiveTransform(source, target)
                calibrated = True
                if count:
                    ground = cv2.perspectiveTransform(
                        positions.astype(np.float32).reshape(1, -1, 2), transform
                    )[0]
                area = zone.ground_width_m * zone.ground_height_m
            density = count / area if area else None
            nearest_px = nearest_m = None
            if count > 1:
                distances = cdist(positions, positions)
                np.fill_diagonal(distances, np.inf)
                nearest_px = float(np.mean(distances.min(axis=1)))
                if ground is not None:
                    distances_m = cdist(ground, ground)
                    np.fill_diagonal(distances_m, np.inf)
                    nearest_m = float(np.mean(distances_m.min(axis=1)))
            history = self.history.setdefault(zone.zone_id, deque(maxlen=240))
            if history and (timestamp <= history[-1][0] or timestamp - history[-1][0] > 2):
                history.clear()
            history.append((timestamp, count))
            while history and timestamp - history[0][0] > config.CROWD_GROWTH_WINDOW_SECONDS + 2:
                history.popleft()
            baseline = next(
                (
                    sample
                    for sample in history
                    if timestamp - sample[0] <= config.CROWD_GROWTH_WINDOW_SECONDS
                ),
                None,
            )
            growth = None
            if baseline and timestamp - baseline[0] >= config.CROWD_GROWTH_WINDOW_SECONDS - 0.5:
                growth = (count - baseline[1]) / max(1, baseline[1])
            vectors = np.array([t.motion for t in inside if t.velocity_valid], dtype=float).reshape(
                -1, 2
            )
            speed = float(np.mean(np.linalg.norm(vectors, axis=1))) if len(vectors) else None
            coherence = None
            if len(vectors) >= 3:
                magnitudes = np.linalg.norm(vectors, axis=1)
                moving = vectors[magnitudes > 5]
                if len(moving) >= 3:
                    coherence = float(
                        np.linalg.norm(
                            np.mean(moving / np.linalg.norm(moving, axis=1)[:, None], axis=0)
                        )
                    )
            occupancy = (
                "critical"
                if zone.max_capacity and count >= zone.max_capacity
                else "warning"
                if zone.max_capacity and count >= 0.8 * zone.max_capacity
                else "monitoring"
            )
            metric = {
                "zone_id": zone.zone_id,
                "zone_name": zone.name,
                "purpose": zone.purpose,
                "count": count,
                "max_capacity": zone.max_capacity,
                "area_m2": area,
                "observed_people_per_m2": round(density, 3) if density is not None else None,
                "mean_nearest_distance_px": round(nearest_px, 1)
                if nearest_px is not None
                else None,
                "mean_nearest_distance_m": round(nearest_m, 3) if nearest_m is not None else None,
                "growth_ratio_10s": round(growth, 3) if growth is not None else None,
                "mean_image_speed_px_s": round(speed, 1) if speed is not None else None,
                "flow_coherence": round(coherence, 3) if coherence is not None else None,
                "state": occupancy,
                "calibration": "quadrilateral ground plane, assumes visible feet on the measured plane"
                if calibrated
                else "area only"
                if area
                else "not calibrated; pixel measurements only",
                "count_scope": "visible person detections; occlusion can substantially undercount; no cross-camera sum or unique attendance",
            }
            metrics.append(metric)
            conditions = []
            if zone.max_capacity and count >= zone.max_capacity:
                conditions.append(
                    (
                        "CROWD_CAPACITY_EXCEEDED",
                        "HIGH",
                        "ZONE AT CAPACITY - REVIEW ENTRY / DIVERSION",
                        inside[0] if inside else None,
                    )
                )
            dense = (
                density is not None
                and zone.density_threshold is not None
                and density >= zone.density_threshold
            )
            close = (
                nearest_m is not None
                and zone.minimum_spacing_m is not None
                and nearest_m < zone.minimum_spacing_m
            )
            if dense or close:
                conditions.append(
                    (
                        "CROWD_DENSITY_CRITICAL",
                        "HIGH",
                        "HIGH OBSERVED CROWD DENSITY - CHECK ZONE",
                        inside[0] if inside else None,
                    )
                )
            if (
                growth is not None
                and growth > config.CROWD_GROWTH_THRESHOLD
                and count - baseline[1] >= config.CROWD_MIN_GROWTH
            ):
                conditions.append(
                    (
                        "CROWD_BUILDUP_WARNING",
                        "MEDIUM",
                        "RAPID CROWD BUILD-UP - GUARD REVIEW",
                        inside[0] if inside else None,
                    )
                )
            # Motion alignment is displayed as evidence, never alone called an imminent stampede.
            if zone.purpose == "pedestrian_only":
                for vehicle in vehicles:
                    if zone.contains_point(
                        (vehicle.box[0] + vehicle.box[2]) / 2, vehicle.box[3], width, height
                    ):
                        conditions.append(
                            (
                                "VEHICLE_INTRUSION",
                                "HIGH",
                                "VEHICLE IN PEDESTRIAN ZONE - CHECK PATH",
                                vehicle,
                            )
                        )
            for kind, severity, title, subject in conditions:
                if subject is None:
                    continue
                key = (zone.zone_id, kind, subject.track_id if kind == "VEHICLE_INTRUSION" else 0)
                active_conditions.add(key)
                state = self.pending.get(key)
                if state is None or timestamp - state["last"] > 2 or timestamp <= state["last"]:
                    state = {"since": timestamp, "observations": 0, "last": timestamp}
                    self.pending[key] = state
                state["last"] = timestamp
                state["observations"] += 1
                required = 3.0 if kind == "VEHICLE_INTRUSION" else config.CROWD_WARNING_SECONDS
                confirmed = timestamp - state["since"] >= required and state["observations"] >= 3
                previous_confirmed = metric.get("warning_confirmed", False)
                if confirmed and severity == "HIGH":
                    metric["state"] = "critical"
                elif not previous_confirmed:
                    metric["state"] = "warning"
                if not previous_confirmed or (confirmed and severity == "HIGH"):
                    metric["active_warning"] = title
                metric["warning_confirmed"] = previous_confirmed or confirmed
                if not confirmed:
                    continue
                for person in [subject] if kind == "VEHICLE_INTRUSION" else inside:
                    if person.threat_score < ({"HIGH": 85, "MEDIUM": 55}[severity]):
                        person.threat_score = {"HIGH": 85, "MEDIUM": 55}[severity]
                        person.threat_level = severity
                        person.active_alert = title
                events.append(
                    {
                        "alert_type": kind,
                        "severity": severity,
                        "track_id": subject.track_id,
                        "object_label": subject.label,
                        "zone_id": zone.zone_id,
                        "display_label": title,
                        "explanation": title
                        + ". Follow the site crowd-management plan after verifying camera evidence.",
                        "metadata": {
                            "confidence": subject.confidence,
                            "duration_seconds": round(timestamp - state["since"], 2),
                            "observations": state["observations"],
                            "required_seconds": required,
                            "crowd": metric.copy(),
                            "not_a_stampede_prediction": True,
                        },
                    }
                )
        self.pending = {
            key: state for key, state in self.pending.items() if key in active_conditions
        }
        self.zone_metrics = metrics
        return events
