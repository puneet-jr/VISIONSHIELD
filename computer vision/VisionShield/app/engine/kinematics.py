"""Kinematic & Behavioral Anomaly Engine
Calculates trajectory dynamics, directional entropy, and sudden aspect-ratio collapse.

1. Fall / Incapacitation Detection:
   Upright humans have Height/Width ratio ~ 1.8 to 3.0.
   A sudden collapse to Height/Width < 0.65 with downward acceleration flags an urgent FALL_DETECTED event.

2. Suspicious Loitering / Prowling:
   Calculates ratio of net displacement to total path length over time.
   Low net displacement + high path length + dwell > threshold indicates loitering / casing behavior.
"""
import numpy as np
import time
from typing import List, Dict, Any
from app.config import config
from app.engine.tracker import TrackedObject

class KinematicsAnalyzer:
    def __init__(self):
        self.fall_ar_threshold = config.FALL_ASPECT_RATIO_THRESHOLD
        self.loitering_time = config.LOITERING_TIME_SECONDS
        self.fall_until = {}

    def process_kinematics(self, tracks: List[TrackedObject], timestamp: float) -> List[Dict[str, Any]]:
        """Evaluate movement patterns, falls, and loitering across tracked persons."""
        events: List[Dict[str, Any]] = []

        active_ids = {t.track_id for t in tracks}
        self.fall_until = {k: v for k, v in self.fall_until.items() if k in active_ids and v > timestamp}
        for track in tracks:
            if track.label != config.PERSON_CLASS:
                continue

            if self.fall_until.get(track.track_id, 0) > timestamp:
                track.threat_level = "HIGH"
                track.threat_score = 88
                track.active_alert = "POSSIBLE FALL - VERIFY"

            # --- Check 1: Fall / Medical Incapacitation Detection ---
            # Aspect ratio H/W collapse
            if (track.aspect_ratio < self.fall_ar_threshold and track.previous_aspect_ratio >= 1.1
                    and len(track.centroid_history) >= 5
                    and track.centroid_history[-1][1] - track.centroid_history[-5][1] >= config.FALL_VELOCITY_THRESHOLD_PX):
                # Check if it was previously upright in the last 15 frames
                recent_points = track.centroid_history[-10:]
                dy = recent_points[-1][1] - recent_points[0][1]  # positive y is downwards in image coordinates
                
                track.threat_score = max(track.threat_score, 88.0)
                track.threat_level = "HIGH"
                track.active_alert = "POSSIBLE FALL - VERIFY"
                self.fall_until[track.track_id] = timestamp + 3

                events.append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "alert_type": "FALL_DETECTED",
                    "severity": "HIGH",
                    "track_id": track.track_id,
                    "object_label": track.label,
                    "threat_score": 88.0,
                    "explanation": f"Subject #{track.track_id} may have fallen. Aspect ratio dropped to {round(track.aspect_ratio, 2)} with sudden downward displacement.",
                    "metadata": {
                        "aspect_ratio": round(track.aspect_ratio, 2),
                        "vertical_drop_px": round(dy, 1)
                    }
                })

            # --- Check 2: Loitering / Prowling Anomaly ---
            if config.ENABLE_LOITERING and track.dwell_time > self.loitering_time and len(track.centroid_history) >= 20:
                history = np.array([(p[0], p[1]) for p in track.centroid_history])
                # Total distance traveled along trajectory
                diffs = np.diff(history, axis=0)
                path_length = np.sum(np.hypot(diffs[:, 0], diffs[:, 1]))
                
                # Net displacement between start and end
                net_displacement = np.hypot(history[-1][0] - history[0][0], history[-1][1] - history[0][1])
                
                # Trajectory efficiency ratio: straight line = 1.0, circling/pacing << 0.35
                efficiency = net_displacement / max(1.0, path_length)

                if efficiency < 0.28 and path_length > 100.0:
                    track.threat_score = max(track.threat_score, 65.0)
                    track.threat_level = "MEDIUM" if track.threat_score < 75 else "HIGH"
                    track.active_alert = f"LOITERING / PROWLING ({round(track.dwell_time, 0)}s)"

                    events.append({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "alert_type": "SUSPICIOUS_LOITERING",
                        "severity": "MEDIUM",
                        "track_id": track.track_id,
                        "object_label": track.label,
                        "threat_score": 65.0,
                        "explanation": f"Subject #{track.track_id} loitering for {round(track.dwell_time, 0)}s with erratic pacing (mobility efficiency: {round(efficiency, 2)}).",
                        "metadata": {
                            "dwell_seconds": round(track.dwell_time, 1),
                            "efficiency": round(efficiency, 2)
                        }
                    })

        return events
