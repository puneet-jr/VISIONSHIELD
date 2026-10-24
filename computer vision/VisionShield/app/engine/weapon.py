"""Knife candidates: confidence, temporal evidence, and person-box proximity.
This is not hand-pose estimation, weapon intent classification, or danger proof.
"""

import time
from app.config import config


class WeaponAnalyzer:
    def __init__(self):
        self.observations = {}

    def process(self, tracks, timestamp):
        events = []
        visible = {t.track_id for t in tracks if t.label == "knife"}
        self.observations = {k: v for k, v in self.observations.items() if k in visible}
        persons = [t for t in tracks if t.label == "person"]
        for item in tracks:
            if item.label != "knife":
                continue
            if config.KNIFE_IMMEDIATE_WARNING:
                item.threat_level = "HIGH"
                item.threat_score = max(item.threat_score, 85)
                item.active_alert = "DANGER - KNIFE DETECTED: CHECK IMMEDIATELY (UNCONFIRMED)"
            if item.confidence < config.KNIFE_CONFIDENCE:
                self.observations.pop(item.track_id, None)
                continue
            cx, cy = item.centroid
            nearby = []
            for person in persons:
                x1, y1, x2, y2 = person.box
                margin = 0.12 * max(x2 - x1, y2 - y1)
                if x1 - margin <= cx <= x2 + margin and y1 - margin <= cy <= y2 + margin:
                    nearby.append(person)
            person_id = (
                min(
                    nearby,
                    key=lambda p: (p.centroid[0] - cx) ** 2 + (p.centroid[1] - cy) ** 2,
                ).track_id
                if nearby
                else None
            )
            old = self.observations.get(item.track_id)
            if old is None or timestamp - old["last"] > 2 or old["person_id"] != person_id:
                old = {
                    "first": timestamp,
                    "last": timestamp,
                    "count": 0,
                    "person_id": person_id,
                }
            old["count"] += 1
            old["last"] = timestamp
            self.observations[item.track_id] = old
            item.threat_level = "HIGH" if config.KNIFE_IMMEDIATE_WARNING else "MEDIUM"
            item.threat_score = max(item.threat_score, 45)
            if not config.KNIFE_IMMEDIATE_WARNING:
                item.active_alert = "KNIFE CANDIDATE - VERIFY"
            confirmed = (
                old["count"] >= config.KNIFE_MIN_OBSERVATIONS
                and timestamp - old["first"] >= config.KNIFE_CONFIRM_SECONDS
            )
            if not confirmed or (config.KNIFE_REQUIRE_PERSON and person_id is None):
                continue
            item.threat_level = "HIGH"
            item.threat_score = max(item.threat_score, 85)
            item.active_alert = "DANGER - KNIFE DETECTED: CHECK IMMEDIATELY"
            events.append(
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "alert_type": "POSSIBLE_WEAPON",
                    "display_label": "DANGER - KNIFE DETECTED",
                    "severity": "HIGH",
                    "track_id": item.track_id,
                    "object_label": "knife",
                    "threat_score": 85,
                    "explanation": "Repeated knife detections. Check immediately; the model can be wrong. Possession, hand contact and intent are not established.",
                    "metadata": {
                        "confidence": round(item.confidence, 3),
                        "near_person_id": person_id,
                        "person_required": config.KNIFE_REQUIRE_PERSON,
                        "observations": old["count"],
                        "duration_seconds": round(timestamp - old["first"], 2),
                    },
                }
            )
        return events
