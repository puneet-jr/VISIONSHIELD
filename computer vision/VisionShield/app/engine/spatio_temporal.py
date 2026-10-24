"""Repeated geometric person-object association; never proof of ownership or carrying."""

import time
import numpy as np
from app.config import config


class SpatioTemporalAnalyzer:
    def __init__(self):
        self.candidates = {}
        self.associations = {}

    @staticmethod
    def distance(item, person):
        x, y = item.centroid
        x1, y1, x2, y2 = person.box
        return float(np.hypot(max(x1 - x, 0, x - x2), max(y1 - y, 0, y - y2)))

    def process_associations(self, tracks, timestamp, zones=None, frame_size=None):
        persons = {t.track_id: t for t in tracks if t.label == config.PERSON_CLASS}
        items = [t for t in tracks if t.label in config.LUGGAGE_CLASSES]
        visible = {item.track_id for item in items}
        self.candidates = {k: v for k, v in self.candidates.items() if k in visible}
        self.associations = {k: v for k, v in self.associations.items() if k in visible}
        events = []
        for item in items:
            near = sorted((self.distance(item, person), pid) for pid, person in persons.items())
            nearest = near[0] if near and near[0][0] <= config.OWNER_ATTACH_DISTANCE_PX else None
            ambiguous = nearest and len(near) > 1 and near[1][0] - near[0][0] < 20
            if nearest and not ambiguous:
                distance, pid = nearest
                state = self.candidates.get(item.track_id)
                if not state or state["person_track_id"] != pid or timestamp - state["last"] > 2:
                    state = {
                        "person_track_id": pid,
                        "first": timestamp,
                        "last": timestamp,
                        "count": 0,
                    }
                state["last"] = timestamp
                state["count"] += 1
                self.candidates[item.track_id] = state
                if (
                    state["count"] >= config.ASSOCIATION_MIN_OBSERVATIONS
                    and timestamp - state["first"] >= config.ASSOCIATION_SECONDS
                ):
                    self.associations[item.track_id] = {
                        "type": "repeated_geometric_proximity",
                        "person_track_id": pid,
                        "person_id": getattr(persons[pid], "person_id", None),
                        "observations": state["count"],
                        "duration_seconds": round(timestamp - state["first"], 2),
                        "distance_px": round(distance, 1),
                        "confidence": round(
                            max(0, 1 - distance / config.OWNER_ATTACH_DISTANCE_PX)
                            * min(item.confidence, persons[pid].confidence),
                            3,
                        ),
                        "ownership_proven": False,
                        "carrying_proven": False,
                        "hand_contact_proven": False,
                    }
                    item.owner_id = pid
            else:
                self.candidates.pop(item.track_id, None)
            association = self.associations.get(item.track_id)
            item.association_evidence = association or {
                "type": "insufficient_evidence",
                "ambiguous": bool(ambiguous),
                "ownership_proven": False,
            }
            linked = persons.get(association["person_track_id"]) if association else None
            close = (
                linked is not None
                and self.distance(item, linked) <= config.ABANDONED_MIN_SEPARATION_PX
            )
            if close or not item.is_stationary:
                item.abandoned_since = None
                continue
            zone_ids = []
            if zones and frame_size:
                w, h = frame_size
                zone_ids = [
                    z.zone_id
                    for z in zones.values()
                    if z.purpose == "object_monitor" and z.contains_point(*item.centroid, w, h)
                ]
            if config.REQUIRE_BAG_MONITOR_ZONE and not zone_ids:
                item.abandoned_since = None
                continue
            # Without a prior associated person this remains a monitored stationary-object candidate.
            if item.abandoned_since is None:
                item.abandoned_since = timestamp
            elapsed = timestamp - item.abandoned_since
            if elapsed < config.ABANDONED_DWELL_SECONDS:
                continue
            item.threat_level = "MEDIUM"
            item.threat_score = 55
            item.active_alert = "UNATTENDED OBJECT CANDIDATE - REVIEW"
            events.append(
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "alert_type": "UNATTENDED_OBJECT_CANDIDATE",
                    "severity": "MEDIUM",
                    "track_id": item.track_id,
                    "object_label": item.label,
                    "threat_score": 55,
                    "explanation": f"Stationary {item.label} observed for {elapsed:.1f}s in a monitored area without a nearby associated person. Verify the scene; ownership is uncertain.",
                    "metadata": {
                        "confidence": item.confidence,
                        "elapsed_seconds": round(elapsed, 2),
                        "zone_ids": zone_ids,
                        "original_owner_id": association["person_track_id"]
                        if association
                        else None,
                        "association": item.association_evidence,
                    },
                }
            )
        return events
