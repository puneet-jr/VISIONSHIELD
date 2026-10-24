"""Spatio-Temporal Entity Association Engine
Novel Computer Vision Layer: Solves the Unattended Baggage & Theft Detection problem.

Algorithm:
1. Ingress Association: When a bag/suitcase enters the scene near a person, they are bound as (Owner, Item).
2. Spatial Separation Tracking: Monitors Euclidean distance between Owner and Item.
3. Abandonment Trigger: If separation > delta_threshold and Item is stationary for > T_abandon,
   raise CRITICAL UNATTENDED HAZARD threat.
4. Unauthorized Retrieval: If a stranger approaches or grabs an unattended item, trigger THEFT ALERT.
"""
import numpy as np
import time
from typing import List, Dict, Any, Optional
from app.config import config
from app.engine.tracker import TrackedObject

class SpatioTemporalAnalyzer:
    def __init__(self):
        self.luggage_classes = set(config.LUGGAGE_CLASSES)
        self.person_class = config.PERSON_CLASS
        self.owner_attach_distance = config.OWNER_ATTACH_DISTANCE_PX
        self.min_separation = config.ABANDONED_MIN_SEPARATION_PX
        self.abandon_dwell_time = config.ABANDONED_DWELL_SECONDS

    def process_associations(self, tracks: List[TrackedObject], timestamp: float) -> List[Dict[str, Any]]:
        """Process owner-luggage associations and detect unattended baggage / theft events."""
        persons: Dict[int, TrackedObject] = {}
        luggage: Dict[int, TrackedObject] = {}
        events: List[Dict[str, Any]] = []

        for t in tracks:
            if t.label == self.person_class:
                persons[t.track_id] = t
            elif t.label in self.luggage_classes:
                luggage[t.track_id] = t

        for item_id, item in luggage.items():
            if not item.is_stationary:
                item.abandoned_since = None
            # Step 1: Initial Owner Binding if not already assigned
            if item.owner_id is None:
                nearest_person_id = None
                min_dist = float("inf")
                
                for pid, person in persons.items():
                    dist = np.hypot(item.centroid[0] - person.centroid[0], item.centroid[1] - person.centroid[1])
                    if dist < min_dist and dist <= self.owner_attach_distance:
                        min_dist = dist
                        nearest_person_id = pid

                if nearest_person_id is not None:
                    item.owner_id = nearest_person_id
                    item.abandoned_since = None
                else:
                    # Item entered frame with no visible owner nearby -> Immediate suspicious object
                    if item.abandoned_since is None and item.is_stationary:
                        item.abandoned_since = timestamp

            # Step 2: Continuous Separation Evaluation
            if item.owner_id is not None:
                owner = persons.get(item.owner_id)

                if owner is None:
                    # Owner completely left the camera field of view!
                    if item.abandoned_since is None and item.is_stationary:
                        item.abandoned_since = timestamp
                else:
                    # Owner is still in the frame -> check distance
                    sep_dist = np.hypot(item.centroid[0] - owner.centroid[0], item.centroid[1] - owner.centroid[1])
                    if sep_dist > self.min_separation and item.is_stationary:
                        if item.abandoned_since is None:
                            item.abandoned_since = timestamp
                    else:
                        # Owner is close enough -> reset abandonment timer
                        item.abandoned_since = None
                        item.threat_score = 10.0
                        item.threat_level = "LOW"
                        item.active_alert = None

            # Step 3: Check Abandonment Dwell Threshold
            if item.abandoned_since is not None:
                elapsed_unattended = timestamp - item.abandoned_since
                if elapsed_unattended >= self.abandon_dwell_time:
                    item.threat_score = min(98.0, 70.0 + (elapsed_unattended * 2.0))
                    item.threat_level = "HIGH"
                    item.active_alert = f"UNATTENDED LUGGAGE ({round(elapsed_unattended, 1)}s)"
                    
                    events.append({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "alert_type": "UNATTENDED_LUGGAGE",
                        "severity": "HIGH",
                        "track_id": item.track_id,
                        "object_label": item.label,
                        "threat_score": item.threat_score,
                        "explanation": f"Unattended {item.label} (ID #{item.track_id}) left stationary for {round(elapsed_unattended, 1)}s without owner.",
                        "metadata": {
                            "item_centroid": item.centroid,
                            "original_owner_id": item.owner_id,
                            "elapsed_seconds": round(elapsed_unattended, 1)
                        }
                    })

            # Step 4: Unauthorized Custody / Theft Check
            # If an item is currently abandoned and a DIFFERENT person enters close proximity
            if item.abandoned_since is not None and item.active_alert:
                for pid, person in persons.items():
                    if pid != item.owner_id:
                        p_dist = np.hypot(item.centroid[0] - person.centroid[0], item.centroid[1] - person.centroid[1])
                        if p_dist < 80.0:
                            events.append({
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "alert_type": "SUSPICIOUS_CUSTODY_TRANSFER",
                                "severity": "HIGH",
                                "track_id": item.track_id,
                                "object_label": item.label,
                                "threat_score": 95.0,
                                "explanation": f"Unknown subject (Person #{pid}) approaching unattended {item.label} (ID #{item.track_id})!",
                                "metadata": {
                                    "original_owner": item.owner_id,
                                    "interceptor_id": pid
                                }
                            })

        return events
