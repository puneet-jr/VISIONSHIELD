"""Explainable review-priority policy. This is rules-based, not a trained threat classifier."""

from datetime import datetime, timezone
from uuid import uuid4
import time
from app.config import config


class AssessmentEngine:
    def __init__(self):
        self.last_events = {}
        self.suppressed = 0

    def assess(self, candidate, tracks, camera_id, captured_at):
        now = time.time()
        track_map = {t.track_id: t for t in tracks}
        track = track_map.get(candidate.get("track_id"))
        metadata = candidate.get("metadata", {})
        kind = candidate.get("alert_type", "OBSERVATION")
        confidence = float(metadata.get("confidence", track.confidence if track else 0))
        persistence = float(
            metadata.get(
                "duration_seconds",
                metadata.get("elapsed_seconds", metadata.get("dwell_seconds", 0)),
            )
        )
        association = metadata.get("association", {})
        rules = []
        severity = candidate.get("severity", "LOW")
        uncertainty = [
            "Object boxes and application rules do not establish intent, authorization, ownership or verified danger."
        ]
        if kind == "POSSIBLE_WEAPON":
            count = metadata.get("observations", 0)
            valid = (
                confidence >= config.KNIFE_CONFIDENCE
                and persistence >= config.KNIFE_CONFIRM_SECONDS
                and count >= config.KNIFE_MIN_OBSERVATIONS
                and metadata.get("near_person_id") is not None
            )
            rules = [
                {
                    "id": "knife_confidence",
                    "observed": confidence,
                    "threshold": config.KNIFE_CONFIDENCE,
                },
                {
                    "id": "knife_persistence_seconds",
                    "observed": persistence,
                    "threshold": config.KNIFE_CONFIRM_SECONDS,
                },
                {
                    "id": "knife_observations",
                    "observed": count,
                    "threshold": config.KNIFE_MIN_OBSERVATIONS,
                },
                {
                    "id": "person_box_proximity",
                    "person_track_id": metadata.get("near_person_id"),
                    "method": "expanded person bounding box; not hand pose",
                },
            ]
            if not valid:
                self.suppressed += 1
                return None
            uncertainty.append(
                "Knife detection is weak on the documented validation sample; proximity does not prove possession or use."
            )
            severity = "HIGH"
        elif kind in {
            "UNATTENDED_LUGGAGE",
            "UNATTENDED_OBJECT_CANDIDATE",
            "OBJECT_PROXIMITY_CHANGE",
        }:
            severity = "MEDIUM"
            rules = [
                {
                    "id": "stationary_separation",
                    "observed_seconds": persistence,
                    "threshold_seconds": config.ABANDONED_DWELL_SECONDS,
                },
                {"id": "person_object_association", "evidence": association},
                {
                    "id": "monitored_spatial_context",
                    "zone_ids": metadata.get("zone_ids", []),
                },
            ]
            uncertainty.append(
                "A bag can be legitimately left still; association is a repeated geometric observation, not proof of ownership."
            )
        elif kind == "PERIMETER_BREACH":
            rules = [
                {"id": "person_footpoint_in_zone", "zone_id": candidate.get("zone_id")},
                {
                    "id": "zone_dwell",
                    "observed_seconds": persistence,
                    "threshold_seconds": metadata.get("required_dwell", config.ZONE_DWELL_SECONDS),
                },
            ]
            uncertainty.append(
                "Site authorization comes from operator policy, not inferred appearance."
            )
        elif kind == "FALL_DETECTED":
            rules = [
                {
                    "id": "possible_fall_geometry",
                    "aspect_ratio": metadata.get("aspect_ratio"),
                    "vertical_drop_px": metadata.get("vertical_drop_px"),
                }
            ]
            uncertainty.append(
                "This geometric heuristic is not a medically validated fall classifier."
            )
        else:
            rules = [
                {
                    "id": "configured_temporal_rule",
                    "event_type": kind,
                    "metadata": metadata,
                }
            ]
        association_person = track_map.get(
            metadata.get("near_person_id", metadata.get("original_owner_id"))
        )
        people = sorted(
            {
                p.person_id
                for p in [track, association_person]
                if p and getattr(p, "person_id", None)
            }
        )
        key = (camera_id, kind, candidate.get("track_id"), candidate.get("zone_id"))
        if now - self.last_events.get(key, 0) < config.ALERT_COOLDOWN_SECONDS:
            self.suppressed += 1
            return None
        self.last_events[key] = now
        self.last_events = {k: v for k, v in self.last_events.items() if now - v < 600}
        event_id = uuid4().hex
        score = {"LOW": 20, "MEDIUM": 55, "HIGH": 85}[severity]
        return {
            **candidate,
            "schema_version": 1,
            "event_id": event_id,
            "occurred_at": now,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "captured_at": captured_at,
            "camera_id": camera_id,
            "severity": severity,
            "threat_score": score,
            "priority": {
                "LOW": "informational_observation",
                "MEDIUM": "review_candidate",
                "HIGH": "elevated_review_priority",
            }[severity],
            "track_ids": sorted(
                {candidate["track_id"]}
                | ({association_person.track_id} if association_person else set())
            ),
            "person_ids": people,
            "detection_confidence": round(confidence, 3),
            "persistence_seconds": round(persistence, 3),
            "assessment": {
                "type": "rules_v1",
                "learned_threat_model": False,
                "score_meaning": "ordinal review priority, not probability of danger",
                "rules_fired": rules,
                "reasons": [candidate.get("explanation", "Configured observation")],
                "limitations": uncertainty,
            },
            "association_evidence": association
            or (
                {
                    "type": "person_box_proximity",
                    "person_track_id": association_person.track_id,
                    "ownership_proven": False,
                }
                if association_person
                else {}
            ),
            "evidence_url": f"/api/events/{event_id}/evidence",
            "status": "new",
        }
