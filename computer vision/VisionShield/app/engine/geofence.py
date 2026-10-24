"""Dynamic Vector Geofencing Engine
Real-time Ray-Casting Point-in-Polygon (PIP) testing.
Supports interactive multi-zone polygons defined or modified live from the web dashboard.
"""

from typing import List
import time
from app.config import config


class GeofenceZone:
    def __init__(
        self,
        zone_id: str,
        name: str,
        points: List[List[float]],
        severity: str = "HIGH",
        color: str = "#ef4444",
        purpose="restricted",
        dwell_seconds=None,
        **policy,
    ):
        for field in (
            "max_capacity",
            "area_m2",
            "density_threshold",
            "minimum_spacing_m",
            "ground_width_m",
            "ground_height_m",
        ):
            setattr(self, field, policy.get(field))
        self.purpose = purpose
        self.dwell_seconds = config.ZONE_DWELL_SECONDS if dwell_seconds is None else dwell_seconds
        self.zone_id = zone_id
        self.name = name
        self.points = points  # Can be normalized 0..1 or pixel coords
        self.severity = severity
        self.color = color

    def contains_point(self, px: float, py: float, frame_width: int, frame_height: int) -> bool:
        """Ray-Casting algorithm to check if (px, py) is inside this polygon."""
        n = len(self.points)
        if n < 3:
            return False

        # Convert normalized points to pixel coordinates if necessary
        poly_pts = []
        for p in self.points:
            x = p[0] * frame_width if p[0] <= 1.0 else p[0]
            y = p[1] * frame_height if p[1] <= 1.0 else p[1]
            poly_pts.append((x, y))

        inside = False
        p1x, p1y = poly_pts[0]
        for i in range(1, n + 1):
            p2x, p2y = poly_pts[i % n]
            if py > min(p1y, p2y):
                if py <= max(p1y, p2y):
                    if px <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or px <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside


class GeofenceAnalyzer:
    def __init__(self, storage_path=None):
        self.zones = {}
        self.zone_dwell = {}
        self.confirmed_zones = set()
        self.storage_path = storage_path
        self.source = "0"
        self.saved = {}
        if storage_path:
            import json
            from pathlib import Path

            try:
                self.saved = json.loads(Path(storage_path).read_text(encoding="utf-8"))
            except FileNotFoundError:
                pass
            except (ValueError, OSError):
                import logging

                logging.getLogger(__name__).warning(
                    "Could not load saved zones; starting without restrictions"
                )
        self.select_source("0")

    def select_source(self, source):
        self.source = str(source)
        self.zones.clear()
        self.zone_dwell.clear()
        self.confirmed_zones.clear()
        for z in self.saved.get(self.source, config.DEFAULT_ZONES):
            self.add_zone(
                z["id"],
                z["name"],
                z["points"],
                z.get("severity", "HIGH"),
                z.get("color", "#7dd3fc"),
                z.get("purpose", "restricted"),
                z.get("dwell_seconds"),
                **{
                    k: z.get(k)
                    for k in (
                        "max_capacity",
                        "area_m2",
                        "density_threshold",
                        "minimum_spacing_m",
                        "ground_width_m",
                        "ground_height_m",
                    )
                },
            )

    def add_zone(
        self,
        zone_id,
        name,
        points,
        severity="HIGH",
        color="#7dd3fc",
        purpose="restricted",
        dwell_seconds=None,
        **policy,
    ):
        self.zones[zone_id] = GeofenceZone(
            zone_id, name, points, severity, color, purpose, dwell_seconds, **policy
        )
        self.zone_dwell[zone_id] = {}

    def get_zones_dict(self):
        return [
            {
                "id": z.zone_id,
                "name": z.name,
                "points": z.points,
                "severity": z.severity,
                "color": z.color,
                "purpose": z.purpose,
                "dwell_seconds": z.dwell_seconds,
                **{
                    k: getattr(z, k)
                    for k in (
                        "max_capacity",
                        "area_m2",
                        "density_threshold",
                        "minimum_spacing_m",
                        "ground_width_m",
                        "ground_height_m",
                    )
                },
            }
            for z in self.zones.values()
        ]

    def replace_zones(self, zones):
        self.zones.clear()
        self.zone_dwell.clear()
        self.confirmed_zones.clear()
        for z in zones:
            self.add_zone(
                z["id"],
                z["name"],
                z["points"],
                z["severity"],
                z["color"],
                z.get("purpose", "restricted"),
                z.get("dwell_seconds"),
                **{
                    k: z.get(k)
                    for k in (
                        "max_capacity",
                        "area_m2",
                        "density_threshold",
                        "minimum_spacing_m",
                        "ground_width_m",
                        "ground_height_m",
                    )
                },
            )
        self.saved[self.source] = self.get_zones_dict()
        if self.storage_path:
            import json
            from pathlib import Path

            target = Path(self.storage_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.saved, indent=2), encoding="utf-8")
            temporary.replace(target)

    def process_geofence(self, tracks, frame_w, frame_h, timestamp):
        events = []
        self.confirmed_zones.clear()
        for zone_id, zone in self.zones.items():
            if zone.purpose not in {"restricted", "allowed_area"}:
                continue
            current_inside = set()
            for track in tracks:
                if track.label not in {config.PERSON_CLASS, *config.ANIMAL_CLASSES}:
                    continue
                cx = (track.box[0] + track.box[2]) / 2
                foot_y = track.box[3]
                inside = zone.contains_point(cx, foot_y, frame_w, frame_h)
                # An allowed-area polygon is the safe inner radius. Outside is red.
                breach = inside if zone.purpose == "restricted" else not inside
                if not breach:
                    continue
                current_inside.add(track.track_id)
                state = self.zone_dwell[zone_id].get(track.track_id)
                if state is None or timestamp - state["last"] > 2:
                    state = {
                        "entered": timestamp,
                        "last": timestamp,
                        "count": 0,
                        "alerted": False,
                    }
                state["last"] = timestamp
                state["count"] += 1
                self.zone_dwell[zone_id][track.track_id] = state
                dwell = timestamp - state["entered"]
                immediate = config.ZONE_IMMEDIATE_WARNING
                confirmed = immediate or (
                    dwell >= zone.dwell_seconds and state["count"] >= config.ZONE_MIN_OBSERVATIONS
                )
                level = "HIGH" if immediate else zone.severity if confirmed else "MEDIUM"
                score = 80 if confirmed else 35
                if score >= track.threat_score:
                    track.threat_score = score
                    track.threat_level = level
                    track.active_alert = (
                        (
                            "OUTSIDE ALLOWED AREA - CHECK IMMEDIATELY: "
                            if zone.purpose == "allowed_area"
                            else "RESTRICTED ZONE - CHECK IMMEDIATELY: "
                        )
                        if confirmed
                        else "ZONE ENTRY PENDING: "
                    ) + zone.name
                if not confirmed:
                    continue
                self.confirmed_zones.add(zone_id)
                if state["alerted"]:
                    continue
                state["alerted"] = True
                events.append(
                    {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "alert_type": "PERIMETER_BREACH",
                        "severity": level,
                        "track_id": track.track_id,
                        "object_label": track.label,
                        "threat_score": score,
                        "zone_id": zone_id,
                        "display_label": (
                            "OUTSIDE ALLOWED AREA - CHECK IMMEDIATELY"
                            if zone.purpose == "allowed_area"
                            else "RESTRICTED ZONE - CHECK IMMEDIATELY"
                        ),
                        "explanation": f"Detected {track.label} #{track.track_id} {'outside the allowed area' if zone.purpose == 'allowed_area' else 'inside'} '{zone.name}'. Check immediately. This is a configured zone warning; verify the scene.",
                        "metadata": {
                            "zone_name": zone.name,
                            "dwell_seconds": round(dwell, 1),
                            "anchor": "bottom_center",
                            "confidence": track.confidence,
                            "required_dwell": 0 if immediate else zone.dwell_seconds,
                            "observations": state["count"],
                            "confirmation_mode": "immediate_detection"
                            if immediate
                            else "temporal_dwell",
                        },
                    }
                )
            self.zone_dwell[zone_id] = {
                tid: state
                for tid, state in self.zone_dwell[zone_id].items()
                if tid in current_inside
            }
        return events
