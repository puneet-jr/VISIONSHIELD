"""Simulated policy evaluation; input boxes are fixtures, never live detections."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
from collections import Counter
from app.engine.tracker import TrackedObject
from app.engine.geofence import GeofenceAnalyzer
from app.engine.spatio_temporal import SpatioTemporalAnalyzer
from app.engine.weapon import WeaponAnalyzer
from app.engine.assessment import AssessmentEngine


def track(tid, label, box, confidence=0.9):
    x1, y1, x2, y2 = box
    t = TrackedObject(
        tid,
        {
            "label": label,
            "class_id": 0,
            "box": box,
            "centroid": ((x1 + x2) / 2, (y1 + y2) / 2),
            "confidence": confidence,
            "aspect_ratio": (y2 - y1) / (x2 - x1),
        },
        100,
    )
    t.is_stationary = True
    return t


def main():
    rows = []
    scenarios = {
        "normal": None,
        "bag_no_zone": None,
        "bag_monitored": "UNATTENDED_OBJECT_CANDIDATE",
        "zone_brief": None,
        "zone_sustained": "PERIMETER_BREACH",
        "knife_low_confidence": None,
        "knife_isolated": None,
        "knife_persistent": "POSSIBLE_WEAPON",
        "knife_far_from_person": None,
        "bag_with_nearby_person": None,
    }
    for scenario, expected in scenarios.items():
        zones = GeofenceAnalyzer()
        bag_engine = SpatioTemporalAnalyzer()
        weapon = WeaponAnalyzer()
        assessment = AssessmentEngine()
        if scenario.startswith("zone_"):
            zones.add_zone("restricted", "Restricted", [[0, 0], [0.5, 0], [0.5, 1], [0, 1]])
        if scenario in {"bag_monitored", "bag_with_nearby_person"}:
            zones.add_zone(
                "monitor",
                "Bag monitor",
                [[0, 0], [1, 0], [1, 1], [0, 1]],
                purpose="object_monitor",
            )
        p = track(1, "person", [20, 20, 100, 200])
        bag = track(2, "backpack", [115, 120, 150, 180])
        knife = track(
            3,
            "knife",
            [60, 70, 80, 100],
            0.3 if scenario == "knife_low_confidence" else 0.9,
        )
        if scenario == "knife_far_from_person":
            knife = track(3, "knife", [500, 70, 520, 100])
        found = []
        first_time = None
        for index in range(41):
            timestamp = 100 + index * 0.25
            tracks = [p]
            if scenario.startswith("bag_"):
                tracks = [bag] if scenario != "bag_with_nearby_person" else [p, bag]
            if scenario.startswith("knife_") and (scenario != "knife_isolated" or index == 0):
                tracks = [p, knife]
            if scenario == "zone_brief" and index >= 3:
                tracks = []
            raw = (
                zones.process_geofence(tracks, 640, 480, timestamp)
                + bag_engine.process_associations(tracks, timestamp, zones.zones, (640, 480))
                + weapon.process(tracks, timestamp)
            )
            for candidate in raw:
                event = assessment.assess(candidate, tracks, "simulated", timestamp)
                if event:
                    found.append(event["alert_type"])
                    if first_time is None:
                        first_time = timestamp - 100
        actual = found[0] if found else None
        rows.append(
            {
                "scenario": scenario,
                "expected": expected,
                "actual": actual,
                "events": len(found),
                "first_alert_seconds": first_time,
                "matches_expectation": actual == expected,
                "simulated_duration_seconds": 10,
            }
        )
    labels = sorted({value for row in rows for value in [row["expected"], row["actual"]] if value})
    per_class = {}
    for label in labels:
        tp = sum(r["expected"] == label and r["actual"] == label for r in rows)
        fp = sum(r["expected"] != label and r["actual"] == label for r in rows)
        fn = sum(r["expected"] == label and r["actual"] != label for r in rows)
        per_class[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn),
            "F1": 2 * tp / max(1, 2 * tp + fp + fn),
        }
    negatives = [row for row in rows if row["expected"] is None]
    false_alarms = sum(row["actual"] is not None for row in negatives)
    report = {
        "scope": "SIMULATED deterministic policy fixtures, not real detection or threat accuracy",
        "scenario_results": rows,
        "per_class": per_class,
        "confusion": dict(
            Counter(f"{r['expected'] or 'NONE'} -> {r['actual'] or 'NONE'}" for r in rows)
        ),
        "negative_simulated_seconds": len(negatives) * 10,
        "false_alarms_per_simulated_hour": false_alarms / max(1, len(negatives) * 10) * 3600,
        "limitations": "Artificial boxes and short scripted durations verify gates only. No real-world false-alarm rate, ownership, carrying or intent is measured.",
    }
    (ROOT / "artifacts/event-evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not all(row["matches_expectation"] for row in rows):
        raise SystemExit("Policy fixture failed")


if __name__ == "__main__":
    main()
