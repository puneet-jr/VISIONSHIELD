import tempfile
from pathlib import Path
import unittest
from app.engine.tracker import TrackedObject
from app.engine.geofence import GeofenceAnalyzer
from app.engine.weapon import WeaponAnalyzer
from app.web.routes import ZonePayload


def track(tid, label, box, confidence=0.9):
    x1, y1, x2, y2 = box
    return TrackedObject(
        tid,
        {
            "label": label,
            "class_id": 0,
            "box": box,
            "centroid": ((x1 + x2) / 2, (y1 + y2) / 2),
            "confidence": confidence,
            "aspect_ratio": (y2 - y1) / max(1, x2 - x1),
        },
        100,
    )


class ZoneTests(unittest.TestCase):
    def engine(self):
        engine = GeofenceAnalyzer()
        engine.add_zone("door", "Staff doorway", [[0.5, 0.5], [0.9, 0.5], [0.9, 0.95], [0.5, 0.95]])
        return engine

    def test_bags_do_not_trigger_zone_breaches(self):
        engine = self.engine()
        bag = track(1, "backpack", [55, 55, 65, 70])
        for ts in [100, 101, 102]:
            self.assertEqual(engine.process_geofence([bag], 100, 100, ts), [])
        self.assertFalse(engine.confirmed_zones)

    def test_feet_dwell_confirmation_exit_and_reentry(self):
        engine = self.engine()
        person = track(1, "person", [55, 10, 65, 70])  # centroid outside, feet inside
        self.assertEqual(engine.process_geofence([person], 100, 100, 100), [])
        self.assertEqual(engine.process_geofence([person], 100, 100, 100.8), [])
        events = engine.process_geofence([person], 100, 100, 101.6)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["metadata"]["anchor"], "bottom_center")
        self.assertEqual(engine.process_geofence([person], 100, 100, 102), [])
        engine.process_geofence([], 100, 100, 102.1)
        self.assertFalse(engine.confirmed_zones)
        self.assertEqual(engine.process_geofence([person], 100, 100, 102.2), [])

    def test_zones_persist_separately_per_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "zones.json"
            engine = GeofenceAnalyzer(path)
            zone = {
                "id": "door",
                "name": "Door",
                "points": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]],
                "severity": "HIGH",
                "color": "#7dd3fc",
            }
            engine.replace_zones([zone])
            engine.select_source("phone_upload")
            self.assertEqual(engine.get_zones_dict(), [])
            engine = GeofenceAnalyzer(path)
            self.assertEqual(len(engine.get_zones_dict()), 1)
            engine.replace_zones([])
            self.assertEqual(GeofenceAnalyzer(path).get_zones_dict(), [])

    def test_degenerate_and_crossed_zones_rejected(self):
        for points in [
            [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
            [[0.1, 0.1], [0.9, 0.9], [0.1, 0.9], [0.9, 0.1]],
            [[0.1, 0.1], [0.1, 0.1], [0.9, 0.9]],
        ]:
            with self.assertRaises(ValueError):
                ZonePayload(id="bad", name="Bad", points=points)


class KnifeTests(unittest.TestCase):
    def test_persistent_near_person_candidate(self):
        engine = WeaponAnalyzer()
        person = track(1, "person", [10, 10, 100, 190])
        knife = track(2, "knife", [80, 80, 95, 120])
        self.assertEqual(engine.process([person, knife], 100), [])
        self.assertEqual(knife.threat_level, "MEDIUM")
        self.assertEqual(engine.process([person, knife], 100.5), [])
        events = engine.process([person, knife], 101.1)
        self.assertEqual(events[0]["alert_type"], "POSSIBLE_WEAPON")
        self.assertEqual(knife.threat_level, "HIGH")
        engine.process([], 101.2)
        self.assertEqual(engine.process([person, knife], 101.3), [])

    def test_isolated_low_confidence_and_one_frame_knives_do_not_escalate(self):
        knife = track(2, "knife", [80, 80, 95, 120])
        engine = WeaponAnalyzer()
        for ts in [100, 100.5, 101.1]:
            self.assertEqual(engine.process([knife], ts), [])
        person = track(1, "person", [10, 10, 100, 190])
        knife.confidence = 0.5
        for ts in [102, 103, 104]:
            self.assertEqual(engine.process([person, knife], ts), [])


class PipelineStateTests(unittest.TestCase):
    def test_red_state_clears_when_person_leaves_zone(self):
        import numpy as np
        from unittest.mock import patch
        from app.engine.pipeline import SurveillancePipeline
        from app.ingestion.video_stream import video_manager
        from app.config import config

        pipe = SurveillancePipeline(persist=False)
        try:
            video_manager.start("phone_upload")
            pipe.generation = video_manager.generation
            pipe.geofence = GeofenceAnalyzer()
            pipe.geofence.add_zone(
                "floor", "Floor", [[0.5, 0.5], [0.9, 0.5], [0.9, 0.95], [0.5, 0.95]]
            )
            person = track(1, "person", [55, 10, 65, 70])
            detection = {
                "box": person.box,
                "centroid": person.centroid,
                "confidence": 0.9,
                "label": "person",
                "class_id": 0,
                "aspect_ratio": person.aspect_ratio,
            }
            with (
                patch.object(pipe.detector, "detect", return_value=[detection]),
                patch("app.engine.pipeline.db.insert_event"),
                patch.object(config, "ENABLE_AI_DOSSIER", False),
            ):
                for ts in [100, 100.8, 101.6]:
                    video_manager.push_phone_frame(np.zeros((100, 100, 3), dtype=np.uint8))
                    video_manager.last_frame_time = ts
                    with patch("app.engine.pipeline.time.time", return_value=ts):
                        pipe.run_cycle()
                self.assertEqual(pipe.tracker.tracks[1].threat_level, "HIGH")
                detection["box"] = [10, 10, 20, 70]
                detection["centroid"] = (15, 40)
                video_manager.push_phone_frame(np.zeros((100, 100, 3), dtype=np.uint8))
                video_manager.last_frame_time = 101.8
                with patch("app.engine.pipeline.time.time", return_value=101.8):
                    pipe.run_cycle()
                self.assertEqual(pipe.tracker.tracks[1].threat_level, "LOW")
                self.assertIsNone(pipe.tracker.tracks[1].active_alert)
                self.assertFalse(pipe.geofence.confirmed_zones)
        finally:
            video_manager.stop()
            pipe.executor.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
