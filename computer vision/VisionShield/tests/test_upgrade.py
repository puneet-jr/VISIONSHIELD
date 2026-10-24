"""Regression coverage for access boundaries, evidence lifecycle and anonymous identity."""

import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.config import AppConfig, config
from app.db.database import Database
from app.engine.assessment import AssessmentEngine
from app.engine.detector import ObjectDetector
from app.engine.identity import IdentityManager
from app.engine.spatio_temporal import SpatioTemporalAnalyzer
from app.engine.tracker import TrackedObject
from app.ingestion.video_stream import VideoStreamManager
from app.main import app
from app.research.datasets import assign_groups, audit


def person(identifier=1, signature=(1.0, 0.0, 0.0)):
    t = TrackedObject(
        identifier,
        {
            "label": "person",
            "class_id": 0,
            "box": [10, 10, 50, 110],
            "centroid": (30, 60),
            "confidence": 0.9,
            "aspect_ratio": 2.5,
        },
        100,
    )
    t.appearance = np.array(signature, dtype=np.float32)
    return t


class IdentityTests(unittest.TestCase):
    def test_reset_links_segments_but_not_other_cameras(self):
        ids = IdentityManager()
        a = person()
        ids.associate([a], "camera-a", 1, 100)
        b = person()
        ids.associate([b], "camera-a", 2, 101)
        self.assertEqual(a.person_id, b.person_id)
        self.assertEqual(b.identity_association, "appearance_link_uncertain")
        c = person()
        ids.associate([c], "camera-b", 3, 102)
        self.assertNotEqual(c.person_id, a.person_id)
        self.assertEqual(len(ids.drain_history()), 3)
        self.assertFalse(ids.drain_history())

    def test_ambiguous_appearance_and_active_identity_not_stolen(self):
        ids = IdentityManager()
        a, b = person(1), person(2)
        ids.associate([a, b], "a", 1, 100)
        self.assertNotEqual(a.person_id, b.person_id)
        c = person(3)
        ids.associate([c], "a", 2, 102)
        self.assertEqual(c.identity_association, "ambiguous_new_identity")
        self.assertNotIn(c.person_id, [a.person_id, b.person_id])

    def test_expiry_discards_descriptor(self):
        ids = IdentityManager()
        a = person()
        ids.associate([a], "a", 1, 100)
        b = person()
        ids.associate([b], "a", 2, 100 + config.IDENTITY_RETENTION_SECONDS + 1)
        self.assertNotEqual(a.person_id, b.person_id)
        self.assertNotIn(a.person_id, ids.people)


class PolicyTests(unittest.TestCase):
    def test_explanation_and_cooldown(self):
        engine = AssessmentEngine()
        p = person()
        p.person_id = "P-test"
        raw = {
            "alert_type": "POSSIBLE_WEAPON",
            "track_id": 2,
            "object_label": "knife",
            "metadata": {
                "confidence": 0.9,
                "duration_seconds": 2,
                "observations": 4,
                "near_person_id": 1,
            },
        }
        event = engine.assess(raw, [p], "a", 100)
        self.assertEqual(event["severity"], "HIGH")
        self.assertFalse(event["assessment"]["learned_threat_model"])
        self.assertEqual(event["person_ids"], ["P-test"])
        self.assertTrue(event["assessment"]["limitations"])
        self.assertIsNone(engine.assess(raw, [p], "a", 100))
        raw["metadata"]["confidence"] = 0.2
        self.assertIsNone(engine.assess(raw, [p], "b", 100))

    def test_unowned_stationary_bag_outside_monitor_is_not_danger(self):
        bag = person()
        bag.label = "backpack"
        bag.is_stationary = True
        engine = SpatioTemporalAnalyzer()
        self.assertFalse(engine.process_associations([bag], 100))
        self.assertFalse(engine.process_associations([bag], 120))
        self.assertIsNone(bag.abandoned_since)


class StorageTests(unittest.TestCase):
    def test_protected_event_review_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Database(str(Path(directory) / "events.sqlite"))
            store.insert_event(
                {"event_id": "fixture", "alert_type": "SIMULATED_TEST", "camera_id": "a"}, "YWJj"
            )
            with (
                patch("app.web.routes.db", store),
                patch.object(config, "TRUST_LOCAL_DASHBOARD", False),
            ):
                client = TestClient(app)
                headers = {"Authorization": "Bearer " + config.ACCESS_TOKEN}
                self.assertEqual(client.get("/api/events/fixture/evidence").status_code, 401)
                self.assertEqual(
                    client.get("/api/events/fixture/evidence", headers=headers).content, b"abc"
                )
                self.assertEqual(
                    client.patch(
                        "/api/events/fixture", headers=headers, json={"status": "false_positive"}
                    ).status_code,
                    200,
                )
                self.assertEqual(
                    len(
                        client.get(
                            "/api/events?status=false_positive&camera=a", headers=headers
                        ).json()["events"]
                    ),
                    1,
                )
                self.assertEqual(
                    client.delete("/api/events/fixture", headers=headers).status_code, 200
                )
                self.assertEqual(
                    client.get("/api/events/fixture/evidence", headers=headers).status_code, 404
                )

    def test_migration_review_filter_evidence_and_retention(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "db.sqlite"
            store = Database(str(path))
            # Reconstruct an old schema by removing newly added columns through a fresh legacy table.
            with sqlite3.connect(path) as conn:
                conn.execute(
                    "INSERT INTO security_events(timestamp,alert_type,severity,object_label,threat_score,explanation) VALUES ('old','OLD','LOW','person',0,'history')"
                )
            conn.close()
            store = Database(str(path))
            old = store.get_recent_events()[0]
            self.assertTrue(old["legacy"])
            self.assertTrue(store.review_event(old["event_id"], "acknowledged"))
            event = {
                "event_id": "test-event",
                "camera_id": "a",
                "alert_type": "TEST",
                "severity": "MEDIUM",
                "occurred_at": time.time(),
            }
            store.insert_event(event, "YWJj")
            self.assertEqual(store.get_event("test-event", True)["evidence_base64"], "YWJj")
            self.assertNotIn("evidence_base64", store.get_event("test-event"))
            self.assertEqual(len(store.get_recent_events(camera="a")), 1)
            self.assertTrue(store.review_event("test-event", "false_positive"))
            self.assertEqual(len(store.get_recent_events(status="false_positive")), 1)
            store.prune(max_events=1)
            self.assertEqual(len(store.get_recent_events()), 1)
            self.assertTrue(store.delete_event("test-event"))
            self.assertIsNone(store.get_event("test-event", True))


class SecurityTests(unittest.TestCase):
    def test_camera_and_operator_boundaries_and_origin(self):
        with patch.object(config, "TRUST_LOCAL_DASHBOARD", False):
            client = TestClient(app)
            self.assertEqual(client.get("/api/info").status_code, 401)
            self.assertEqual(client.get("/health/live").status_code, 200)
            camera = {"Authorization": "Bearer " + config.CAMERA_TOKEN}
            for path in [
                "/api/events",
                "/api/info",
                "/api/insights",
                "/api/connection",
                "/static/js/dashboard.js",
            ]:
                self.assertEqual(client.get(path, headers=camera).status_code, 403, path)
            self.assertEqual(client.get("/api/camera/status", headers=camera).status_code, 200)
            operator = {"Authorization": "Bearer " + config.ACCESS_TOKEN}
            self.assertEqual(client.get("/api/info", headers=operator).status_code, 200)
            self.assertEqual(
                client.post(
                    "/api/source",
                    json={"source": "phone_upload"},
                    headers={**operator, "Origin": "https://untrusted.example"},
                ).status_code,
                403,
            )
            with self.assertRaises(Exception):
                with client.websocket_connect("/ws/preview", headers=camera):
                    pass
            with self.assertRaises(Exception):
                with client.websocket_connect(
                    "/ws/events",
                    headers={**operator, "Origin": "https://untrusted.example"},
                ):
                    pass

    def test_environment_validation_and_recording_confinement(self):
        for env in [
            {"INPUT_SIZE": "639"},
            {"KNIFE_CONFIDENCE": "1.2"},
            {"DEPLOYMENT_MODE": "production"},
            {"ALLOWED_ORIGINS": '["*"]'},
            {"PUBLIC_BASE_URL": "http://example.com"},
            {"TRACK_RETENTION_SECONDS": "nan"},
        ]:
            with self.assertRaises(ValueError):
                AppConfig.from_env(env)
        valid = AppConfig.from_env({"VISIONSHIELD_INPUT_SIZE": "320", "DEBUG": "false"})
        self.assertEqual(valid.INPUT_SIZE, 320)
        for name in [
            "recording:../outside.mp4",
            "recording:missing.mp4",
            "recording:a.txt",
        ]:
            with self.assertRaises(ValueError):
                VideoStreamManager.recording_path(name)


class RecoveryTests(unittest.TestCase):
    def test_failed_camera_reopens_without_blocking_api(self):
        first, second = MagicMock(), MagicMock()
        first.isOpened.return_value = False
        second.isOpened.return_value = True
        second.read.return_value = (True, np.zeros((120, 160, 3), dtype=np.uint8))
        manager = VideoStreamManager()
        with patch("app.ingestion.video_stream.cv2.VideoCapture", side_effect=[first, second]):
            try:
                manager.start("0")
                deadline = time.monotonic() + 4
                while not manager.camera_online and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(manager.camera_online)
                self.assertIsNone(manager.source_error)
            finally:
                manager.stop()
        self.assertTrue(first.release.called)

    def test_model_failure_is_visible_and_does_not_fake_predictions(self):
        detector = ObjectDetector()
        with (
            patch.object(config, "INFERENCE_BACKEND", "torch"),
            patch("ultralytics.YOLO", side_effect=RuntimeError("fixture load failure")) as load,
        ):
            self.assertFalse(detector.detect(np.zeros((32, 32, 3), dtype=np.uint8)))
            self.assertIn("fixture load failure", detector.error)
            detector.detect(np.zeros((32, 32, 3), dtype=np.uint8))
            self.assertEqual(load.call_count, 1)
        detector.model = MagicMock()
        detector.model.predict.side_effect = RuntimeError("fixture inference failure")
        self.assertFalse(detector.detect(np.zeros((32, 32, 3), dtype=np.uint8)))
        self.assertIn("fixture inference failure", detector.error)


class DatasetTests(unittest.TestCase):
    def test_group_splitting_keeps_sequences_together(self):
        records = [{"sequence_id": str(i // 2)} for i in range(20)]
        result = assign_groups(records)
        self.assertEqual(result, assign_groups(records))
        for group in {r["sequence_id"] for r in result}:
            self.assertEqual(len({r["split"] for r in result if r["sequence_id"] == group}), 1)
        self.assertEqual({r["split"] for r in result}, {"train", "val", "test"})

    def test_duplicate_and_invalid_annotations_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.zeros((30, 40, 3), dtype=np.uint8)
            cv2.imwrite(str(root / "a.jpg"), image)
            (root / "a.txt").write_text("0 .5 .5 2 2")
            manifest = {
                "names": ["person"],
                "records": [
                    {
                        "image": "a.jpg",
                        "label": "a.txt",
                        "sequence_id": "same",
                        "split": split,
                    }
                    for split in ["train", "val", "test"]
                ],
            }
            result = audit(root, manifest)
            self.assertFalse(result["valid"])
            for problem in [
                "Exact duplicate",
                "Near duplicate",
                "Sequence leakage",
                "Invalid YOLO",
            ]:
                self.assertTrue(any(problem in error for error in result["errors"]), problem)


if __name__ == "__main__":
    unittest.main()
