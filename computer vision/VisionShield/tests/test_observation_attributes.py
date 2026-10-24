"""Posture/colour/motion and camera coordinate regressions; fixtures are simulated."""

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock
from types import SimpleNamespace
import asyncio
import numpy as np
from fastapi.testclient import TestClient
from app.config import config
from app.core.cameras import CameraRegistry, movement
from app.engine.attributes import crop_colour, describe_colours
from app.engine.geofence import GeofenceAnalyzer
from app.engine.posture import classify_posture, PoseAnalyzer
from app.engine.weapon import WeaponAnalyzer
from app.engine.assessment import AssessmentEngine
from app.main import app
from test_alert_rules import track


def joints(kind="standing"):
    points = np.zeros((17, 3))
    for shoulder, hip, knee, ankle, x in [(5, 11, 13, 15, 40), (6, 12, 14, 16, 50)]:
        values = (
            [(x, 20), (x, 70), (x, 120), (x, 170)]
            if kind == "standing"
            else [(x, 20), (x, 70), (x + 45, 75), (x + 45, 130)]
        )
        for index, (px, py) in zip([shoulder, hip, knee, ankle], values):
            points[index] = [px, py, 0.95]
    return points


class AttributeTests(unittest.TestCase):
    def test_keypoint_posture_and_occlusion(self):
        self.assertEqual(classify_posture(joints())[0], "standing")
        self.assertEqual(classify_posture(joints("sitting"))[0], "sitting")
        points = joints()
        points[[13, 14, 15, 16], 2] = 0
        self.assertEqual(classify_posture(points)[0], "unknown")
        self.assertEqual(classify_posture(np.full((17, 3), np.nan))[0], "unknown")

    def test_colour_labels_and_distinct_clothing_regions(self):
        for bgr, expected in [
            ((0, 0, 255), "red"),
            ((255, 0, 0), "blue"),
            ((0, 255, 0), "green"),
            ((0, 0, 0), "black"),
            ((255, 255, 255), "white"),
        ]:
            frame = np.full((200, 100, 3), bgr, dtype=np.uint8)
            self.assertEqual(crop_colour(frame, [0, 0, 100, 200])["name"], expected)
        frame = np.zeros((200, 100, 3), dtype=np.uint8)
        frame[:110] = (0, 0, 255)
        frame[110:] = (255, 0, 0)
        person = track(1, "person", [0, 0, 100, 200])
        describe_colours(frame, person)
        self.assertEqual(person.colours["upper_clothing"]["name"], "red")
        self.assertEqual(person.colours["lower_clothing"]["name"], "blue")

    def test_speed_units_gap_and_explicit_calibration(self):
        person = track(1, "person", [0, 0, 40, 100])
        self.assertIsNone(movement(person, {}, 640, 480)["speed_px_s"])
        detection = {
            "box": [10, 0, 50, 100],
            "centroid": (30, 50),
            "confidence": 0.9,
            "aspect_ratio": 2.5,
        }
        person.update(detection, 100.5)
        values = movement(person, {}, 640, 480)
        self.assertGreater(values["speed_px_s"], 0)
        self.assertIsNone(values["speed_m_s"])
        calibrated = movement(
            person,
            {"metres_per_pixel": 0.02, "reference_width": 640, "reference_height": 480},
            640,
            480,
        )
        self.assertAlmostEqual(calibrated["speed_m_s"], values["speed_px_s"] * 0.02, places=2)
        self.assertIsNone(
            movement(
                person,
                {"metres_per_pixel": 0.02, "reference_width": 640, "reference_height": 480},
                640,
                360,
            )["speed_m_s"]
        )
        person.update(detection, 105)
        self.assertIsNone(movement(person, {}, 640, 480)["speed_px_s"])

    def test_pose_results_expire_and_old_generation_is_discarded(self):
        analyzer = PoseAnalyzer(enabled=True)
        analyzer.last_submit = time.monotonic() + 100
        person = track(1, "person", [0, 0, 40, 100])
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        try:
            analyzer.generation = 1
            analyzer.cache = [
                {
                    "box": person.box,
                    "posture": "standing",
                    "confidence": 0.9,
                    "keypoints": joints().tolist(),
                }
            ]
            analyzer.cache_time = 100
            analyzer.history[1].extend(["standing", "standing"])
            analyzer.apply([person], frame, 100.2, 1)
            self.assertEqual(person.posture["label"], "standing")
            analyzer.apply([person], frame, 104, 1)
            self.assertEqual(person.posture["label"], "unknown")
            analyzer.apply([person], frame, 100.2, 2)
            self.assertFalse(analyzer.cache)
        finally:
            analyzer.close()


class NewWarningPolicyTests(unittest.TestCase):
    def test_person_and_animal_warn_red_immediately_but_bag_does_not(self):
        with patch.object(config, "ZONE_IMMEDIATE_WARNING", True):
            for label in ["person", "dog", "cat", "cow", "bird"]:
                engine = GeofenceAnalyzer()
                engine.add_zone("z", "Restricted", [[0, 0], [1, 0], [1, 1], [0, 1]])
                subject = track(1, label, [10, 10, 30, 60])
                events = engine.process_geofence([subject], 100, 100, 100)
                self.assertEqual(subject.threat_level, "HIGH", label)
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0]["metadata"]["required_dwell"], 0)
                self.assertEqual(events[0]["object_label"], label)
                self.assertFalse(engine.process_geofence([subject], 100, 100, 100.1))
                engine.process_geofence([], 100, 100, 100.2)
                self.assertFalse(engine.confirmed_zones)
            bag = track(2, "backpack", [10, 10, 30, 60])
            self.assertFalse(engine.process_geofence([bag], 100, 100, 100.3))

    def test_visible_knife_red_and_persistent_isolated_event(self):
        with (
            patch.object(config, "KNIFE_IMMEDIATE_WARNING", True),
            patch.object(config, "KNIFE_REQUIRE_PERSON", False),
        ):
            knife = track(2, "knife", [10, 10, 30, 60])
            engine = WeaponAnalyzer()
            self.assertFalse(engine.process([knife], 100))
            self.assertEqual(knife.threat_level, "HIGH")
            self.assertIn("DANGER", knife.active_alert)
            self.assertFalse(engine.process([knife], 100.5))
            candidate = engine.process([knife], 101.1)[0]
            event = AssessmentEngine().assess(candidate, [knife], "test", 101.1)
            self.assertIsNotNone(event)
            self.assertIn("DANGER", event["display_label"])
            self.assertFalse(event["person_ids"])


class CameraTests(unittest.TestCase):
    def test_coordinates_are_camera_scoped_and_gps_is_not_config_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cameras.json"
            registry = CameraRegistry(path)
            registry.update(
                "webcam:0",
                {
                    "name": "Door",
                    "location": {"latitude": 12.0, "longitude": 77.0, "source": "operator_entered"},
                },
            )
            self.assertIsNone(registry.get("webcam:1")["location"])
            registry.update(
                "phone:a",
                {"location": {"latitude": 1.0, "longitude": 2.0, "source": "camera_browser_gps"}},
                persist=False,
            )
            self.assertIsNotNone(registry.get("phone:a")["location"])
            self.assertIsNone(CameraRegistry(path).get("phone:a")["location"])

    def test_location_permissions_and_scale_validation(self):
        from app.ingestion.video_stream import video_manager

        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "app.web.routes.camera_registry", CameraRegistry(Path(directory) / "cameras.json")
            ),
            patch.object(config, "TRUST_LOCAL_DASHBOARD", False),
        ):
            client = TestClient(app)
            operator = {"Authorization": "Bearer " + config.ACCESS_TOKEN}
            camera = {"Authorization": "Bearer " + config.CAMERA_TOKEN}
            payload = {"name": "Camera", "location": {"latitude": 12.5, "longitude": 77.5}}
            self.assertEqual(
                client.put("/api/camera/settings", headers=camera, json=payload).status_code, 403
            )
            self.assertEqual(
                client.put("/api/camera/settings", headers=operator, json=payload).status_code, 200
            )
            payload["location"]["latitude"] = 91
            self.assertEqual(
                client.put("/api/camera/settings", headers=operator, json=payload).status_code, 422
            )
            payload = {"name": "Camera", "metres_per_pixel": 0.02}
            self.assertEqual(
                client.put("/api/camera/settings", headers=operator, json=payload).status_code, 422
            )
            with (
                patch.object(video_manager, "source_type", "phone_upload"),
                patch.object(video_manager, "camera_online", True),
                patch.object(video_manager, "camera_id", "phone:" + "a" * 32),
            ):
                self.assertEqual(
                    client.post(
                        "/api/camera/location", headers=camera, json={"latitude": 1, "longitude": 2}
                    ).status_code,
                    403,
                )
                client.cookies.set("vs_device", "a" * 32)
                self.assertEqual(
                    client.post(
                        "/api/camera/location", headers=camera, json={"latitude": 1, "longitude": 2}
                    ).status_code,
                    200,
                )
            self.assertEqual(
                client.patch(
                    "/api/person-annotations/missing", headers=camera, json={"label": "man"}
                ).status_code,
                403,
            )
            self.assertEqual(
                client.patch(
                    "/api/person-annotations/missing", headers=operator, json={"label": "man"}
                ).status_code,
                404,
            )


class SocketCleanupTests(unittest.IsolatedAsyncioTestCase):
    def socket(self):
        return SimpleNamespace(
            accept=AsyncMock(),
            receive=AsyncMock(return_value={"type": "websocket.disconnect"}),
            send_json=AsyncMock(),
            send_text=AsyncMock(),
            send_bytes=AsyncMock(),
            close=AsyncMock(),
        )

    async def test_idle_preview_exits_when_client_disconnects(self):
        from app.web.routes import preview_socket

        with patch("app.web.routes.authorized", return_value=True):
            await asyncio.wait_for(preview_socket(self.socket()), 0.5)

    async def test_idle_event_socket_releases_subscription(self):
        from app.web.routes import websocket_events
        from app.core.event_hub import event_hub

        before = len(event_hub._subscribers)
        with patch("app.web.routes.authorized", return_value=True):
            await asyncio.wait_for(websocket_events(self.socket()), 0.5)
        self.assertEqual(len(event_hub._subscribers), before)


if __name__ == "__main__":
    unittest.main()
