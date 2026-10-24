import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
from app.engine.crowd import CrowdDynamicsAnalyzer
from app.engine.geofence import GeofenceAnalyzer, GeofenceZone
from app.engine.tracker import TrackedObject
from app.engine.assessment import AssessmentEngine
from app.web.routes import ZonePayload
from app.core.streams import StreamRegistry
from app.config import config


def person(tid, x=20, label="person"):
    return TrackedObject(
        tid,
        {
            "label": label,
            "class_id": 0,
            "box": [x - 5, 20, x + 5, 50],
            "centroid": (x, 35),
            "confidence": 0.9,
            "aspect_ratio": 3,
        },
        100,
    )


def zone(**policy):
    return GeofenceZone(
        "entry", "Entry", [[0, 0], [1, 0], [1, 1], [0, 1]], purpose="crowd_monitor", **policy
    )


class CrowdRules(unittest.TestCase):
    def test_pending_vehicle_does_not_hide_confirmed_capacity(self):
        engine = CrowdDynamicsAnalyzer()
        z = zone(max_capacity=1)
        z.purpose = "pedestrian_only"
        for timestamp in [100, 101, 102]:
            self.process(engine, [person(1)], z, timestamp)
        self.process(engine, [person(1), person(2, 60, label="car")], z, 103)
        self.assertEqual(engine.zone_metrics[0]["state"], "critical")
        self.assertTrue(engine.zone_metrics[0]["warning_confirmed"])
        self.assertIn("CAPACITY", engine.zone_metrics[0]["active_warning"])

    def process(self, engine, tracks, z, t):
        return engine.process(tracks, {"entry": z}, 100, 100, t)

    def test_capacity_exact_threshold_persistence_clear(self):
        engine = CrowdDynamicsAnalyzer()
        people = [person(1), person(2, 60)]
        z = zone(max_capacity=2)
        self.assertEqual(self.process(engine, people, z, 100), [])
        self.assertEqual(self.process(engine, people, z, 101), [])
        events = self.process(engine, people, z, 102)
        self.assertEqual(events[0]["alert_type"], "CROWD_CAPACITY_EXCEEDED")
        result = AssessmentEngine().assess(events[0], people, "camera", 102)
        self.assertEqual(result["severity"], "HIGH")
        self.assertIn("undercount", str(result["assessment"]["limitations"]))
        self.process(engine, [], z, 103)
        self.assertEqual(engine.pending, {})
        self.assertNotIn("active_warning", engine.zone_metrics[0])

    def test_gap_does_not_confirm(self):
        engine = CrowdDynamicsAnalyzer()
        z = zone(max_capacity=1)
        for t in [100, 101, 110, 111]:
            self.assertEqual(self.process(engine, [person(1)], z, t), [])

    def test_uncalibrated_distance_stays_pixels(self):
        engine = CrowdDynamicsAnalyzer()
        self.process(engine, [person(1), person(2, 60)], zone(), 100)
        metric = engine.zone_metrics[0]
        self.assertEqual(metric["mean_nearest_distance_px"], 40)
        self.assertIsNone(metric["mean_nearest_distance_m"])
        self.assertIsNone(metric["observed_people_per_m2"])

    def test_area_only_density_not_distance(self):
        engine = CrowdDynamicsAnalyzer()
        z = zone(area_m2=2, density_threshold=0.8)
        for t in [100, 101, 102]:
            events = self.process(engine, [person(1), person(2, 60)], z, t)
        self.assertEqual(engine.zone_metrics[0]["observed_people_per_m2"], 1)
        self.assertIsNone(engine.zone_metrics[0]["mean_nearest_distance_m"])
        self.assertEqual(events[0]["alert_type"], "CROWD_DENSITY_CRITICAL")

    def test_homography_ground_spacing(self):
        engine = CrowdDynamicsAnalyzer()
        z = zone(ground_width_m=10, ground_height_m=5)
        self.process(engine, [person(1), person(2, 60)], z, 100)
        self.assertAlmostEqual(engine.zone_metrics[0]["mean_nearest_distance_m"], 4)
        self.assertEqual(engine.zone_metrics[0]["observed_people_per_m2"], 0.04)

    def test_vehicle_three_seconds_person_not_intrusion(self):
        engine = CrowdDynamicsAnalyzer()
        z = zone()
        z.purpose = "pedestrian_only"
        tracks = [person(1, label="car"), person(2, 60)]
        for t in [100, 101, 102]:
            self.assertEqual(self.process(engine, tracks, z, t), [])
        events = self.process(engine, tracks, z, 103)
        self.assertEqual([e["alert_type"] for e in events], ["VEHICLE_INTRUSION"])
        self.assertEqual(events[0]["object_label"], "car")

    def test_buildup_full_window_and_minimum_growth(self):
        engine = CrowdDynamicsAnalyzer()
        z = zone()
        for t in range(100, 110):
            self.assertEqual(
                self.process(engine, [person(i, i * 8 + 5) for i in range(4)], z, t), []
            )
        for t in [110, 111]:
            self.assertEqual(
                self.process(engine, [person(i, i * 8 + 5) for i in range(9)], z, t), []
            )
        events = self.process(engine, [person(i, i * 8 + 5) for i in range(9)], z, 112)
        self.assertEqual(events[0]["alert_type"], "CROWD_BUILDUP_WARNING")
        self.assertTrue(events[0]["metadata"]["not_a_stampede_prediction"])

    def test_motion_alone_does_not_claim_stampede(self):
        tracks = [person(i, 15 + 20 * i) for i in range(3)]
        for track in tracks:
            track.velocity_valid = True
            track.motion = np.array([50, 0])
        engine = CrowdDynamicsAnalyzer()
        for t in range(100, 115):
            self.assertEqual(self.process(engine, tracks, zone(), t), [])
        self.assertEqual(engine.zone_metrics[0]["flow_coherence"], 1)

    def test_zone_policy_validation_and_persistence(self):
        base = {
            "id": "entry",
            "name": "Entry",
            "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "purpose": "river_edge",
            "max_capacity": 12,
        }
        for extra in [
            {"density_threshold": 2},
            {"minimum_spacing_m": 1},
            {"ground_width_m": 2},
            {"area_m2": float("nan")},
        ]:
            with self.assertRaises(ValueError):
                ZonePayload.model_validate(base | extra)
        values = ZonePayload.model_validate(
            base | {"area_m2": 6, "density_threshold": 2}
        ).model_dump()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "zones.json"
            engine = GeofenceAnalyzer(path)
            engine.replace_zones([values])
            loaded = GeofenceAnalyzer(path).get_zones_dict()[0]
            self.assertEqual(loaded["max_capacity"], 12)
            self.assertEqual(loaded["area_m2"], 6)


class StreamIsolation(unittest.TestCase):
    def test_two_devices_separate_tracks_buffers_gps_and_zones(self):
        from app.main import app
        from app.core.streams import streams
        import cv2

        device1 = "a" * 32
        device2 = "b" * 32
        client = TestClient(app)
        headers = {"Authorization": "Bearer " + config.CAMERA_TOKEN}
        # Cookie passed as a handshake header to avoid a shared test browser identity.
        with client.websocket_connect(
            "/ws/camera", headers=headers | {"cookie": "vs_device=" + device1}
        ) as first:
            frame1 = cv2.imencode(".jpg", np.full((64, 64, 3), 20, np.uint8))[1].tobytes()
            first.send_bytes(frame1)
            self.assertTrue(first.receive_json()["ok"])
            with client.websocket_connect(
                "/ws/camera", headers=headers | {"cookie": "vs_device=" + device2}
            ) as second:
                frame2 = cv2.imencode(".jpg", np.full((64, 64, 3), 220, np.uint8))[1].tobytes()
                second.send_bytes(frame2)
                self.assertTrue(second.receive_json()["ok"])
                a = streams.for_device(device1)
                b = streams.for_device(device2)
                self.assertIsNot(a.video, b.video)
                self.assertIsNot(a.pipeline.tracker, b.pipeline.tracker)
                self.assertLess(a.video.latest_frame.mean(), 30)
                self.assertGreater(b.video.latest_frame.mean(), 210)
                self.assertEqual(a.video.camera_id, "phone:" + device1)
                self.assertEqual(b.video.camera_id, "phone:" + device2)
                with client.websocket_connect(
                    "/ws/camera", headers=headers | {"cookie": "vs_device=" + device1}
                ) as duplicate:
                    from starlette.websockets import WebSocketDisconnect

                    with self.assertRaises(WebSocketDisconnect):
                        duplicate.receive_json()
                gps = client.post(
                    "/api/camera/location",
                    headers=headers | {"cookie": "vs_device=" + device2},
                    json={"latitude": 12, "longitude": 77},
                )
                self.assertEqual(gps.status_code, 200)
                from app.core.cameras import camera_registry

                self.assertEqual(camera_registry.get(b.video.camera_id)["location"]["latitude"], 12)
        self.assertFalse(a.busy)
        self.assertFalse(b.busy)
        for context in streams.all():
            if context.key != "main":
                streams._close(context)
                streams.contexts.pop(context.key)
        a.video.stop()
        a.device = ""

    def test_stream_limit(self):
        registry = StreamRegistry()
        with (
            patch.object(config, "MAX_CAMERA_STREAMS", 1),
            patch.object(registry.default.video, "source_type", "webcam"),
        ):
            with self.assertRaises(ValueError):
                registry.claim("c" * 32)

    def test_guard_can_monitor_but_cannot_administer(self):
        from app.main import app

        client = TestClient(app)
        headers = {"Authorization": "Bearer " + config.GUARD_TOKEN}
        self.assertEqual(client.get("/api/streams", headers=headers).status_code, 200)
        self.assertFalse(client.get("/api/streams", headers=headers).json()["can_configure"])
        for method, path in [
            ("post", "/api/source"),
            ("post", "/api/streams/main/zones"),
            ("delete", "/api/events/missing"),
        ]:
            self.assertEqual(getattr(client, method)(path, headers=headers).status_code, 403)
        self.assertEqual(
            client.patch(
                "/api/events/missing", headers=headers, json={"status": "false_positive"}
            ).status_code,
            403,
        )
        self.assertEqual(
            client.patch(
                "/api/events/missing", headers=headers, json={"status": "acknowledged"}
            ).status_code,
            404,
        )
        self.assertEqual(
            client.post("/api/pair", json={"token": config.GUARD_TOKEN}).json()["role"], "guard"
        )


if __name__ == "__main__":
    unittest.main()
