import time
import threading
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from app.engine.tracker import ObjectTracker
from app.ingestion.video_stream import VideoStreamManager


def detection(x, label="person"):
    return {
        "label": label,
        "class_id": 0,
        "box": [x, 10, x + 40, 110],
        "centroid": (x + 20, 60),
        "confidence": 0.9,
        "aspect_ratio": 2.5,
    }


def scene(*people):
    frame = np.zeros((160, 640, 3), dtype=np.uint8)
    for x, colour in people:
        frame[10:110, x : x + 40] = colour
    return frame


class TrackingTests(unittest.TestCase):
    def test_same_id_after_missed_detection(self):
        tracker = ObjectTracker()
        tracker.update([detection(20)], 100, scene((20, (0, 0, 220))))
        tracker.update([], 100.3, scene())
        result = tracker.update([detection(30)], 100.8, scene((30, (0, 0, 220))))
        self.assertEqual([t.track_id for t in result if t.disappeared_count == 0], [1])

    def test_returning_person_keeps_id_after_short_exit(self):
        tracker = ObjectTracker()
        tracker.update([detection(20)], 100, scene((20, (0, 0, 220))))
        tracker.update([], 101, scene())
        result = tracker.update([detection(400)], 108, scene((400, (0, 0, 220))))
        active = [t for t in result if t.disappeared_count == 0]
        self.assertEqual(active[0].track_id, 1)
        self.assertTrue(active[0].reacquired)

    def test_different_clothing_does_not_reuse_lost_id(self):
        tracker = ObjectTracker()
        tracker.update([detection(20)], 100, scene((20, (0, 0, 220))))
        tracker.update([], 101, scene())
        result = tracker.update([detection(400)], 108, scene((400, (220, 0, 0))))
        self.assertEqual([t.track_id for t in result if t.disappeared_count == 0], [2])

    def test_appearance_prevents_switch_when_people_cross(self):
        tracker = ObjectTracker()
        tracker.update(
            [detection(20), detection(200)],
            100,
            scene((20, (0, 0, 220)), (200, (220, 0, 0))),
        )
        result = tracker.update(
            [detection(20), detection(200)],
            100.5,
            scene((20, (220, 0, 0)), (200, (0, 0, 220))),
        )
        by_id = {t.track_id: t for t in result if t.disappeared_count == 0}
        self.assertEqual(by_id[1].box[0], 200)
        self.assertEqual(by_id[2].box[0], 20)

    def test_expired_person_gets_new_id(self):
        tracker = ObjectTracker(retention_seconds=5)
        tracker.update([detection(20)], 100, scene((20, (0, 0, 220))))
        result = tracker.update([detection(20)], 106, scene((20, (0, 0, 220))))
        self.assertEqual(result[0].track_id, 2)

    def test_many_fast_misses_do_not_expire_person_by_frame_count(self):
        tracker = ObjectTracker()
        tracker.update([detection(20)], 100, scene((20, (0, 0, 220))))
        for i in range(40):
            tracker.update([], 100 + (i + 1) * 0.01, scene())
        result = tracker.update([detection(25)], 100.5, scene((25, (0, 0, 220))))
        self.assertEqual([t.track_id for t in result if t.disappeared_count == 0], [1])

    def test_ambiguous_return_does_not_claim_verified_identity(self):
        tracker = ObjectTracker()
        tracker.update(
            [detection(20), detection(200)],
            100,
            scene((20, (0, 0, 220)), (200, (0, 0, 220))),
        )
        tracker.update([], 101, scene())
        result = tracker.update([detection(500)], 108, scene((500, (0, 0, 220))))
        self.assertEqual([t.track_id for t in result if t.disappeared_count == 0], [3])

    def test_pool_is_bounded(self):
        tracker = ObjectTracker(max_tracks=3)
        for i in range(8):
            tracker.update([detection(i * 50, label=str(i))], 100 + i * 0.1)
        self.assertLessEqual(len(tracker.tracks), 3)


class PreviewTests(unittest.TestCase):
    def test_preview_advances_without_inference(self):
        manager = VideoStreamManager()
        try:
            manager.start("phone_upload")
            for i in range(8):
                manager.push_phone_frame(np.full((120, 160, 3), i * 20, dtype=np.uint8))
                time.sleep(0.055)
            self.assertGreaterEqual(manager.jpeg_id, 5)
            image = cv2.imdecode(np.frombuffer(manager.jpeg, np.uint8), cv2.IMREAD_COLOR)
            self.assertGreater(float(image.mean()), 100)
        finally:
            manager.stop()

    def test_slow_camera_open_does_not_block_start(self):
        manager = VideoStreamManager()
        gate = threading.Event()
        from unittest.mock import MagicMock

        cap = MagicMock()
        cap.isOpened.return_value = False

        def open_camera(*args):
            gate.wait(1)
            return cap

        try:
            with patch("app.ingestion.video_stream.cv2.VideoCapture", side_effect=open_camera):
                started = time.perf_counter()
                manager.start("0")
                self.assertLess(time.perf_counter() - started, 0.3)
                gate.set()
        finally:
            gate.set()
            manager.stop()

    def test_late_phone_frame_cannot_overwrite_new_source(self):
        manager = VideoStreamManager()
        try:
            manager.start("phone_upload")
            old = manager.generation
            manager.start("phone_upload")
            self.assertFalse(
                manager.push_phone_frame(
                    np.zeros((120, 160, 3), dtype=np.uint8), expected_generation=old
                )
            )
            self.assertIsNone(manager.latest_frame)
        finally:
            manager.stop()

    def test_source_switch_clears_preview_and_generation(self):
        manager = VideoStreamManager()
        try:
            manager.start("phone_upload")
            manager.push_phone_frame(np.zeros((120, 160, 3), dtype=np.uint8))
            time.sleep(0.1)
            generation = manager.generation
            manager.start("phone_upload")
            self.assertGreater(manager.generation, generation)
            self.assertIsNone(manager.jpeg)
        finally:
            manager.stop()


if __name__ == "__main__":
    unittest.main()
