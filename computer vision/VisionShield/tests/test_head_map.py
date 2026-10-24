"""Head mapping uses real model outputs, coordinate transforms and explicit opt-in."""

import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from app.config import config
from app.engine.head_map import HeadMapDetector


class HeadMapTests(unittest.TestCase):
    def test_disabled_by_default_after_failed_count_benchmark(self):
        self.assertFalse(config.ENABLE_HEAD_MAP)

    def test_head_centres_come_from_model_boxes(self):
        detector = HeadMapDetector()
        detector.model = SimpleNamespace(
            detect=lambda frame, classes: [{"box": [10, 20, 30, 40], "confidence": 0.8}]
        )
        with patch.object(config, "HEAD_MAP_TILED", False):
            result = detector.detect(np.zeros((80, 100, 3), dtype=np.uint8))
        self.assertEqual(result["points"], [[20, 30, 0.8]])
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["method"], "learned_head_detector")

    def test_failed_model_does_not_fabricate_points(self):
        detector = HeadMapDetector()

        def fail(*args):
            raise ValueError("fixture failure")

        detector.model = SimpleNamespace(detect=fail)
        self.assertIsNone(detector.detect(np.zeros((40, 40, 3), dtype=np.uint8)))
        self.assertIn("fixture failure", detector.error)

    def test_overlapping_tiles_have_disjoint_centre_ownership(self):
        detector = HeadMapDetector()
        calls = []

        def detect(frame, classes):
            calls.append(frame.shape)
            cx = 50 - int(frame[0, 0, 0])
            cy = 50 - int(frame[0, 0, 1])
            return [{"box": [cx - 5, cy - 5, cx + 5, cy + 5], "confidence": 0.9}]

        detector.model = SimpleNamespace(detect=detect)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :, 0] = np.arange(100)[None, :]
        image[:, :, 1] = np.arange(100)[:, None]
        with patch.object(config, "HEAD_MAP_TILED", True):
            result = detector.detect(image)
        self.assertEqual(len(calls), 4)
        # The same real image point appears in all four overlapping crops.
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["points"], [[50, 50, 0.9]])


if __name__ == "__main__":
    unittest.main()
