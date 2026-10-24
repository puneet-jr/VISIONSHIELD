"""Visible crop colours only. No demographic inference or clothing semantics."""

import cv2
import numpy as np


NAMES = [
    "black",
    "white",
    "grey",
    "red",
    "orange",
    "yellow",
    "green",
    "cyan",
    "blue",
    "purple",
    "pink",
    "brown",
]


def crop_colour(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = max(0, int(x1)), min(w, int(x2))
    y1, y2 = max(0, int(y1)), min(h, int(y2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return {"name": "unknown", "coverage": 0.0}
    hsv = cv2.cvtColor(cv2.resize(frame[y1:y2, x1:x2], (32, 32)), cv2.COLOR_BGR2HSV)
    hue, saturation, value = [hsv[:, :, i] for i in range(3)]
    colours = np.full(hue.shape, 3, dtype=np.uint8)
    colours[(hue >= 10) & (hue < 23)] = 4
    colours[(hue >= 23) & (hue < 35)] = 5
    colours[(hue >= 35) & (hue < 85)] = 6
    colours[(hue >= 85) & (hue < 100)] = 7
    colours[(hue >= 100) & (hue < 130)] = 8
    colours[(hue >= 130) & (hue < 160)] = 9
    colours[(hue >= 160) & (hue < 175)] = 10
    colours[(hue >= 5) & (hue < 25) & (value < 160) & (saturation >= 55)] = 11
    colours[saturation < 45] = 2
    colours[(saturation < 45) & (value >= 200)] = 1
    colours[value < 45] = 0
    counts = np.bincount(colours.ravel(), minlength=len(NAMES))
    index = int(counts.argmax())
    coverage = float(counts[index] / colours.size)
    return {
        "name": NAMES[index] if coverage >= 0.30 else "mixed",
        "coverage": round(coverage, 2),
        "method": "dominant colour in bounding-box crop; lighting/background can change results",
    }


def describe_colours(frame, track):
    x1, y1, x2, y2 = track.box
    bw = x2 - x1
    bh = y2 - y1
    if track.label == "person":
        track.colours = {
            "upper_clothing": crop_colour(
                frame, [x1 + 0.2 * bw, y1 + 0.22 * bh, x2 - 0.2 * bw, y1 + 0.52 * bh]
            ),
            "lower_clothing": crop_colour(
                frame, [x1 + 0.2 * bw, y1 + 0.55 * bh, x2 - 0.2 * bw, y1 + 0.85 * bh]
            ),
        }
    else:
        track.colours = {
            "object": crop_colour(
                frame, [x1 + 0.15 * bw, y1 + 0.15 * bh, x2 - 0.15 * bw, y2 - 0.15 * bh]
            )
        }
