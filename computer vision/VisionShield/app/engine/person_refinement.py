"""Learned person proposals from a high-resolution full-frame pass plus enlarged overlapping tiles.
Low-confidence proposals recover occluded/distant people; results are deduplicated globally.
This refines visible body detections, not head annotations or unique attendance.
"""
from types import SimpleNamespace
import logging
import time
from app.config import ROOT, config
from app.engine.accelerated_detector import AcceleratedYOLO
import cv2
import numpy as np


def merge_people(detections, iou=.5):
    people = [d for d in detections if d['label'] == 'person']
    other = [d for d in detections if d['label'] != 'person']
    if not people:
        return other
    boxes = [[d['box'][0], d['box'][1], d['box'][2] - d['box'][0], d['box'][3] - d['box'][1]] for d in people]
    indices = cv2.dnn.NMSBoxes(boxes, [float(d['confidence']) for d in people], 0., iou)
    return other + [people[int(index)] for index in np.asarray(indices, dtype=int).reshape(-1)]


class PersonRefiner:
    def __init__(self):
        self.model = None
        self.error = None
        self.cache = {}

    def _load(self):
        options = SimpleNamespace(CPU_THREADS=config.CPU_THREADS, INPUT_SIZE=config.PERSON_FULL_INPUT_SIZE,
            CONFIDENCE_THRESHOLD=config.PERSON_REFINE_CONFIDENCE, IOU_THRESHOLD=.5, MAX_DETECTIONS=500)
        self.model = AcceleratedYOLO(ROOT / config.PERSON_REFINE_MODEL, options)

    def _detect(self, image, size):
        self.model.config.INPUT_SIZE = size
        return self.model.detect(image, [0])

    def _proposals(self, frame):
        h, w = frame.shape[:2]
        proposals = []
        for detection in self._detect(frame, config.PERSON_FULL_INPUT_SIZE):
            proposals.append({**detection, 'detection_source': 'full_frame_high_resolution'})
        for row in range(2):
            for col in range(2):
                x0 = max(0, round((col * .5 - .12) * w)); x1 = min(w, round(((col + 1) * .5 + .12) * w))
                y0 = max(0, round((row * .5 - .12) * h)); y1 = min(h, round(((row + 1) * .5 + .12) * h))
                for detection in self._detect(frame[y0:y1, x0:x1], config.PERSON_TILE_INPUT_SIZE):
                    x, y, x2, y2 = detection['box']
                    # Internal top cuts can produce a second lower-body box for
                    # someone already detected above. Keep the complete upper body.
                    if y0 and y < 2:
                        continue
                    anchor_x = (x + x2) / 2 + x0; anchor_y = y + .12 * (y2 - y) + y0
                    if not (col * w / 2 <= anchor_x < (col + 1) * w / 2 and row * h / 2 <= anchor_y < (row + 1) * h / 2):
                        continue
                    proposals.append({**detection, 'box': [x + x0, y + y0, x2 + x0, y2 + y0],
                        'centroid': ((x + x2) / 2 + x0, (y + y2) / 2 + y0), 'detection_source': 'enlarged_person_tile'})
        # Reject slivers and implausible shapes that low confidence would otherwise admit.
        return [{**p, 'label': 'person'} for p in proposals
            if .4 <= (p['box'][3] - p['box'][1]) / max(1e-6, p['box'][2] - p['box'][0]) <= 6]

    def refine(self, frame, base, timestamp=None, key=None):
        """Run the expensive pass at most every PERSON_REFINE_INTERVAL_SECONDS per camera `key`.
        Between passes, recent proposals are merged with the fresh base detections."""
        if self.model is None:
            try:
                self._load()
            except Exception as exc:
                self.error = str(exc)
                logging.getLogger(__name__).warning('Person refinement unavailable: %s', exc)
                return base
        now = timestamp if timestamp is not None else time.time()
        state = self.cache.get(key)
        if state and (state['shape'] != frame.shape[:2] or not 0 <= now - state['at'] <= config.PERSON_CACHE_SECONDS):
            state = None
        if state is None or now - state['at'] >= config.PERSON_REFINE_INTERVAL_SECONDS:
            state = {'shape': frame.shape[:2], 'at': now, 'people': self._proposals(frame)}
            self.cache[key] = state
        self.error = None
        # Fresh base boxes win ties through confidence; stale proposals only fill gaps.
        return merge_people(list(base) + state['people'], .45)


person_refiner = PersonRefiner()
