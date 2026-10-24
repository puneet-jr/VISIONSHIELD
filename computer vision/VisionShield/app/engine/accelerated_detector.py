"""YOLO11 ONNX inference on CPU (ONNX Runtime).
The exported graph uses the same learned weights. Preview delivery is independent.
"""

import json
from hashlib import sha256
from pathlib import Path
import cv2
import numpy as np


class AcceleratedYOLO:
    def __init__(self, source, config):
        source = Path(source)
        graph = source.with_suffix(".onnx")
        metadata = json.loads(graph.with_suffix(".onnx.json").read_text(encoding="utf-8"))
        if metadata["source_sha256"] != sha256(source.read_bytes()).hexdigest():
            raise ValueError(
                "Optimized graph belongs to different weights. Rerun scripts/optimize_model.py"
            )
        if metadata["graph_sha256"] != sha256(graph.read_bytes()).hexdigest():
            raise ValueError("Optimized graph checksum mismatch")
        self.names = {int(k): v for k, v in metadata["names"].items()}
        self.config = config
        self.name = "onnx-cpu"
        self.session = None
        if self.name == "onnx-cpu":
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = config.CPU_THREADS
            options.inter_op_num_threads = 1
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            self.session = ort.InferenceSession(
                str(graph), sess_options=options, providers=["CPUExecutionProvider"]
            )
            self.input = self.session.get_inputs()[0].name

    @staticmethod
    def prepare(frame, size):
        h, w = frame.shape[:2]
        ratio = min(size / h, size / w)
        nw, nh = round(w * ratio), round(h * ratio)
        dw, dh = (size - nw) % 32, (size - nh) % 32
        left, top = round(dw / 2 - 0.1), round(dh / 2 - 0.1)
        right, bottom = round(dw / 2 + 0.1), round(dh / 2 + 0.1)
        image = cv2.resize(frame, (nw, nh)) if (nw, nh) != (w, h) else frame
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        image = (
            np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
            / 255.0
        )
        return image, ratio, (left, top)

    def detect(self, frame, class_ids):
        image, ratio, offset = self.prepare(frame, self.config.INPUT_SIZE)
        prediction = self.session.run(None, {self.input: image})[0]
        return self.decode(prediction, frame.shape[:2], ratio, offset, class_ids)

    def decode(self, prediction, shape, ratio, offset, class_ids):
        rows = prediction[0].T
        classes = rows[:, 4:].argmax(axis=1)
        scores = rows[np.arange(len(rows)), 4 + classes]
        mask = (scores >= self.config.CONFIDENCE_THRESHOLD) & np.isin(classes, class_ids)
        boxes = rows[mask, :4].copy()
        scores = scores[mask]
        classes = classes[mask]
        if not len(boxes):
            return []
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 0] = (boxes[:, 0] - offset[0]) / ratio
        boxes[:, 1] = (boxes[:, 1] - offset[1]) / ratio
        boxes[:, 2:] /= ratio
        indices = cv2.dnn.NMSBoxesBatched(
            boxes.tolist(),
            scores.tolist(),
            classes.tolist(),
            self.config.CONFIDENCE_THRESHOLD,
            self.config.IOU_THRESHOLD,
        )
        detections = []
        h, w = shape
        for index in np.asarray(indices, dtype=int).reshape(-1)[:40]:
            x, y, bw, bh = map(float, boxes[index])
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(w), x + bw), min(float(h), y + bh)
            if x2 <= x1 or y2 <= y1:
                continue
            cid = int(classes[index])
            bw, bh = x2 - x1, y2 - y1
            detections.append(
                {
                    "box": [x1, y1, x2, y2],
                    "confidence": round(float(scores[index]), 3),
                    "class_id": cid,
                    "label": self.names[cid].lower(),
                    "centroid": ((x1 + x2) / 2, (y1 + y2) / 2),
                    "width": bw,
                    "height": bh,
                    "aspect_ratio": bh / bw,
                }
            )
        return detections
