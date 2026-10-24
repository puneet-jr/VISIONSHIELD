"""YOLOv11 Detection Engine
Lightweight, thread-safe inference optimized for CPU execution.
Automatically loads or downloads the 5MB nano model on first invocation.
"""

import numpy as np
import logging
from typing import List, Dict, Any
from app.config import config

logger = logging.getLogger("VisionShield.Detector")


class ObjectDetector:
    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or config.YOLO_MODEL
        self.device = device or config.DEVICE
        self.model = None
        self.error = None
        self.backend = "torch"
        self.class_ids = []
        self.enabled_labels = []

    def _load_model(self):
        """Lazy load YOLO model using ultralytics."""
        try:
            from pathlib import Path

            if (
                config.INFERENCE_BACKEND != "torch"
                and Path(self.model_name).with_suffix(".onnx.json").exists()
            ):
                try:
                    from app.engine.accelerated_detector import AcceleratedYOLO

                    self.model = AcceleratedYOLO(self.model_name, config)
                    self.backend = self.model.name
                    wanted = {
                        "person",
                        "backpack",
                        "handbag",
                        "suitcase",
                        "knife",
                        *config.ANIMAL_CLASSES,
                        *config.VEHICLE_CLASSES,
                    }
                    self.class_ids = [
                        i
                        for i, name in self.model.names.items()
                        if config.DETECT_ALL_CLASSES or name.lower() in wanted
                    ]
                    self.enabled_labels = [self.model.names[i] for i in self.class_ids]
                    logger.info("Detector backend ready: %s", self.backend)
                    return
                except Exception as exc:
                    logger.warning("Acceleration unavailable: %s; using PyTorch", exc)
            from ultralytics import YOLO
            import torch

            torch.set_num_threads(config.CPU_THREADS)
            logger.info(f"Loading YOLO model '{self.model_name}' on device '{self.device}'...")
            self.model = YOLO(self.model_name)
            wanted = {
                "person",
                "backpack",
                "handbag",
                "suitcase",
                "knife",
                *config.ANIMAL_CLASSES,
                *config.VEHICLE_CLASSES,
            }
            self.class_ids = [
                i
                for i, name in self.model.names.items()
                if config.DETECT_ALL_CLASSES or name.lower() in wanted
            ]
            self.enabled_labels = [self.model.names[i] for i in self.class_ids]
            if not self.class_ids:
                raise ValueError("Model has none of the required person/bag/knife labels")
            logger.info("YOLO model successfully initialized.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
            self.error = str(e)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run object detection on an input BGR frame.

        Returns a list of dicts:
        [{
            "box": [x1, y1, x2, y2],
            "confidence": float,
            "class_id": int,
            "label": str,
            "centroid": (cx, cy),
            "width": w,
            "height": h,
            "aspect_ratio": h / w
        }]
        """
        if self.model is None and self.error is None:
            self._load_model()
        if self.model is None or frame is None:
            return []

        try:
            if self.backend != "torch":
                result = self.model.detect(frame, self.class_ids)
                self.error = None
                return result
            results = self.model.predict(
                source=frame,
                conf=config.CONFIDENCE_THRESHOLD,
                iou=config.IOU_THRESHOLD,
                imgsz=config.INPUT_SIZE,
                device=self.device,
                verbose=False,
                classes=self.class_ids,
                max_det=config.MAX_DETECTIONS,
            )

            self.error = None
            detections = []
            if not results:
                return detections

            result = results[0]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                return detections

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            names = result.names

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                confidence = float(confs[i])
                class_id = int(classes[i])
                label = names.get(class_id, f"class_{class_id}").lower()

                w = float(x2 - x1)
                h = float(y2 - y1)
                if w <= 0 or h <= 0:
                    continue

                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)
                aspect_ratio = float(h / w)

                detections.append(
                    {
                        "box": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": round(confidence, 3),
                        "class_id": class_id,
                        "label": label,
                        "centroid": (cx, cy),
                        "width": w,
                        "height": h,
                        "aspect_ratio": aspect_ratio,
                    }
                )

            return detections

        except Exception as e:
            self.error = str(e)
            logger.error(f"Detection inference error: {e}")
            return []
