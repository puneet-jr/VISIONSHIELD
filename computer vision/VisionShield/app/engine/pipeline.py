"""VisionShield Master Computer Vision Processing Pipeline
Coordinates Detection, Stateful Tracking, Spatio-Temporal Association,
Kinematics, Dynamic Geofencing, HUD Annotation, and Real-Time Event Dispatch.
"""

import cv2
import base64
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import logging
from typing import Dict

from app.config import config
from app.engine.detector import ObjectDetector
from app.engine.tracker import ObjectTracker
from app.engine.spatio_temporal import SpatioTemporalAnalyzer
from app.engine.kinematics import KinematicsAnalyzer
from app.engine.geofence import GeofenceAnalyzer
from app.engine.weapon import WeaponAnalyzer
from app.engine.identity import IdentityManager
from app.engine.assessment import AssessmentEngine
from app.engine.incident_agent import incident_dispatcher
from app.ingestion.video_stream import video_manager
from app.core.event_hub import event_hub
from app.db.database import db

logger = logging.getLogger("VisionShield.Pipeline")


class SurveillancePipeline:
    def __init__(self, persist=True):
        self.persist = persist
        self.identities = IdentityManager()
        self.assessment = AssessmentEngine()
        self.stage_times = {}
        self.processed_frames = 0
        self.skipped_frames = 0
        self.detector = ObjectDetector()
        self.tracker = ObjectTracker(retention_seconds=config.TRACK_RETENTION_SECONDS)
        self.spatio_temporal = SpatioTemporalAnalyzer()
        self.kinematics = KinematicsAnalyzer()
        self.geofence = GeofenceAnalyzer(config.ZONES_PATH)
        self.weapon = WeaponAnalyzer()

        self.is_running = False
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="incident")
        self.report_future = None
        self.event_times = {}
        self.last_frame_id = -1
        self.generation = -1
        self.inference_ms = 0.0
        self.processing_fps = 0.0
        self.last_processed_at = None
        self.snapshot = {}
        self.last_detection_counts = {}
        self.last_incident_time: Dict[str, float] = {}  # alert_type -> cooldown
        self.incident_cooldown = (
            12.0  # seconds between triggering heavy VLM incident reports per event type
        )

    def run_cycle(self):
        """Single processing step called continuously by the surveillance service loop."""
        with video_manager.lock:
            frame_id = video_manager.frame_id
            generation = video_manager.generation
            captured_at = video_manager.last_frame_time
            if frame_id == self.last_frame_id:
                return
            frame = (
                video_manager.latest_frame.copy()
                if video_manager.latest_frame is not None
                else None
            )
        previous_frame_id = self.last_frame_id
        self.last_frame_id = frame_id
        if generation != self.generation:
            self.processed_frames = 0
            self.skipped_frames = 0
            self.assessment = AssessmentEngine()
            self.tracker = ObjectTracker(retention_seconds=config.TRACK_RETENTION_SECONDS)
            self.spatio_temporal = SpatioTemporalAnalyzer()
            self.kinematics = KinematicsAnalyzer()
            self.weapon = WeaponAnalyzer()
            self.event_times.clear()
            self.last_incident_time.clear()
            self.geofence.select_source(video_manager.source_path)
            self.generation = generation
        if frame is None:
            return

        timestamp = captured_at or time.time()
        h, w = frame.shape[:2]

        # 1. Object Detection
        started = time.perf_counter()
        detections = self.detector.detect(frame) if video_manager.camera_online else []
        self.inference_ms = (time.perf_counter() - started) * 1000

        if generation != video_manager.generation:
            return

        stage = time.perf_counter()
        # 2. Stateful Tracking
        tracks = [
            t
            for t in self.tracker.update(detections, timestamp, frame=frame)
            if t.disappeared_count == 0
        ]

        camera_id = video_manager.camera_id
        self.identities.associate(tracks, camera_id, generation, timestamp)
        pending_segments = self.identities.drain_history()
        if self.persist:
            db.record_identity_segments(pending_segments)
        self.stage_times["tracking_identity_ms"] = round((time.perf_counter() - stage) * 1000, 2)
        stage = time.perf_counter()
        # Recompute live severity; historical incidents stay in the timeline, not on clear tracks.
        for track in tracks:
            track.threat_score = 0
            track.threat_level = "LOW"
            track.active_alert = None

        # 3. Behavioral Analysis Modules
        st_events = self.spatio_temporal.process_associations(
            tracks, timestamp, self.geofence.zones, (w, h)
        )
        km_events = self.kinematics.process_kinematics(tracks, timestamp)
        gf_events = self.geofence.process_geofence(tracks, w, h, timestamp)

        weapon_events = self.weapon.process(tracks, timestamp)
        all_events = st_events + km_events + gf_events + weapon_events

        self.stage_times["association_temporal_ms"] = round((time.perf_counter() - stage) * 1000, 2)
        stage = time.perf_counter()
        for candidate in all_events:
            event = self.assessment.assess(candidate, tracks, camera_id, timestamp)
            if event is None:
                continue
            evidence = None
            if config.STORE_EVENT_IMAGES:
                scale = min(1.0, 640 / max(h, w))
                thumb = cv2.resize(frame, (round(w * scale), round(h * scale)))
                ok, image = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 65])
                if ok:
                    evidence = base64.b64encode(image).decode("ascii")
            if self.persist:
                db.insert_event(event, evidence)
            event_hub.publish({"type": "SECURITY_EVENT", "data": event})
            if (
                event["severity"] == "HIGH"
                and config.ENABLE_AI_DOSSIER
                and (self.report_future is None or self.report_future.done())
            ):
                self.report_future = self.executor.submit(
                    self._generate_report, event, frame.copy()
                )
        self.stage_times["assessment_persistence_ms"] = round(
            (time.perf_counter() - stage) * 1000, 2
        )
        self.stage_times["detection_ms"] = round(self.inference_ms, 2)
        self.processed_frames += 1
        if previous_frame_id >= 0 and previous_frame_id < frame_id:
            self.skipped_frames += max(0, frame_id - previous_frame_id - 1)

        # Publish small metadata independently of the camera preview.
        now = time.perf_counter()
        if self.last_processed_at:
            self.processing_fps = 1 / max(now - self.last_processed_at, 0.001)
        self.last_processed_at = now
        counts = {}
        for detection in detections:
            counts[detection["label"]] = counts.get(detection["label"], 0) + 1
        self.last_detection_counts = counts
        if generation == video_manager.generation:
            self.snapshot = {
                "frame_id": frame_id,
                "generation": generation,
                "captured_at": captured_at,
                "processed_at": time.time(),
                "width": w,
                "height": h,
                "tracks": [t.to_dict() for t in tracks],
                "zones": self.geofence.get_zones_dict(),
                "confirmed_zones": list(self.geofence.confirmed_zones),
                "camera_id": camera_id,
                "source_type": video_manager.source_type,
                "stage_times": dict(self.stage_times),
                "counts": counts,
                "inference_ms": round(self.inference_ms, 1),
                "processing_fps": round(self.processing_fps, 1),
            }

    def _generate_report(self, event, frame):
        try:
            dossier = incident_dispatcher.generate_dossier(event, frame)
            event_hub.publish({"type": "TACTICAL_INCIDENT", "data": dossier})
        except Exception:
            logger.exception("Incident report failed")


# Global pipeline instance
pipeline = SurveillancePipeline()
