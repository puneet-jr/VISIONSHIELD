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
from app.core.cameras import camera_registry, movement
from app.engine.attributes import describe_colours
from app.engine.posture import PoseAnalyzer
from app.engine.crowd import CrowdDynamicsAnalyzer
from app.engine.head_map import head_detector
from app.engine.person_refinement import person_refiner
from app.engine.orientation import orient_phone

logger = logging.getLogger("VisionShield.Pipeline")


class SurveillancePipeline:
    def __init__(self, persist=True, video=None, detector=None, zone_source=None, enable_pose=None):
        self.video = video or video_manager
        self.zone_source = zone_source
        self.crowd = CrowdDynamicsAnalyzer()
        self.head_map = None
        self.last_head_at = 0
        self.persist = persist
        self.identities = IdentityManager()
        self.assessment = AssessmentEngine()
        self.stage_times = {}
        self.processed_frames = 0
        self.skipped_frames = 0
        self.detector = detector or ObjectDetector()
        self.tracker = ObjectTracker(
            retention_seconds=config.TRACK_RETENTION_SECONDS, max_tracks=config.MAX_DETECTIONS * 2
        )
        self.spatio_temporal = SpatioTemporalAnalyzer()
        self.kinematics = KinematicsAnalyzer()
        self.geofence = GeofenceAnalyzer(config.ZONES_PATH)
        self.weapon = WeaponAnalyzer()
        self.pose = PoseAnalyzer(
            enabled=config.ENABLE_POSE and persist if enable_pose is None else enable_pose
        )
        self.person_annotations = {}

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
        with self.video.lock:
            frame_id = self.video.frame_id
            generation = self.video.generation
            captured_at = self.video.last_frame_time
            if frame_id == self.last_frame_id:
                return
            frame = self.video.latest_frame.copy() if self.video.latest_frame is not None else None
        previous_frame_id = self.last_frame_id
        self.last_frame_id = frame_id
        if generation != self.generation:
            self.processed_frames = 0
            self.skipped_frames = 0
            self.assessment = AssessmentEngine()
            self.tracker = ObjectTracker(
                retention_seconds=config.TRACK_RETENTION_SECONDS,
                max_tracks=config.MAX_DETECTIONS * 2,
            )
            self.spatio_temporal = SpatioTemporalAnalyzer()
            self.kinematics = KinematicsAnalyzer()
            self.weapon = WeaponAnalyzer()
            self.event_times.clear()
            self.last_incident_time.clear()
            self.geofence.select_source(self.zone_source or self.video.source_path)
            if not self.geofence.zones and self.geofence.source not in self.geofence.saved:
                self.geofence.add_zone(
                    "default-crowd",
                    "Observed crowd count review",
                    [[0, 0], [1, 0], [1, 1], [0, 1]],
                    purpose="crowd_monitor",
                    max_capacity=config.CROWD_DEFAULT_CAPACITY,
                )
            self.crowd = CrowdDynamicsAnalyzer()
            self.head_map = None
            self.last_head_at = 0
            self.generation = generation
        if frame is None:
            return

        timestamp = captured_at or time.time()
        h, w = frame.shape[:2]
        if (
            self.persist
            and config.ENABLE_HEAD_MAP
            and timestamp - self.last_head_at >= config.HEAD_MAP_INTERVAL_SECONDS
        ):
            self.head_map = head_detector.detect(frame)
            self.last_head_at = timestamp
            if self.head_map is not None:
                self.head_map["captured_at"] = timestamp

        # 1. Object Detection
        started = time.perf_counter()
        detections = self.detector.detect(frame) if self.video.camera_online else []
        if self.persist and config.AUTO_ORIENT_PHONE and self.video.source_type=='phone_upload' and not self.video.orientation_checked and frame.std()>10:
            angle,oriented,proposals=orient_phone(frame,detections,self.detector.detect)
            if self.video.set_phone_rotation(angle,generation):
                frame=oriented;detections=proposals;h,w=frame.shape[:2]
        refinement_ms=0
        if self.persist and config.ENABLE_PERSON_REFINEMENT and self.video.camera_online:
            refine_started=time.perf_counter()
            detections=person_refiner.refine(frame,detections)
            refinement_ms=round((time.perf_counter()-refine_started)*1000,1)
        self.inference_ms = (time.perf_counter() - started) * 1000

        if generation != self.video.generation:
            return

        stage = time.perf_counter()
        # 2. Stateful Tracking
        tracks = [
            t
            for t in self.tracker.update(detections, timestamp, frame=frame)
            if t.disappeared_count == 0
        ]

        camera_id = self.video.camera_id
        self.identities.associate(tracks, camera_id, generation, timestamp)
        pending_segments = self.identities.drain_history()
        if self.persist:
            db.record_identity_segments(pending_segments)
        self.stage_times["tracking_identity_ms"] = round((time.perf_counter() - stage) * 1000, 2)
        stage = time.perf_counter()
        camera_settings = camera_registry.get(camera_id)
        self.person_annotations = {
            pid: annotation
            for pid, annotation in self.person_annotations.items()
            if pid in self.identities.people
        }
        for track in tracks:
            describe_colours(frame, track)
            track.movement = movement(track, camera_settings, w, h)
            if track.label == "person":
                track.gender = self.person_annotations.get(
                    track.person_id, {"label": None, "source": "not inferred from appearance"}
                )
        self.pose.apply(tracks, frame, timestamp, generation)
        self.stage_times["attributes_ms"] = round((time.perf_counter() - stage) * 1000, 2)
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
        crowd_events = self.crowd.process(tracks, self.geofence.zones, w, h, timestamp)
        all_events = st_events + km_events + gf_events + weapon_events + crowd_events

        self.stage_times["association_temporal_ms"] = round((time.perf_counter() - stage) * 1000, 2)
        stage = time.perf_counter()
        for candidate in all_events:
            event = self.assessment.assess(candidate, tracks, camera_id, timestamp)
            if event is None:
                continue
            event["camera_location"] = camera_settings.get("location")
            subject = next((t for t in tracks if t.track_id == event["track_id"]), None)
            if subject:
                event["observed_attributes"] = {
                    "colours": getattr(subject, "colours", {}),
                    "movement": getattr(subject, "movement", None),
                    "posture": getattr(subject, "posture", None),
                }
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
        if generation == self.video.generation:
            self.snapshot = {
                "frame_id": frame_id,
                "generation": generation,
                "captured_at": captured_at,
                "processed_at": time.time(),
                "width": w,
                "height": h,
                "tracks": [t.to_dict() for t in tracks],
                "person_map":{'points':[[round((t.box[0]+t.box[2])/2,1),round(t.box[1]+.12*(t.box[3]-t.box[1]),1),t.track_id] for t in tracks if t.label=='person'],
                    'count':sum(t.label=='person' for t in tracks),'frame_id':frame_id,'coordinates':'image pixels; estimated position from detected person box, not a head detector'},
                "zones": self.geofence.get_zones_dict(),
                "confirmed_zones": list(self.geofence.confirmed_zones),
                "camera_id": camera_id,
                "camera": camera_settings,
                "pose": self.pose.information(),
                "source_type": self.video.source_type,
                "stage_times": dict(self.stage_times),
                "counts": counts,
                "person_detection":{'method':'full_frame_plus_enlarged_tiles' if self.persist and config.ENABLE_PERSON_REFINEMENT and person_refiner.error is None else 'full_frame','refinement_ms':refinement_ms,'error':person_refiner.error,'phone_rotation':self.video.phone_rotation},
                "head_map": self.head_map if timestamp - self.last_head_at < 1.5 else None,
                "crowd": {
                    "zones": self.crowd.zone_metrics,
                    "count_scope": "visible detections, not unique attendance",
                },
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
