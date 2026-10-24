"""Bounded latest-frame capture and shared JPEG output; no frame backlog."""

import logging
import threading
import time
import cv2
import numpy as np
from pathlib import Path
from app.config import config

logger = logging.getLogger("VisionShield.VideoStream")


class VideoStreamManager:
    def __init__(self):
        self.cap = None
        self.source_type = "webcam"
        self.source_path = "0"
        self.camera_id = "webcam:0"
        self.preview_metadata = {}
        self.source_frames = 0
        self.preview_skipped = 0
        self.source_error = None
        self.ingress_capture_at = None
        self.recorded = False
        self.latest_frame = None
        self.jpeg = None
        self.jpeg_id = 0
        self.jpeg_start_id = 0
        self.frame_id = 0
        self.generation = 0
        self.is_running = False
        self.camera_online = False
        self.lock = threading.RLock()
        self.source_lock = threading.RLock()
        self.thread = None
        self.stop_event = threading.Event()
        self.preview_thread = None
        self.frame_ready = threading.Event()
        self.preview_fps = 0.0
        self.preview_encoded_at = 0.0
        self.fps = 0.0
        self.frame_count = 0
        self.last_frame_time = 0.0

    def start(self, source="0"):
        source = str(source)
        if source.startswith("recording:"):
            self.recording_path(source)
        elif source not in {"0", "1", "phone_upload"}:
            raise ValueError("Use an enumerated webcam, phone_upload or a registered recording")
        with self.source_lock:
            self.stop()
            with self.lock:
                self.generation += 1
                self.latest_frame = self.jpeg = None
                self.camera_online = False
                self.fps = 0.0
                self.last_frame_time = 0.0
                self.source_path = source
                self.camera_id = (
                    "phone:unpaired-session"
                    if source == "phone_upload"
                    else source
                    if source.startswith("recording:")
                    else "webcam:" + source
                )
                self.source_frames = 0
                self.jpeg_start_id = self.jpeg_id
                self.preview_skipped = 0
                self.ingress_capture_at = None
                self.source_error = None
                self.preview_metadata = {}
                self.recorded = source.startswith("recording:")
                self.source_type = (
                    "phone_upload"
                    if source == "phone_upload"
                    else "webcam"
                    if source.isdigit()
                    else "recording"
                )
            self.stop_event = threading.Event()
            self.is_running = True
            self.preview_thread = threading.Thread(
                target=self._preview_loop,
                args=(self.stop_event, self.generation),
                daemon=True,
                name="preview",
            )
            self.preview_thread.start()
            if source == "phone_upload":
                return
            self.thread = threading.Thread(
                target=self._capture_loop,
                args=(source, self.stop_event, self.generation),
                daemon=True,
                name="capture",
            )
            self.thread.start()

    def stop(self):
        with self.source_lock:
            self.is_running = False
            self.stop_event.set()
            self.frame_ready.set()
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=2)
            self.thread = None
            if self.preview_thread and self.preview_thread.is_alive():
                self.preview_thread.join(timeout=1)
            self.preview_thread = None
            self.camera_online = False

    def _store(self, frame, online=True, generation=None, captured_at=None):
        now = time.time()
        with self.lock:
            if generation is not None and generation != self.generation:
                return False
            if self.last_frame_time:
                rate = 1 / max(now - self.last_frame_time, 0.001)
                self.fps = 0.8 * self.fps + 0.2 * rate
            self.last_frame_time = now
            self.ingress_capture_at = captured_at
            self.source_frames += 1
            self.frame_count += 1
            self.frame_id += 1
            self.latest_frame = frame
            self.camera_online = online
        self.frame_ready.set()

    def push_phone_frame(self, frame, expected_generation=None, captured_at=None):
        with self.source_lock:
            if expected_generation is not None and (
                self.generation != expected_generation or self.source_type != "phone_upload"
            ):
                return False
            if self.source_type != "phone_upload":
                self.start("phone_upload")
            self._store(frame, captured_at=captured_at)
            return True

    def get_raw_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def _preview_loop(self, stopped, generation):
        last_id = -1
        previous = None
        while not stopped.is_set():
            self.frame_ready.wait(0.1)
            self.frame_ready.clear()
            with self.lock:
                if generation != self.generation:
                    return
                if self.frame_id == last_id or self.latest_frame is None:
                    continue
                frame = self.latest_frame
                if last_id >= 0:
                    self.preview_skipped += max(0, self.frame_id - last_id - 1)
                last_id = self.frame_id
                captured_at = self.ingress_capture_at or self.last_frame_time
                received_at = self.last_frame_time
                frame_camera = self.camera_id
            started = time.perf_counter()
            # Encode the latest camera image independently of slow inference.
            h, w = frame.shape[:2]
            scale = min(1.0, 960 / max(h, w))
            if scale < 1:
                frame = cv2.resize(frame, (round(w * scale), round(h * scale)))
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
            now = time.perf_counter()
            if ok:
                with self.lock:
                    if generation == self.generation and not stopped.is_set():
                        self.jpeg = encoded.tobytes()
                        self.jpeg_id += 1
                        self.preview_encoded_at = time.time()
                        self.preview_metadata = {
                            "frame_id": last_id,
                            "generation": generation,
                            "captured_at": captured_at,
                            "received_at": received_at,
                            "encoded_at": self.preview_encoded_at,
                            "camera_id": frame_camera,
                            "recorded": self.recorded,
                        }
                        if previous:
                            self.preview_fps = 1 / max(now - previous, 0.001)
                        previous = now
            stopped.wait(max(0, 1 / config.PREVIEW_FPS - (time.perf_counter() - started)))

    def _capture_loop(self, source, stopped, generation):
        # Device opening is background work: slow Windows drivers cannot block app startup.
        import os

        recording = source.startswith("recording:")

        def open_capture():
            if recording:
                return cv2.VideoCapture(str(self.recording_path(source)))
            if source.isdigit() and os.name == "nt":
                return cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
            return cv2.VideoCapture(int(source) if source.isdigit() else source)

        cap = open_capture()
        if stopped.is_set() or generation != self.generation:
            cap.release()
            return
        self.cap = cap
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        period = 1 / max(1, cap.get(cv2.CAP_PROP_FPS)) if recording else 0
        failures = 0
        try:
            while not stopped.is_set():
                cycle = time.perf_counter()
                if not cap.isOpened():
                    self.source_error = "Camera could not be opened"
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        frame,
                        "Connect a camera to begin",
                        (130, 235),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (170, 190, 200),
                        1,
                    )
                    if generation == self.generation:
                        self._store(frame, online=False, generation=generation)
                    if recording:
                        break
                    if stopped.wait(2):
                        break
                    cap.release()
                    cap = open_capture()
                    if stopped.is_set() or generation != self.generation:
                        break
                    self.cap = cap
                    continue
                ok, frame = cap.read()
                if stopped.is_set() or generation != self.generation:
                    break
                if ok:
                    failures = 0
                    self.source_error = None
                    self._store(frame, generation=generation)
                else:
                    self.camera_online = False
                    if recording:
                        break
                    self.source_error = "Camera read failed; waiting for recovery"
                    failures += 1
                    if failures >= 10:
                        cap.release()
                        if stopped.wait(2):
                            break
                        cap = open_capture()
                        if stopped.is_set() or generation != self.generation:
                            break
                        self.cap = cap
                        failures = 0
                    stopped.wait(0.1)
                if period:
                    stopped.wait(max(0, period - (time.perf_counter() - cycle)))
        finally:
            cap.release()

    @staticmethod
    def recording_path(source):
        name = source.removeprefix("recording:")
        if Path(name).name != name or Path(name).suffix.lower() not in {
            ".avi",
            ".mp4",
            ".mov",
            ".mkv",
        }:
            raise ValueError("Invalid recording name")
        root = Path(config.RECORDINGS_DIR).resolve()
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Recording must exist in RECORDINGS_DIR")
        return path


video_manager = VideoStreamManager()
