"""Bounded latest-frame capture and shared JPEG output; no frame backlog."""
import logging
import threading
import time
import cv2
import numpy as np

logger = logging.getLogger("VisionShield.VideoStream")


class VideoStreamManager:
    def __init__(self):
        self.cap = None
        self.source_type = "webcam"
        self.source_path = "0"
        self.latest_frame = None
        self.jpeg = None
        self.jpeg_id = 0
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
        with self.source_lock:
            self.stop()
            with self.lock:
                self.generation += 1
                self.latest_frame = self.jpeg = None
                self.camera_online = False
                self.fps = 0.0
                self.last_frame_time = 0.0
                self.source_path = source
                self.source_type = "phone_upload" if source == "phone_upload" else "webcam" if source.isdigit() else "ip_stream"
            self.stop_event = threading.Event()
            self.is_running = True
            self.preview_thread = threading.Thread(target=self._preview_loop, args=(self.stop_event, self.generation), daemon=True, name="preview")
            self.preview_thread.start()
            if source == "phone_upload":
                return
            self.thread = threading.Thread(target=self._capture_loop, args=(source, self.stop_event, self.generation), daemon=True, name="capture")
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

    def _store(self, frame, online=True, generation=None):
        now = time.time()
        with self.lock:
            if generation is not None and generation != self.generation:
                return False
            if self.last_frame_time:
                rate = 1 / max(now - self.last_frame_time, 0.001)
                self.fps = 0.8 * self.fps + 0.2 * rate
            self.last_frame_time = now
            self.frame_count += 1
            self.frame_id += 1
            self.latest_frame = frame
            self.camera_online = online
        self.frame_ready.set()

    def push_phone_frame(self, frame, expected_generation=None):
        with self.source_lock:
            if expected_generation is not None and (self.generation != expected_generation or self.source_type != "phone_upload"):
                return False
            if self.source_type != "phone_upload":
                self.start("phone_upload")
            self._store(frame)
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
                last_id = self.frame_id
            started = time.perf_counter()
            # Encode the latest camera image independently of slow inference.
            h, w = frame.shape[:2]
            scale = min(1.0, 960 / max(h, w))
            if scale < 1:
                frame = cv2.resize(frame, (round(w*scale), round(h*scale)))
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
            now = time.perf_counter()
            if ok:
                with self.lock:
                    if generation == self.generation and not stopped.is_set():
                        self.jpeg = encoded.tobytes()
                        self.jpeg_id += 1
                        self.preview_encoded_at = time.time()
                        if previous:
                            self.preview_fps = 1 / max(now-previous, .001)
                        previous = now
            stopped.wait(max(0, 1/24 - (time.perf_counter()-started)))

    def _capture_loop(self, source, stopped, generation):
        # Device opening is background work: slow Windows drivers cannot block app startup.
        import os
        if source.isdigit() and os.name == "nt":
            cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if stopped.is_set() or generation != self.generation:
            cap.release()
            return
        self.cap = cap
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        try:
            while not stopped.is_set():
                if not cap.isOpened():
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "Connect a camera to begin", (130, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (170, 190, 200), 1)
                    if generation == self.generation:
                        self._store(frame, online=False, generation=generation)
                    stopped.wait(.25)
                    continue
                ok, frame = cap.read()
                if stopped.is_set() or generation != self.generation:
                    break
                if ok:
                    self._store(frame, generation=generation)
                else:
                    self.camera_online = False
                    stopped.wait(.1)
        finally:
            cap.release()


video_manager = VideoStreamManager()
