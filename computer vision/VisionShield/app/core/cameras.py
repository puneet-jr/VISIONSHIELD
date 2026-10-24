"""Camera-mounted coordinates and explicit scene scale; never server IP geolocation."""

from pathlib import Path
from threading import RLock
import json
import time
from app.config import config


class CameraRegistry:
    def __init__(self, path=None):
        self.path = Path(path or config.CAMERA_SETTINGS_PATH)
        self.lock = RLock()
        self.live_locations = {}
        try:
            self.records = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.records = {}

    def get(self, camera_id):
        with self.lock:
            return {
                "camera_id": camera_id,
                "location": None,
                "name": camera_id,
                **self.records.get(camera_id, {}),
                **self.live_locations.get(camera_id, {}),
            }

    def update(self, camera_id, values, persist=True):
        with self.lock:
            if not persist:
                self.live_locations[camera_id] = {**values, "updated_at": time.time()}
                if len(self.live_locations) > 128:
                    self.live_locations.pop(next(iter(self.live_locations)))
                return self.get(camera_id)
            if "location" in values:
                self.live_locations.pop(camera_id, None)
            self.records[camera_id] = {
                **self.records.get(camera_id, {}),
                **values,
                "updated_at": time.time(),
            }
            if len(self.records) > 128:
                oldest = min(self.records, key=lambda key: self.records[key].get("updated_at", 0))
                self.records.pop(oldest, None)
            if persist:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.path.with_suffix(".tmp")
                temporary.write_text(json.dumps(self.records, indent=2), encoding="utf-8")
                temporary.replace(self.path)
            return self.get(camera_id)


def movement(track, settings, width, height):
    valid = getattr(track, "velocity_valid", False)
    vx, vy = map(float, track.motion)
    speed = (vx * vx + vy * vy) ** 0.5
    result = {
        "speed_px_s": round(speed, 1) if valid else None,
        "velocity_px_s": {"x": round(vx, 1), "y": round(vy, 1)} if valid else None,
        "coordinate_system": "image pixels; +x right, +y down",
        "speed_m_s": None,
        "speed_km_h": None,
        "calibration": "not calibrated; no physical-speed claim",
    }
    scale = settings.get("metres_per_pixel")
    rw = settings.get("reference_width")
    rh = settings.get("reference_height")
    if valid and scale and rw and rh and abs(width / height - rw / rh) < 0.01:
        metres = speed * scale * rw / width
        result.update(
            speed_m_s=round(metres, 2),
            speed_km_h=round(metres * 3.6, 2),
            calibration="operator scalar calibration; approximate on the reference plane only, not perspective-correct",
        )
    return result


camera_registry = CameraRegistry()
