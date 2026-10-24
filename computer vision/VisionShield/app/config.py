"""Validated, environment-driven configuration for a single-process research server."""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import get_type_hints
from urllib.parse import urlsplit
import json
import os
import secrets
import math

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "data" / "ultralytics"))


@dataclass
class AppConfig:
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEPLOYMENT_MODE: str = "demo"
    TRUST_LOCAL_DASHBOARD: bool = True
    PUBLIC_BASE_URL: str = ""
    ALLOWED_ORIGINS: list[str] = field(default_factory=list)
    ACCESS_TOKEN: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    CAMERA_TOKEN: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    GUARD_TOKEN: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    MAX_CAMERA_STREAMS: int = 4
    MAX_DETECTIONS: int = 250
    CROWD_DEFAULT_CAPACITY: int = 20
    ENABLE_HEAD_MAP: bool = False
    HEAD_MAP_TILED: bool = False
    HEAD_CONFIDENCE: float = 0.35
    HEAD_MAP_INTERVAL_SECONDS: float = 0.6
    ENABLE_PERSON_REFINEMENT: bool = True
    PERSON_REFINE_MODEL: str = "yolo11s.pt"
    PERSON_REFINE_CONFIDENCE: float = 0.25
    # Keep the CPU refinement worker bounded so it cannot delay the live view.
    PERSON_FULL_INPUT_SIZE: int = 960
    PERSON_TILE_INPUT_SIZE: int = 512
    PERSON_REFINE_INTERVAL_SECONDS: float = 1.5
    PERSON_CACHE_SECONDS: float = 6.0
    AUTO_ORIENT_PHONE: bool = True
    CROWD_WARNING_SECONDS: float = 2.0
    CROWD_GROWTH_WINDOW_SECONDS: float = 10.0
    CROWD_GROWTH_THRESHOLD: float = 0.30
    CROWD_MIN_GROWTH: int = 5
    TARGET_FPS: int = 12
    PREVIEW_FPS: int = 24
    CPU_THREADS: int = max(1, min(4, os.cpu_count() or 2))
    DEBUG: bool = False
    YOLO_MODEL: str = str(ROOT / "yolo11s.pt")
    CONFIDENCE_THRESHOLD: float = 0.40
    DETECT_ALL_CLASSES: bool = True
    INFERENCE_BACKEND: str = "auto"
    IOU_THRESHOLD: float = 0.45
    INPUT_SIZE: int = 640
    DEVICE: str = "cpu"
    TRACK_RETENTION_SECONDS: float = 30.0
    IDENTITY_RETENTION_SECONDS: float = 120.0
    IDENTITY_DISTANCE_THRESHOLD: float = 0.10
    IDENTITY_MATCH_MARGIN: float = 0.04
    PERSON_CLASS: str = "person"
    ANIMAL_CLASSES: list[str] = field(
        default_factory=lambda: [
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
        ]
    )
    ENABLE_POSE: bool = True
    POSE_MODEL: str = str(ROOT / "yolo11n-pose.pt")
    POSE_INTERVAL_SECONDS: float = 0.75
    POSE_MAX_AGE_SECONDS: float = 1.5
    CAMERA_SETTINGS_PATH: str = str(ROOT / "data/cameras.json")
    LUGGAGE_CLASSES: list[str] = field(default_factory=lambda: ["backpack", "handbag", "suitcase"])
    VEHICLE_CLASSES: list[str] = field(
        default_factory=lambda: ["car", "motorcycle", "bicycle", "bus", "truck", "train"]
    )
    OWNER_ATTACH_DISTANCE_PX: float = 120.0
    ASSOCIATION_SECONDS: float = 1.0
    ASSOCIATION_MIN_OBSERVATIONS: int = 3
    ABANDONED_MIN_SEPARATION_PX: float = 180.0
    ABANDONED_DWELL_SECONDS: float = 6.0
    REQUIRE_BAG_MONITOR_ZONE: bool = True
    FALL_ASPECT_RATIO_THRESHOLD: float = 0.65
    FALL_VELOCITY_THRESHOLD_PX: float = 25.0
    ENABLE_LOITERING: bool = False
    ENABLE_FALL_ALERTS: bool = False
    LOITERING_TIME_SECONDS: float = 10.0
    DEFAULT_ZONES: list[dict] = field(default_factory=list)
    ZONE_DWELL_SECONDS: float = 1.5
    ZONE_MIN_OBSERVATIONS: int = 3
    ZONE_IMMEDIATE_WARNING: bool = True
    KNIFE_IMMEDIATE_WARNING: bool = True
    KNIFE_REQUIRE_PERSON: bool = False
    KNIFE_CONFIDENCE: float = 0.55
    KNIFE_CONFIRM_SECONDS: float = 1.0
    KNIFE_MIN_OBSERVATIONS: int = 3
    ALERT_COOLDOWN_SECONDS: float = 10.0
    EVIDENCE_RETENTION_DAYS: int = 7
    MAX_STORED_EVENTS: int = 2000
    STORE_EVENT_IMAGES: bool = True
    ZONES_PATH: str = str(ROOT / "data" / "zones.json")
    DB_PATH: str = str(ROOT / "data" / "visionshield.db")
    RECORDINGS_DIR: str = str(ROOT / "data" / "recordings")
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-flash-latest"
    ENABLE_AI_DOSSIER: bool = False

    def __post_init__(self):
        for setting in fields(self):
            value = getattr(self, setting.name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{setting.name} must be finite")
        for name in [
            "PORT",
            "TARGET_FPS",
            "PREVIEW_FPS",
            "CPU_THREADS",
            "INPUT_SIZE",
            "MAX_STORED_EVENTS",
        ]:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.PORT > 65535 or self.INPUT_SIZE % 32 or self.INPUT_SIZE > 1280:
            raise ValueError("PORT must be <=65535; INPUT_SIZE must be a multiple of 32 <=1280")
        for name in [
            "CONFIDENCE_THRESHOLD",
            "IOU_THRESHOLD",
            "KNIFE_CONFIDENCE",
            "IDENTITY_DISTANCE_THRESHOLD",
            "IDENTITY_MATCH_MARGIN",
        ]:
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0,1]")
        for name in [
            "TRACK_RETENTION_SECONDS",
            "IDENTITY_RETENTION_SECONDS",
            "ABANDONED_DWELL_SECONDS",
            "ZONE_DWELL_SECONDS",
            "ALERT_COOLDOWN_SECONDS",
            "KNIFE_CONFIRM_SECONDS",
            "POSE_INTERVAL_SECONDS",
            "POSE_MAX_AGE_SECONDS",
            "ASSOCIATION_SECONDS",
            "OWNER_ATTACH_DISTANCE_PX",
            "ABANDONED_MIN_SEPARATION_PX",
        ]:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            min(
                self.KNIFE_MIN_OBSERVATIONS,
                self.ZONE_MIN_OBSERVATIONS,
                self.ASSOCIATION_MIN_OBSERVATIONS,
            )
            < 2
        ):
            raise ValueError("Confirmation requires at least two observations")
        if not 1 <= self.MAX_CAMERA_STREAMS <= 8 or not 1 <= self.MAX_DETECTIONS <= 1000:
            raise ValueError("Camera streams must be 1-8 and detections 1-1000")
        if (
            self.CROWD_WARNING_SECONDS <= 0
            or self.CROWD_GROWTH_WINDOW_SECONDS < 5
            or not 0 < self.CROWD_GROWTH_THRESHOLD <= 10
            or self.CROWD_MIN_GROWTH < 1
        ):
            raise ValueError("Invalid crowd persistence/growth configuration")
        if self.EVIDENCE_RETENTION_DAYS < 1:
            raise ValueError(
                "Evidence retention must be at least one day; use STORE_EVENT_IMAGES=false to disable images"
            )
        if self.INFERENCE_BACKEND not in {"auto", "torch", "onnx"}:
            raise ValueError("INFERENCE_BACKEND must be auto, torch or onnx")
        if self.DEPLOYMENT_MODE not in {"demo", "production"}:
            raise ValueError("DEPLOYMENT_MODE must be demo or production")
        if self.PUBLIC_BASE_URL:
            self.PUBLIC_BASE_URL = self.PUBLIC_BASE_URL.rstrip("/")
            url = urlsplit(self.PUBLIC_BASE_URL)
            if (
                url.scheme != "https"
                or not url.hostname
                or url.username
                or url.password
                or url.query
                or url.fragment
                or url.path not in {"", "/"}
            ):
                raise ValueError(
                    "PUBLIC_BASE_URL must be an HTTPS origin without credentials, path or query"
                )
        if "*" in self.ALLOWED_ORIGINS:
            raise ValueError("Credentialed CORS requires explicit origins")
        for origin in self.ALLOWED_ORIGINS:
            parsed = urlsplit(origin)
            if (
                parsed.scheme not in {"https", "http"}
                or not parsed.hostname
                or parsed.path not in {"", "/"}
            ):
                raise ValueError("ALLOWED_ORIGINS must contain origins")
        if self.DEPLOYMENT_MODE == "production":
            self.TRUST_LOCAL_DASHBOARD = False
            if not self.PUBLIC_BASE_URL or len(self.ACCESS_TOKEN) < 32:
                raise ValueError(
                    "Production requires PUBLIC_BASE_URL and an ACCESS_TOKEN of at least 32 characters"
                )

    @classmethod
    def from_env(cls, environment=None):
        process_environment = environment is None
        environment = dict(os.environ if environment is None else environment)
        # DEBUG is commonly used by Node/IDE tooling; never interpret that unrelated value as ours.
        if process_environment:
            environment.pop("DEBUG", None)
        for setting in fields(cls):
            prefixed = "VISIONSHIELD_" + setting.name
            if prefixed in environment:
                environment[setting.name] = environment[prefixed]
        if "VISIONSHIELD_TOKEN" in environment and "ACCESS_TOKEN" not in environment:
            environment["ACCESS_TOKEN"] = environment["VISIONSHIELD_TOKEN"]
        hints = get_type_hints(cls)
        values = {}
        for setting in fields(cls):
            if setting.name not in environment:
                continue
            raw = environment[setting.name]
            kind = hints[setting.name]
            try:
                if kind is bool:
                    if str(raw).lower() not in {"true", "false", "1", "0"}:
                        raise ValueError("expected true/false or 1/0")
                    values[setting.name] = str(raw).lower() in {"true", "1"}
                elif kind in {int, float, str}:
                    values[setting.name] = kind(raw)
                else:
                    values[setting.name] = json.loads(raw)
                    if not isinstance(values[setting.name], list):
                        raise ValueError("expected a JSON array")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid {setting.name}: {exc}") from exc
        if values.get("DEPLOYMENT_MODE") == "production" and "ACCESS_TOKEN" not in environment:
            raise ValueError("Set a persistent ACCESS_TOKEN explicitly for production")
        return cls(**values)


config = AppConfig.from_env()
