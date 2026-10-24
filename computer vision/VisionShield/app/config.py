"""VisionShield Application Configuration
Clean, type-safe settings with robust defaults for CPU execution.
"""
from dataclasses import dataclass, field
import os
import secrets
from pathlib import Path
from typing import List, Dict, Any
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / "data" / "ultralytics"))

@dataclass
class AppConfig:
    # Server & Streaming
    HOST: str = "0.0.0.0"
    PORT: int = int(os.getenv("PORT", "8000"))
    PUBLIC_BASE_URL: str = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
    ACCESS_TOKEN: str = os.getenv("VISIONSHIELD_TOKEN") or secrets.token_urlsafe(32)
    TARGET_FPS: int = 12
    CPU_THREADS: int = max(1, min(4, os.cpu_count() or 2))
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    DEBUG: bool = False
    
    # Computer Vision Engine
    YOLO_MODEL: str = os.getenv("YOLO_MODEL", str(Path(__file__).resolve().parents[1] / "yolo11s.pt"))  # COCO-pretrained small model; --fast selects nano
    CONFIDENCE_THRESHOLD: float = 0.40
    DETECT_ALL_CLASSES: bool = os.getenv("DETECT_ALL_CLASSES", "1") == "1"
    INFERENCE_BACKEND: str = os.getenv("INFERENCE_BACKEND", "auto")
    IOU_THRESHOLD: float = 0.45
    INPUT_SIZE: int = int(os.getenv("INPUT_SIZE", "640"))
    DEVICE: str = os.getenv("DEVICE", "cpu")  # Works flawlessly on standard laptop CPU
    
    # Target Object Classes
    PERSON_CLASS: str = "person"
    LUGGAGE_CLASSES: List[str] = field(default_factory=lambda: [
        "backpack", "handbag", "suitcase"
    ])
    VEHICLE_CLASSES: List[str] = field(default_factory=lambda: [
        "car", "motorcycle", "bicycle"
    ])
    
    # Layer 1: Spatio-Temporal Entity Association (Luggage / Owner)
    OWNER_ATTACH_DISTANCE_PX: float = 120.0  # Max pixel distance to consider person owning the bag
    ABANDONED_MIN_SEPARATION_PX: float = 180.0  # Distance before separation flag triggers
    ABANDONED_DWELL_SECONDS: float = 6.0  # Seconds before marking unattended luggage as THREAT
    
    # Layer 2: Kinematic Anomaly & Fall Detection
    FALL_ASPECT_RATIO_THRESHOLD: float = 0.65  # H / W sudden collapse indicates fall
    FALL_VELOCITY_THRESHOLD_PX: float = 25.0  # Rapid downward vertical velocity
    ENABLE_LOITERING: bool = os.getenv("ENABLE_LOITERING", "0") == "1"
    LOITERING_TIME_SECONDS: float = 10.0  # Dwell time in perimeter before loitering alert
    
    # Layer 3: Interactive Geofencing
    # No automatic exclusion area: the operator must define the actual floor boundary.
    DEFAULT_ZONES: List[Dict[str, Any]] = field(default_factory=list)
    ZONE_DWELL_SECONDS: float = 1.5
    ZONE_MIN_OBSERVATIONS: int = 3
    KNIFE_CONFIDENCE: float = 0.55
    KNIFE_CONFIRM_SECONDS: float = 1.0
    KNIFE_MIN_OBSERVATIONS: int = 3
    ZONES_PATH: str = str(Path(__file__).resolve().parents[1] / "data" / "zones.json")

    # Layer 4: Multimodal Agentic Dispatcher
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    ENABLE_AI_DOSSIER: bool = True
    
    # Database
    DB_PATH: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "visionshield.db")

config = AppConfig()
