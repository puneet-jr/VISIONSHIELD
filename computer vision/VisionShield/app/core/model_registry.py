"""Registry reports the actual selected checkpoint; no automatic promotion of smoke-trained models."""

from pathlib import Path
from hashlib import sha256
import json
import platform
from importlib.metadata import version
from app.config import config, ROOT

_cached = {}


def model_information(detector):
    path = Path(detector.model_name)
    key = (str(path), path.stat().st_mtime_ns if path.exists() else 0)
    if key not in _cached:
        digest = sha256(path.read_bytes()).hexdigest() if path.exists() else None
        entries = []
        for registry in [
            ROOT / "artifacts/model-card.json",
            ROOT / "data/models/registry.json",
        ]:
            try:
                data = json.loads(registry.read_text(encoding="utf-8"))
                entries.extend(data if isinstance(data, list) else data.get("models", []))
            except (FileNotFoundError, ValueError):
                pass
        record = next((m for m in entries if m.get("sha256") == digest), {})
        _cached[key] = {
            "checkpoint": path.name,
            "sha256": digest,
            "architecture": record.get("architecture", path.stem.upper() + " object detector"),
            "source": record.get("source", "unregistered local checkpoint"),
            "training_status": record.get(
                "training_status",
                "vendor_pretrained"
                if record.get("custom_training_performed") is False
                else "unregistered",
            ),
            "training_dataset": record.get(
                "dataset",
                "MS COCO 2017" if record.get("custom_training_performed") is False else "unknown",
            ),
            "license": record.get(
                "license",
                "Ultralytics AGPL-3.0 or commercial license; verify the applicable terms",
            ),
            "ultralytics_version": version("ultralytics"),
            "hardware": platform.processor(),
            "registry_entry": record,
        }
    info = dict(_cached[key])
    info.update(
        backend=detector.backend,
        input_size=config.INPUT_SIZE,
        loaded=detector.model is not None,
        error=detector.error,
        supported_classes=detector.enabled_labels,
        threat_model={"type": "rules_v1", "trained": False},
        identity_model={
            "type": "torso colour descriptor + motion",
            "learned_embedding": False,
            "real_identity": False,
            "cross_camera": False,
        },
    )
    evaluations = {}
    for name in [
        "knife-validation-yolo11s.json",
        "backend-benchmark.json",
        "tracking-evaluation.json",
        "event-evaluation.json",
        "detector-heldout.json",
        "training-smoke.json",
        "pose-verification.json",
    ]:
        file = ROOT / "artifacts" / name
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            evaluations[name] = {
                k: v
                for k, v in data.items()
                if k
                not in {
                    "rows",
                    "prediction_comparison",
                    "measurements_ms",
                    "frame_results",
                }
            }
        except (FileNotFoundError, ValueError):
            pass
    info["evaluations"] = evaluations
    info["limitations"] = [
        "General object detection, not intent or verified danger.",
        "Knife recall is limited on the documented small selected COCO check.",
        "Anonymous person association is camera/session scoped and uncertain.",
        "Synthetic fixtures measure implementation behavior, not deployment accuracy.",
    ]
    return info
