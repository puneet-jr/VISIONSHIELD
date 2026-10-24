"""Export existing official weights without quantization, retraining or server startup."""

from pathlib import Path
from hashlib import sha256
import json
import os
import argparse

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "data/ultralytics"))
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=str(ROOT / "yolo11s.pt"))
args = parser.parse_args()
path = Path(args.model).resolve()
model = YOLO(path)
graph = Path(model.export(format="onnx", imgsz=640, simplify=False, opset=17, dynamic=True))
metadata = {
    "source_sha256": sha256(path.read_bytes()).hexdigest(),
    "graph_sha256": sha256(graph.read_bytes()).hexdigest(),
    "names": model.names,
    "export": "float32 dynamic ONNX opset 17, no quantization",
    "source": path.name,
}
graph.with_suffix(".onnx.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print("Optimized graph and provenance saved:", graph.name)
