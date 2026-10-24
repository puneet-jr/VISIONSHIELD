"""Acquire pinned publisher head weights, verify LFS hash, export a local ONNX graph."""

from pathlib import Path
from hashlib import sha256
import json
import sys
import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import config


def main():
    repo = "wesjos/YOLO-head-detection"
    filename = "v8s-head.pt"
    with httpx.Client(follow_redirects=True, timeout=90) as client:
        metadata = (
            client.get("https://huggingface.co/api/models/" + repo, params={"blobs": "true"})
            .raise_for_status()
            .json()
        )
        revision = metadata["sha"]
        entry = next(item for item in metadata["siblings"] if item["rfilename"] == filename)
        expected = entry["lfs"]["sha256"]
        root = ROOT / "data/models"
        root.mkdir(parents=True, exist_ok=True)
        path = root / filename
        if not path.exists() or sha256(path.read_bytes()).hexdigest() != expected:
            response = client.get(
                f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
            ).raise_for_status()
            if (
                len(response.content) > 30 * 1024**2
                or sha256(response.content).hexdigest() != expected
            ):
                raise ValueError("Head weight checksum/size mismatch")
            path.write_bytes(response.content)
    from ultralytics import YOLO
    import torch

    torch.set_num_threads(config.CPU_THREADS)
    model = YOLO(str(path))
    model.model.names = {0: "head"}
    graph = Path(model.export(format="onnx", imgsz=640, simplify=False, opset=17, dynamic=True))
    provenance = {
        "source_sha256": expected,
        "graph_sha256": sha256(graph.read_bytes()).hexdigest(),
        "names": {0: "head"},
        "source": f"https://huggingface.co/{repo}/tree/{revision}",
        "revision": revision,
        "license": "AGPL-3.0",
        "training": "Publisher reports approximately 15000 images; dataset identity/splits not disclosed. No local crowd training claimed.",
        "export": "float32 dynamic ONNX opset17",
    }
    graph.with_suffix(".onnx.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (ROOT / "artifacts/head-model-provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print("Head model verified and exported:", graph)


if __name__ == "__main__":
    main()
