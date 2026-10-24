"""Audited fine-tuning -> validation selection -> held-out test -> export -> registry.
No automatic deployment promotion. Small smoke runs verify plumbing, not security accuracy.
"""

from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
from uuid import uuid4
import argparse, json, sys, time, os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "data/ultralytics"))
from app.research.datasets import audit


def metric_values(result):
    return {
        "precision": float(result.box.mp),
        "recall": float(result.box.mr),
        "mAP50": float(result.box.map50),
        "mAP50_95": float(result.box.map),
        "speed_ms_per_image": result.speed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "datasets/security_smoke")
    parser.add_argument("--model", type=Path, default=ROOT / "yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--freeze", type=int, default=10)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    report = audit(args.root)
    if not report["valid"]:
        raise ValueError(report["errors"])
    from ultralytics import YOLO
    import torch

    torch.set_num_threads(4)
    run_name = (
        args.run_name
        or "smoke-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:4]
    )
    source = args.model.resolve()
    data = args.root.resolve() / "data.yaml"
    started = time.perf_counter()
    model = YOLO(str(source))
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        seed=42,
        deterministic=True,
        project=str(ROOT / "runs/security"),
        name=run_name,
        exist_ok=False,
        save=True,
        plots=False,
        amp=False,
        freeze=args.freeze,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        fliplr=0.5,
        patience=10,
        verbose=False,
    )
    best = Path(model.trainer.best)
    selected = YOLO(str(best))
    validation = selected.val(
        data=str(data),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        plots=False,
        verbose=False,
    )
    # The best checkpoint has already been selected using validation. No thresholds are tuned on test.
    heldout = selected.val(
        data=str(data),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        plots=False,
        verbose=False,
    )
    baseline = YOLO(str(source)).val(
        data=str(data),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        plots=False,
        verbose=False,
    )
    graph = Path(
        selected.export(format="onnx", imgsz=args.imgsz, dynamic=True, simplify=False, opset=17)
    )
    checksum = sha256(best.read_bytes()).hexdigest()
    graph.with_suffix(".onnx.json").write_text(
        json.dumps(
            {
                "source_sha256": checksum,
                "graph_sha256": sha256(graph.read_bytes()).hexdigest(),
                "names": selected.names,
                "source": best.name,
                "export": "float32 dynamic ONNX opset 17",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    record = {
        "model_name": run_name,
        "architecture": source.stem,
        "version": run_name,
        "checkpoint": str(best),
        "sha256": checksum,
        "source_checkpoint": str(source),
        "source_sha256": sha256(source.read_bytes()).hexdigest(),
        "source": "Local transfer-learning smoke experiment initialized from official Ultralytics checkpoint",
        "dataset": str(args.root.resolve()),
        "manifest_sha256": sha256((args.root / "manifest.json").read_bytes()).hexdigest(),
        "training_status": "locally_fine_tuned_smoke_only",
        "custom_training_performed": True,
        "promoted": False,
        "license": "Ultralytics AGPL-3.0 or commercial terms; individual COCO image rights",
        "configuration": vars(args) | {"root": str(args.root), "model": str(args.model)},
        "seed": 42,
        "date": datetime.now(timezone.utc).isoformat(),
        "names": selected.names,
        "dataset_audit": report,
        "validation": metric_values(validation),
        "heldout_test": metric_values(heldout),
        "baseline_heldout_test": metric_values(baseline),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "export": str(graph),
        "limitations": "Tiny 8/4/4 pipeline smoke run. COCO pretraining overlaps local training images. Held-out test is only held out from local fine-tuning; metrics are not deployment evidence.",
    }
    target = ROOT / "data/models/registry.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    entries = json.loads(target.read_text()) if target.exists() else []
    entries.append(record)
    target.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    output = ROOT / "artifacts/training-smoke.json"
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(output),
                "promoted": False,
                "validation": record["validation"],
                "heldout_test": record["heldout_test"],
                "baseline_test": record["baseline_heldout_test"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
