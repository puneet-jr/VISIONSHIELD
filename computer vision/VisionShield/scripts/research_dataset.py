"""Acquire an explicitly small, legitimate COCO subset; audit/convert/split reviewed datasets."""

from pathlib import Path
import argparse, json, sys, shutil
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.research.datasets import audit, digest, assign_groups, write_yaml

LABEL_URL = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip"


def acquire(root):
    import httpx

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = ROOT / "datasets/coco_val_sample/coco2017labels.zip"
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", LABEL_URL, follow_redirects=True, timeout=60) as response:
            response.raise_for_status()
            with archive.open("wb") as output:
                for block in response.iter_bytes():
                    output.write(block)
    old_manifest = ROOT / "datasets/coco_val_sample/manifest.json"
    excluded = (
        {Path(r["file"]).stem for r in json.loads(old_manifest.read_text())}
        if old_manifest.exists()
        else set()
    )
    names = json.loads((ROOT / "yolo11s.onnx.json").read_text())["names"]
    selected = []
    with ZipFile(archive) as data:
        for original, split, positive_count, negative_count in [
            ("train2017", "train", 4, 4),
            ("val2017", "val", 2, 2),
            ("val2017", "test", 2, 2),
        ]:
            counts = {True: 0, False: 0}
            for member in sorted(
                n for n in data.namelist() if f"/labels/{original}/" in n and n.endswith(".txt")
            ):
                stem = Path(member).stem
                if stem in excluded:
                    continue
                raw = data.read(member).decode("utf-8")
                positive = any(line.split()[0] == "43" for line in raw.splitlines() if line.strip())
                if counts[positive] >= (positive_count if positive else negative_count):
                    continue
                selected.append((stem, original, split, raw))
                excluded.add(stem)
                counts[positive] += 1
                if counts[True] == positive_count and counts[False] == negative_count:
                    break
    records = []
    clipped = 0
    with httpx.Client(timeout=40, follow_redirects=True, trust_env=False) as client:
        for stem, original, split, raw in selected:
            image = root / "images" / split / (stem + ".jpg")
            label = root / "labels" / split / (stem + ".txt")
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://s3.amazonaws.com/images.cocodataset.org/{original}/{stem}.jpg"
            if not image.exists():
                response = client.get(url)
                response.raise_for_status()
                image.write_bytes(response.content)
            lines = []
            for line in raw.splitlines():
                if not line.strip():
                    continue
                cls, x, y, w, h = map(float, line.split())
                a, b, c, d = (
                    max(0, x - w / 2),
                    max(0, y - h / 2),
                    min(1, x + w / 2),
                    min(1, y + h / 2),
                )
                if abs((c - a) - w) > 0.00001 or abs((d - b) - h) > 0.00001:
                    clipped += 1
                if c > a and d > b:
                    lines.append(
                        f"{int(cls)} {(a + c) / 2:.7f} {(b + d) / 2:.7f} {c - a:.7f} {d - b:.7f}"
                    )
            label.write_text("\n".join(lines) + "\n", encoding="utf-8")
            records.append(
                {
                    "image": image.relative_to(root).as_posix(),
                    "label": label.relative_to(root).as_posix(),
                    "split": split,
                    "sequence_id": f"coco:{original}:{stem}",
                    "source_url": url,
                    "annotations_source": LABEL_URL,
                    "sha256": digest(image),
                }
            )
            print("Prepared", split, stem, flush=True)
    manifest = {
        "dataset": "COCO miniature fine-tuning smoke subset",
        "seed": 42,
        "names": names,
        "records": records,
        "annotation_license": "COCO annotations CC BY 4.0; Ultralytics YOLO conversion AGPL-3.0",
        "image_rights": "Individual image licenses; COCO does not own image copyright. See https://cocodataset.org/#termsofuse",
        "clipped_boxes": clipped,
        "split_policy": "COCO train images for local train; disjoint COCO val images for local val/test, excluding previous development sample",
        "limitations": "16 images are only a pipeline smoke test. Pretrained vendor weights already saw COCO training images; test is held out from local fine-tuning, not an independent deployment benchmark.",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_yaml(root, manifest)
    report = audit(root)
    (root / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["valid"]:
        raise ValueError(report["errors"])
    print(json.dumps(report, indent=2))


def convert_coco(annotation_file, image_root, output):
    source = json.loads(Path(annotation_file).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    categories = sorted(source["categories"], key=lambda c: c["id"])
    mapping = {c["id"]: i for i, c in enumerate(categories)}
    annotations = {}
    for annotation in source["annotations"]:
        if not annotation.get("iscrowd"):
            annotations.setdefault(annotation["image_id"], []).append(annotation)
    records = []
    basenames = [Path(image["file_name"]).stem.lower() for image in source["images"]]
    if len(basenames) != len(set(basenames)):
        raise ValueError(
            "Duplicate image basenames would overwrite labels; rename source files first"
        )
    for image in source["images"]:
        original = (Path(image_root) / image["file_name"]).resolve()
        if not original.is_relative_to(Path(image_root).resolve()):
            raise ValueError("Image path escapes root")
        target = output / "images" / original.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(original, target)
        label = output / "labels" / (target.stem + ".txt")
        label.parent.mkdir(exist_ok=True)
        lines = []
        for annotation in annotations.get(image["id"], []):
            x, y, w, h = annotation["bbox"]
            width, height = image["width"], image["height"]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x + w), min(height, y + h)
            if x2 > x1 and y2 > y1:
                lines.append(
                    f"{mapping[annotation['category_id']]} {(x1 + x2) / (2 * width)} {(y1 + y2) / (2 * height)} {(x2 - x1) / width} {(y2 - y1) / height}"
                )
        label.write_text("\n".join(lines), encoding="utf-8")
        records.append(
            {
                "image": target.relative_to(output).as_posix(),
                "label": label.relative_to(output).as_posix(),
                "sequence_id": str(image.get("sequence_id", f"static-image:{image['id']}")),
                "sha256": digest(target),
            }
        )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "names": {i: c["name"] for i, c in enumerate(categories)},
                "records": records,
                "source": str(annotation_file),
                "conversion": "xywh pixels to clipped normalized YOLO boxes; crowd annotations excluded",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


parser = argparse.ArgumentParser()
commands = parser.add_subparsers(dest="command", required=True)
for name in ["acquire-smoke", "audit", "split"]:
    p = commands.add_parser(name)
    p.add_argument("--root", type=Path, default=ROOT / "datasets/security_smoke")
p = commands.add_parser("convert-coco")
p.add_argument("--annotations", type=Path, required=True)
p.add_argument("--images", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.command == "acquire-smoke":
    acquire(args.root)
elif args.command == "convert-coco":
    convert_coco(args.annotations, args.images, args.output)
else:
    if args.command == "split":
        file = args.root / "manifest.json"
        manifest = json.loads(file.read_text())
        manifest["records"] = assign_groups(manifest["records"])
        manifest["seed"] = 42
        file.write_text(json.dumps(manifest, indent=2))
        write_yaml(args.root, manifest)
    result = audit(args.root)
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        sys.exit(1)
