"""Manifest-driven YOLO dataset audits and sequence-safe splitting."""

from pathlib import Path
from hashlib import sha256
from collections import Counter, defaultdict
import json
import random
import cv2
import numpy as np


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def dhash(image):
    small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (9, 8))
    return int("".join("1" if x else "0" for x in (small[:, 1:] > small[:, :-1]).flatten()), 2)


def audit(root, manifest=None):
    root = Path(root).resolve()
    manifest = (
        json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest is None
        else manifest
    )
    records = manifest["records"]
    names = manifest["names"]
    errors = []
    counts = defaultdict(Counter)
    hashes = {}
    groups = {}
    perceptual = []
    bands = defaultdict(set)
    for record in records:
        split = record.get("split")
        if split not in {"train", "val", "test"}:
            errors.append("Missing valid split: " + record["image"])
            continue
        image_path = (root / record["image"]).resolve()
        label_path = (root / record["label"]).resolve()
        if not image_path.is_relative_to(root) or not label_path.is_relative_to(root):
            errors.append("Path escapes dataset root")
            continue
        if not image_path.is_file() or not label_path.is_file():
            errors.append("Missing image or label: " + record["image"])
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            errors.append("Invalid image: " + record["image"])
            continue
        checksum = digest(image_path)
        if record.get("sha256") and checksum != record["sha256"]:
            errors.append("Checksum changed: " + record["image"])
        if checksum in hashes and hashes[checksum] != split:
            errors.append("Exact duplicate across splits: " + record["image"])
        hashes[checksum] = split
        group = record.get("sequence_id")
        if not group:
            errors.append("Missing sequence_id: " + record["image"])
        elif group in groups and groups[group] != split:
            errors.append("Sequence leakage: " + str(group))
        groups[group] = split
        fingerprint = dhash(image)
        # With <=2 differing bits, at least one of three disjoint bands is equal.
        # This preserves the screen while avoiding an unconditional quadratic scan.
        band_keys = [
            (0, fingerprint & ((1 << 21) - 1)),
            (1, (fingerprint >> 21) & ((1 << 21) - 1)),
            (2, fingerprint >> 42),
        ]
        candidates = set().union(*(bands[key] for key in band_keys))
        for old_index in candidates:
            old_hash, old_split, old_name = perceptual[old_index]
            if old_split != split and (fingerprint ^ old_hash).bit_count() <= 2:
                errors.append("Near duplicate across splits: " + record["image"] + " / " + old_name)
        for key in band_keys:
            bands[key].add(len(perceptual))
        perceptual.append((fingerprint, split, record["image"]))
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                parts = list(map(float, line.split()))
                if (
                    len(parts) != 5
                    or parts[0] != int(parts[0])
                    or not 0 <= int(parts[0]) < len(names)
                ):
                    raise ValueError("class/format")
                _, x, y, w, h = parts
                if (
                    not np.isfinite(parts).all()
                    or w <= 0
                    or h <= 0
                    or x - w / 2 < -0.001
                    or y - h / 2 < -0.001
                    or x + w / 2 > 1.001
                    or y + h / 2 > 1.001
                ):
                    raise ValueError("box bounds")
                counts[split][str(int(parts[0]))] += 1
            except ValueError:
                errors.append("Invalid YOLO box: " + record["label"])
    splits = Counter(r.get("split") for r in records)
    for split in ["train", "val", "test"]:
        if not splits[split]:
            errors.append("Empty " + split + " split")
    return {
        "valid": not errors,
        "errors": errors,
        "images_per_split": dict(splits),
        "class_instances": dict(counts),
        "sequence_count": len(groups),
        "sample_count": len(records),
        "leakage_check": "exact hash, 64-bit dHash <=2 across splits, sequence IDs",
        "limitations": "Perceptual hashing is a near-duplicate screen, not proof that related content is absent.",
    }


def assign_groups(records, seed=42, ratios=(0.7, 0.2, 0.1)):
    groups = sorted({r["sequence_id"] for r in records})
    if len(groups) < 3:
        raise ValueError("At least three independent source groups are required")
    random.Random(seed).shuffle(groups)
    n = len(groups)
    train = max(1, min(n - 2, round(n * ratios[0])))
    val = max(1, min(n - train - 1, round(n * ratios[1])))
    mapping = {
        g: "train" if i < train else "val" if i < train + val else "test"
        for i, g in enumerate(groups)
    }
    return [{**r, "split": mapping[r["sequence_id"]]} for r in records]


def write_yaml(root, manifest):
    import yaml

    root = Path(root).resolve()
    for split in ["train", "val", "test"]:
        (root / (split + ".txt")).write_text(
            "".join(
                str(root / r["image"]) + "\n" for r in manifest["records"] if r["split"] == split
            ),
            encoding="utf-8",
        )
    payload = {
        "path": str(root),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": {int(k): v for k, v in manifest["names"].items()}
        if isinstance(manifest["names"], dict)
        else manifest["names"],
    }
    (root / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
