"""Download a reproducible, small COCO val2017 check set; never used to train."""

from pathlib import Path
from zipfile import ZipFile
import json
import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "coco_val_sample"
LABEL_URL = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / "coco2017labels.zip"
    with httpx.Client(timeout=60, follow_redirects=True, trust_env=False) as client:
        if not archive.exists():
            with client.stream("GET", LABEL_URL) as response:
                response.raise_for_status()
                with archive.open("wb") as target:
                    for chunk in response.iter_bytes():
                        target.write(chunk)
        with ZipFile(archive) as dataset:
            names = sorted(
                n for n in dataset.namelist() if "/labels/val2017/" in n and n.endswith(".txt")
            )
            positive, negative = [], []
            for name in names:
                text = dataset.read(name).decode("utf-8")
                group = (
                    positive
                    if any(line.split()[0] == "43" for line in text.splitlines() if line.strip())
                    else negative
                )
                if len(group) < 10:
                    group.append((name, text))
                if len(positive) == len(negative) == 10:
                    break
            assert len(positive) == len(negative) == 10, (
                "Could not select labeled validation sample"
            )
            manifest = []
            for name, text in positive + negative:
                stem = Path(name).stem
                source = f"https://s3.amazonaws.com/images.cocodataset.org/val2017/{stem}.jpg"
                target = OUT / (stem + ".jpg")
                if not target.exists():
                    response = client.get(source)
                    response.raise_for_status()
                    target.write_bytes(response.content)
                (OUT / (stem + ".txt")).write_text(text, encoding="utf-8")
                manifest.append(
                    {
                        "file": target.name,
                        "source": source,
                        "labels_source": LABEL_URL,
                        "split": "val2017",
                    }
                )
                print("Prepared", target.name, flush=True)
            (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        "Prepared 20 COCO val2017 images: first 10 knife-positive and first 10 knife-negative IDs. This is a small biased check, not a full benchmark."
    )


if __name__ == "__main__":
    main()
