"""Acquire and inspect the user's crowd sources without inventing box annotations."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse, json, csv, io
from hashlib import sha256
from zipfile import ZipFile
import httpx

SOURCES = {
    "mall": {
        "url": "https://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/mall_dataset.zip",
        "license": "Research only, noncommercial; cite Loy et al.",
        "kind": "head point/count annotations, not person boxes",
    },
    "hajj": {
        "url": "https://www.kaggle.com/api/v1/datasets/download/ziya07/hajj-and-umrah-crowd-management-dataset",
        "license": "Kaggle metadata CC0; inspect downloaded contents",
        "kind": "unverified until downloaded",
    },
    "vscrowd-annotations": {
        "url": "https://huggingface.co/datasets/HopLeeTop/VSCrowd/resolve/main/annotations.zip",
        "license": "HF publisher lists MIT; verify underlying video rights",
        "kind": "head boxes/points/tracking IDs; videos are separate multi-part ~19 GB download",
    },
}


def acquire(name):
    root = ROOT / "datasets/crowd" / name
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "source.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        size = 0
        with httpx.stream(
            "GET", SOURCES[name]["url"], follow_redirects=True, timeout=90
        ) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for block in response.iter_bytes():
                    size += len(block)
                    if size > 200 * 1024**2:
                        raise ValueError("Archive exceeds 200 MiB acquisition budget")
                    output.write(block)
        temporary.replace(archive)
    with ZipFile(archive) as zipped:
        members = [member for member in zipped.namelist() if not member.endswith("/")]
        inspected = {}
        for member in members:
            if member.lower().endswith(".csv"):
                reader = csv.DictReader(io.StringIO(zipped.read(member).decode("utf-8-sig")))
                rows = list(reader)
                inspected[member] = {
                    "columns": reader.fieldnames,
                    "rows": len(rows),
                    "sample": rows[:2],
                }
        report = {
            **SOURCES[name],
            "acquired": True,
            "bytes": archive.stat().st_size,
            "sha256": sha256(archive.read_bytes()).hexdigest(),
            "files": members,
            "csv": inspected,
            "trained": False,
        }
        if name == "hajj":
            report["kind"] = (
                "tabular CSV, not detector image/box data"
                if inspected
                and not any(m.lower().endswith((".jpg", ".png", ".jpeg")) for m in members)
                else "inspect files before training"
            )
    (root / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "files"} | {"file_count": len(members)},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=[*SOURCES, "all"])
    args = parser.parse_args()
    for source in SOURCES if args.source == "all" else [args.source]:
        acquire(source)
