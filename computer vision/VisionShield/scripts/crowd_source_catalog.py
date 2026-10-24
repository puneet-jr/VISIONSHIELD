"""Record actual acquisition evidence and public metadata for all supplied sources."""

from pathlib import Path
import json, zipfile
import httpx

ROOT = Path(__file__).resolve().parents[1]
sources = [
    {
        "name": "VSCrowd",
        "url": "https://github.com/HopLee6/VSCrowd-Dataset",
        "status": "annotations acquired; video files not acquired",
        "use": "Head-box/tracking annotation audit; not full-body training. Videos require a separate approximately 19 GB multipart download.",
        "license": "Publisher HF metadata MIT; verify video rights",
    },
    {
        "name": "UCF crowd segmentation",
        "url": "https://www.crcv.ucf.edu/data/crowd.php",
        "status": "legacy page unavailable during acquisition",
        "use": "Segmentation/motion research candidate; not assumed to contain person boxes or danger labels.",
        "license": "Verify original source terms",
    },
    {
        "name": "Mall",
        "url": "https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html",
        "status": "2000 images and head points acquired",
        "use": "Frozen-model 24-frame counting evaluation and explicitly labelled sampled recording; no training on benchmark frames.",
        "license": "Research/noncommercial; cite Loy et al.",
    },
    {
        "name": "Religious gathering / crowd rally",
        "url": "https://www.kaggle.com/datasets/dataclusterlabs/religious-gathering-crowd-rally-crowd",
        "status": "metadata check pending",
        "use": "Candidate scene images; annotation geometry and license must be verified before detector training.",
    },
    {
        "name": "Crowd Detection all videos",
        "url": "https://universe.roboflow.com/crowddetection-eddqo/crowd-detection-all-videos",
        "status": "publisher page checked; image export not acquired",
        "use": "4033 images, class named 0; verify geometry and semantic mapping. Export requires workspace/API access.",
        "license": "Publisher page CC BY 4.0",
    },
    {
        "name": "Hajj and Umrah",
        "url": "https://www.kaggle.com/datasets/ziya07/hajj-and-umrah-crowd-management-dataset",
        "status": "CSV acquired and inspected; no images",
        "use": "Tabular dataset audit only. Not detector image training or validated stampede ground truth.",
        "license": "Kaggle metadata CC0",
    },
    {
        "name": "Roboflow crowd counting",
        "url": "https://universe.roboflow.com/crowd-dataset/crowd-counting-dataset-w3o7w",
        "status": "publisher page checked; image export not acquired",
        "use": "2898 images, people/head counting; verify head versus full-body geometry. Export requires workspace/API access.",
        "license": "Publisher page CC BY 4.0",
    },
    {
        "name": "UniDataPro crowd counting",
        "url": "https://huggingface.co/datasets/UniDataPro/crowd-counting",
        "status": "public card checked; preview only, full data not acquired",
        "use": "40-image coarse count-bin preview; full head annotations are not public preview. Not suitable as supplied for person-box training.",
        "license": "CC BY NC ND 4.0; full product access separately licensed",
    },
    {
        "name": "Kaggle crowd counting",
        "url": "https://www.kaggle.com/datasets/fmena14/crowd-counting",
        "status": "metadata check pending",
        "use": "Candidate counting data; inspect format and sequence split before training.",
    },
]
for entry in sources:
    if "kaggle.com/" in entry["url"] and entry["status"] == "metadata check pending":
        slug = entry["url"].split("/datasets/")[1]
        query = slug.split("/")[1]
        try:
            response = httpx.get(
                "https://www.kaggle.com/api/v1/datasets/list",
                params={"search": query},
                follow_redirects=True,
                timeout=25,
            )
            response.raise_for_status()
            item = next(item for item in response.json() if item.get("ref") == slug)
            entry.update(
                status="public metadata checked; archive not acquired",
                license=item.get("licenseName"),
                reported_bytes=item.get("totalBytes"),
                description=item.get("subtitle"),
            )
        except Exception as exc:
            entry["status"] = "metadata/archive not acquired: " + type(exc).__name__
for name, index in [("mall", 2), ("hajj", 5), ("vscrowd-annotations", 0)]:
    manifest = json.loads(
        (ROOT / "datasets/crowd" / name / "manifest.json").read_text(encoding="utf-8")
    )
    sources[index]["acquisition"] = {
        key: manifest[key] for key in ("bytes", "sha256", "acquired", "trained")
    }
    if name == "hajj":
        sources[index]["tables"] = {
            key: {"rows": value["rows"], "columns": value["columns"]}
            for key, value in manifest["csv"].items()
        }
    if name == "vscrowd-annotations":
        with zipfile.ZipFile(ROOT / "datasets/crowd" / name / "source.zip") as data:
            names = [n for n in data.namelist() if n.endswith(".txt")]
            example = data.read(names[0]).decode("utf-8").splitlines()[0].strip().split()
            sources[index]["annotation_example"] = {
                "file": names[0],
                "tokens_first_frame": len(example),
                "format": "frame ID then repeated head ID,x1,y1,x2,y2,point x,point y; verify delimiters before conversion",
            }
report = {
    "model_status": "Live yolo11s pretrained COCO detector; no crowd-trained weights promoted. Mall evaluation exposes dense-crowd undercount.",
    "trained_on_supplied_sources": False,
    "sources": sources,
}
(ROOT / "artifacts/crowd-dataset-catalog.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print(json.dumps(report, indent=2))
