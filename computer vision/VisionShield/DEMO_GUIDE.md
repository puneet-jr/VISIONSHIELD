# VisionShield demo and model notes

## What is verified and what remains uncertain

The dependency check, Python/JavaScript syntax checks, 29 regression tests, desktop/mobile browser flow and authenticated public camera transport have been exercised. These are software checks, not a guarantee of security accuracy. A real phone camera-permission and staged physical scene test is still required on your device.

The current system detects objects and applies explicit scene rules. It cannot tell whether someone is authorized, whether a knife is being used innocently, whether a person intends harm, or whether a medical emergency is real. Treat red as **a confirmed rule condition requiring review**, not proof of real danger.

## Starting

```powershell
cd "C:\Users\punee\Desktop\MY-PROJECTS\COMPUTER_VISION-V2\computer vision\VisionShield"
.\.venv\Scripts\python.exe -E run.py --remote
```

Open http://localhost:8000. One running server is enough for laptop and phone testing. If it is already running there, use it. Stop a previous terminal launch with Ctrl+C before restarting on the same port. The remote tunnel takes a few seconds to become ready; open Connect phone after readiness. Restarting creates a new private pairing link, so scan again.

Laptop-only mode: use `run.py` without `--remote`. Speed mode: use `run.py --remote --fast` for YOLO11n at 416 px. The default is YOLO11s at 640 px for better small-object detail, with a CPU speed cost. Do not claim 30 FPS.

All required packages and both weight files are installed. For a fresh setup, run `setup.ps1`. No Gemini key is needed for the local system. Without one, incident briefs are offline text templates; no image-understanding model has reviewed the thumbnail.

## Laptop demo

1. Choose Default Webcam (0). If absent, try Secondary Webcam (1). Close other applications using the camera. An unavailable camera should show offline rather than fabricated detections.
2. Keep the camera fixed and use good lighting. Show a person walking outside any zone. The box should be green, and there should be no zone alert. Initially there are no zones.
3. Create a restricted floor zone as described below. Step inside and remain for at least two seconds and three processed observations. Expect amber while entry is pending, then a red zone/person warning and one history entry.
4. Step outside. Live zone/person colours should clear on a fresh processed frame. The old event and last incident remain in history with their timestamp; this does not mean the danger is still present.
5. For the bag scenario, let the camera see a person and a bag close together first. Put the bag down, allow it to settle, then walk away while the bag stays visible. After six stationary/separated seconds, expect an unattended-bag candidate. Return near the bag to clear the live condition. Ownership uses image-pixel proximity, so stage it close enough to establish the association; it is not identity verification or a calibrated three-metre measurement.
6. Keep the siren OFF during setup; enable it only when ready. Only high-severity event notifications trigger it. Loitering alerts are disabled by default because camera/box jitter and ordinary movement can look like pacing. Possible-fall logic uses bounding-box change and downward displacement, not a medically validated fall model. Do not perform an actual fall for a demo.

## Mobile demo across networks

1. Run with `--remote`. On the laptop dashboard choose Connect phone and scan the private QR code.
2. Open the HTTPS link in a full mobile browser, allow camera access, choose the rear camera, and tap Start streaming. No app installation is needed.
3. Detail mode requests up to 960 px for small objects. Fast mode uses 640 px and less bandwidth. Keep the page foregrounded and screen awake. The browser wake-lock request is best effort.
4. Confirm the dashboard source changes to Mobile Phone Stream and the camera shows online. Repeat the same person, bag and zone tests. Both sources use the same server-side model and rules.
5. To prove different-network operation, turn Wi-Fi off on the phone and use mobile data. The laptop must stay online. The tunnel routes traffic over the internet; it is not a direct LAN connection and its latency affects FPS.
6. Mount the phone firmly before drawing zones. Redraw after moving, rotating or changing camera. Zones for webcam 0, webcam 1 and phone mode are stored separately, but all phone devices share the phone-mode configuration; re-pairing a different phone does not calibrate the scene automatically.
7. Changing the source on the dashboard now closes an old phone sender and releases its camera; it cannot silently take over the webcam again. Stop also releases camera tracks. Only one phone sender is accepted at a time.

## Drawing useful zones

A zone is a 2D polygon in the camera image. It defines your access policy; the dataset does not decide which part of a room is restricted.

- Choose an actual staff-only doorway, storage area or equipment boundary. Keep normal walkways, public seating and the whole room outside it.
- Wait for a live frame to be visible. Name the zone, click Draw Restricted Zone, then click 3-20 vertices in perimeter order. Click Save zone. Each save adds a zone; Remove saved zones clears all zones for that camera source. Escape cancels an unfinished drawing.
- Draw around the visible **floor footprint**, because entry uses a person's bounding-box bottom-centre (approximate foot position), not their head or box centre. A person's arm overlapping the outline should not by itself count as entry.
- Use non-crossing edges and a meaningful area. Duplicate vertices, collapsed shapes and crossed edges are rejected. Up to 20 zones can be saved.
- Idle zones use a neutral blue outline. A person inside is amber while confirming. A HIGH zone becomes red only after at least 1.5 seconds and three consecutive observed frames; a missing observation or long gap restarts confirmation. Leaving clears the live warning. Bags do not trigger person-entry alerts.
- Coordinates are normalized by image dimensions. These are perspective image boundaries, not calibrated distances in metres. Occluded feet, moving cameras and tracking errors can still cause mistakes.

## Knife detection: exact capability

The app now enables the pretrained **knife** class (YOLO's zero-based COCO class index 43). Previously the application filter excluded it. All 80 pretrained COCO classes are now displayed, including ordinary objects such as cups, phones and chairs. Person, bag and knife rules determine security warnings; other detections remain normal observations. Firearms and arbitrary dangerous objects are not supported by this model's enabled classes.

At confidence >= 0.40, an object can be drawn as a knife detection. A knife candidate needs confidence >= 0.55 for warning evidence. It is amber while verifying. A HIGH POSSIBLE_WEAPON alert requires the same tracked knife in at least three observations over at least one second, near a visible person's bounding box. The spatial gate is the person's box expanded by 12% of its longest side. It does not locate a hand, establish that someone holds the item, or classify intent. An isolated knife stays an amber candidate; low-confidence or one-frame detections do not escalate. Absence resets confirmation.

For a safe presentation, show a labelled dataset photo or a prerecorded controlled example rather than wave a sharp object around. A photo demonstration tests the detector; it does not prove physical weapon or dangerous-action recognition. A toy prop may be missed because it differs from the training images. Small, hidden, blurred, edge-on or distant blades can be missed even with good lighting.

## Dataset, training and weights: what to tell judges

**Accurate presentation statement:**

> We use official Ultralytics YOLO11s detection weights pretrained on MS COCO 2017. The network supplies object boxes and class confidences. Our contribution is phone ingestion, temporal tracking, person-bag association, operator-defined zones, confirmed warning rules, incident dispatch and the dashboard. We have not trained a custom weapon, violence or fall model. Knife detection is a candidate signal with documented misses, not certified threat recognition.

COCO has 80 object categories; its standard 2017 split has 118,287 training images and 5,000 validation images. YOLO detection learns boxes and class labels from annotated images. At runtime the `.pt` file supplies learned parameters; live webcam/mobile frames are inference inputs and do not retrain it. Zone entry and bag separation are programmed temporal/geometric rules, with no separate training dataset. The model was not trained on this project's demo footage.

References: [official YOLO11 models](https://docs.ultralytics.com/models/yolo11/), [COCO dataset/configuration](https://docs.ultralytics.com/datasets/detect/coco/), [COCO terms and image rights](https://cocodataset.org/#termsofuse).

The exact checkpoint metadata, SHA-256 hashes, source URLs and enabled class IDs are recorded in `artifacts/model-card.json`. Checkpoints record a COCO config, 640 px training size and a configured 600 epochs; that describes the supplied checkpoint metadata, not training performed by our team.

We downloaded official COCO validation labels and twenty images: the first ten knife-positive IDs and first ten knife-negative IDs, containing sixteen annotated knife instances. This is a deliberately small, selected development check, not an independent deployment benchmark. Predictions are matched one-to-one against boxes at IoU >= 0.5.

| Model at 640 px | Threshold | Matched knives | Unmatched knife predictions | Missed knives |
|---|---:|---:|---:|---:|
| YOLO11n | 0.40 | 3 | 1 | 13 |
| YOLO11s | 0.40 | 7 | 3 | 9 |
| YOLO11s | 0.55 warning gate | 6 | 3 | 10 |

The warning gate additionally uses time and person proximity; these still-image counts do not measure complete video alert precision. This result motivated using the larger model, but does not establish reliable knife recognition. Full outputs are in `artifacts/knife-validation.json` and `artifacts/knife-validation-yolo11s.json`.

Reproduce the check:

```powershell
.\.venv\Scripts\python.exe -E scripts/prepare_validation.py
.\.venv\Scripts\python.exe -E scripts/evaluate_detector.py --model yolo11s.pt --output artifacts/knife-validation-yolo11s.json
```

The downloaded images/labels are in `datasets/coco_val_sample` and their source manifest is saved there. They are evaluation inputs, not newly trained data. COCO annotations and individual image rights should be respected; do not redistribute images as if this team owns them.

## What a serious weapon-specific training stage needs

Before calling this reliable security detection, collect/curate and box-label relevant camera scenes: visible and partly occluded knives, varied sizes, illumination, viewpoints and backgrounds. Include negative examples such as phones, pens, tools, scissors, empty hands and ordinary kitchen scenes. Separate knife-object detection from the contextual question of dangerous use.

Split by recording session/location/device, for example 70/20/10 train/validation/test, keeping adjacent frames and near-duplicates in the same split. Fine-tune a pretrained model, select thresholds on validation data, and evaluate on an untouched test set. Report per-class precision, recall, mAP50/mAP50-95 and video false alerts per hour, with a special count of missed small/occluded objects. Do not train on the twenty-image evaluation sample and call that evidence of generalization.

`security-dataset.template.yaml` is a schema example in `scripts/`, not a populated dataset. An optional fine-tuning entrypoint is available once reviewed annotations exist:

```powershell
.\.venv\Scripts\python.exe -E scripts/train_detector.py --data path/to/your/data.yaml --epochs 50 --device cpu
```

CPU fine-tuning can be slow; no custom training was silently launched. To use a genuinely fine-tuned checkpoint later, set `$env:YOLO_MODEL` to its `best.pt` path before launch. Classes are selected by name, so retain person/backpack/handbag/suitcase/knife labels for those modules.


## Tracking and latency revision

The app is left stopped after the requested shutdown. Start it yourself with the launch command above when you want to test.

The preview is now a separate latest-frame JPEG worker, capped at 24 FPS, rather than a video produced once per inference. Bounding boxes, zones and warning labels are drawn by a browser canvas from small age-limited telemetry snapshots. Stale boxes are hidden, and a short motion prediction helps bridge inference updates. Video/AI FPS are displayed separately; a smooth preview is not a claim of 24 neural inferences per second.

The installed float32 ONNX graph was exported from the same official YOLO11s weights, without quantization or retraining. Runtime checks its graph and source-weight hashes. ONNX Runtime CPU is selected automatically when the matching optimized graph exists, with PyTorch fallback. `scripts/optimize_model.py` rebuilds the graph when weights change; setup.ps1 runs it on a fresh setup. The graph manifest is yolo11s.onnx.json. Intel GPU acceleration was trialled but dynamic inference stalled, so it is not enabled or required.

Offline sustained check on this machine: 20.4 preview FPS while inference ran at 5.6 FPS; the last inference took 159.8 ms. A separate 20-image alternating backend comparison measured median 241.1 ms for PyTorch and 208.4 ms for ONNX, with 111/111 boxes matched at IoU >0.98. These are measured test workloads, not guaranteed phone/network performance.

Phone upload now defaults to 640 px Live mode and allows at most three unacknowledged frames. Congestion skips fresh captures instead of building a backlog. A synthetic 500 ms round-trip test sent seven frames in about 1.1 seconds with a maximum window of three. Detail mode remains available at 960 px. Network upload rate, preview rate and inference rate are different measurements.

Tracking uses class-aware motion prediction, box overlap and a small torso-colour signature. Missing detections do not immediately create new IDs. Person tracks can be reacquired for up to 30 seconds when appearance is a strong, unambiguous match, including a short exit and return. The cache is bounded to 128 tracks and is cleared when the source changes. Torso signatures are held only in process memory.

The UI calls these TRACK IDs. This is not face recognition or verified identity: lighting, occlusion, similar clothing, a changed outfit, a long absence or switching cameras can cause an ID change or error. Ambiguous appearance matches prefer a new ID over falsely claiming someone is the same person. Rehearse a short exit/reentry and two-person crossing under the actual demo lighting.

Recheck offline without opening localhost:

```powershell
.\.venv\Scripts\python.exe -E -m unittest discover -s tests -v
node tests/mobile_transport_test.cjs
.\.venv\Scripts\python.exe -E scripts/benchmark_preview.py
```

The revised UI/transport has been checked with syntax, ASGI, simulated-latency and pipeline tests while keeping localhost stopped. A fresh real-browser/mobile acceptance check remains for the next user-started demo.
