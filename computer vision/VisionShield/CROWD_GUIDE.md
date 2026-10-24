# Crowd operations demo

Start from the VisionShield directory:

```powershell
.\.venv\Scripts\python.exe -E run.py --remote
```

Open **http://localhost:8000/guard** for the camera overview. The main dashboard at `/` retains webcam/recording selection, detailed tracking and event review. Ctrl+C stops the server and temporary tunnel.

HTTPS tunnelling is now enabled by default with `run.py`; `--remote` remains compatible and `--local` explicitly disables it. Pairing never issues an HTTP localhost QR: it waits for the public HTTPS origin. Use the private `/phone#token=...` URL from Connect phone on every mobile. All cameras appear in **Crowd camera overview**, including phones connected while Webcam 0 is selected. Switch front/rear cameras directly in the phone selector; the stream restarts. Laptop camera opening tries DirectShow and falls back to Media Foundation on Windows.

New, unconfigured camera scenes receive an observed-count review zone with a default threshold of 20 visible people. Green is monitoring; red requires a confirmed configured crowd rule. Set your own capacity using Zone settings. This default is a review threshold, not a declaration that 20 people is safe. Explicitly saved zone settings, including an intentionally empty list, take precedence.

## Laptop, phones and guards

1. For a laptop plus phones, select Webcam 0 in the main dashboard first. Then use Connect phone → **Camera upload only** for each phone. Every paired browser gets its own camera context. If the main source is waiting for a phone, the first phone uses that slot. The default limit is four camera slots including the main source.
2. Keep each phone fixed, rear camera selected, page foregrounded and screen awake. Start streaming on every phone. Phones can use cellular data or different Wi-Fi networks through the HTTPS tunnel. Both phone and laptop need internet. Remote traffic passes through Cloudflare.
3. Open `/guard` on the laptop to see all streams. To give guards viewing access, select **Guard monitoring & acknowledge** in the pairing dialog. Their separate QR permits viewing cameras, evidence and acknowledging events. It cannot change sources, zones or delete events. Treat all pairing links as private.
4. A second sender from the **same paired browser** is rejected; different paired browsers can stream simultaneously. An idle additional camera slot is reclaimed after two minutes when another device connects. Changing the main source affects only its sender. Restarting the server changes generated tokens; pair again.
5. Camera location is installation coordinates entered by the operator, or GPS explicitly shared by that phone. It is not the laptop’s location. Camera tracks and anonymous IDs remain separate; camera counts are never added as unique attendance.

One serial inference worker shares a model across cameras. Each camera has its own latest-frame slot, JPEG preview, tracker, identity memory, zone state and crowd history. Preview is independent of inference; no inference task is launched per viewer. More cameras reduce analysis FPS. Pose enrichment runs only on the main slot to keep the additional crowd streams lighter. Physical phones still require a hardware/browser rehearsal.

## Configure and test zones

Use **Zone settings** on a camera card in `/guard` while logged in as the operator. The camera-specific form replaces that camera’s zones with one polygon; use the main dashboard drawing editor to retain multiple zones on the main source. Coordinates are normalized image coordinates: `(0,0)` top-left, `(1,1)` bottom-right. Polygons and calibration must be redrawn after moving the camera.

| Purpose | Rule | Rehearsal |
|---|---|---|
| Crowd occupancy | Visible people **>= capacity**, continuously for at least 2 seconds and 3 observations | Set capacity 2, show two people with feet inside the polygon; step out to clear the live warning. |
| River-edge crowd zone | The same configured capacity/density rules, labelled for that sensitive area | Draw only the monitored safe-standing ground footprint and supply site-specific limits. The system does not locate water or select a safe river distance. |
| Pedestrians only | A detected car, truck, bus, bicycle, motorcycle or train stays inside for 3 seconds and 3 observations | Use a fixed-camera recording with a supported vehicle. An ordinary person alone is not a vehicle intrusion. |
| Restricted | Any supported person/animal footpoint inside, immediate warning under the current default | Reserve for an area where **every** detected entry merits review. Do not use it for normal public crowd areas. |
| Bag monitoring (main dashboard) | Stationary separated bag candidate, default 6 seconds | Rehearse association and separation as described in DEMO_GUIDE. |

Crowd membership uses the detected **bottom-centre footpoint**. Supply a measured area to display **observed people/m²**. A density warning requires your explicit site threshold. No universal “safe” capacity is invented. The number of detected people is a potentially substantial undercount, especially with occlusion.

Metre spacing additionally requires four ordered image corners of an actual flat rectangular ground area, plus measured width and depth. Enter the corners around its perimeter, corresponding to `(0,0), (width,0), (width,depth), (0,depth)`. The homography projects visible footpoints onto that rectangle. Supply both dimensions; a conflicting area, crossed/concave polygon or missing calibration is rejected. Without this calibration distances stay in **pixels**. Mean nearest-neighbour spacing is an observed average, not the closest possible pair or physical pressure. The camera’s older scalar speed calibration is a separate, approximate setting.

Rapid buildup requires a complete approximately 10-second observation window, **>30% count growth and at least 5 additional visible people**, followed by the normal 2-second/3-observation confirmation. Gaps over 2 seconds reset the observation window. Image-speed and flow-alignment metrics are descriptive. Coherent movement alone never creates a stampede warning. The system has **no validated stampede, crowd-pressure or crush prediction model**. Guards verify evidence and follow the site’s existing management plan.

## Datasets actually used

Acquisition and inspection are reproducible:

```powershell
.\.venv\Scripts\python.exe -E scripts/crowd_datasets.py all
.\.venv\Scripts\python.exe -E scripts/crowd_source_catalog.py
.\.venv\Scripts\python.exe -E scripts/evaluate_crowd.py
```

- [Mall](https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html): all 2,000 frames and head-point/count annotations downloaded, checksum recorded. Used for a **frozen-model** 24-frame counting check. Head points are not full-body YOLO boxes. Research/noncommercial terms apply; cite the authors.
- [VSCrowd](https://github.com/HopLee6/VSCrowd-Dataset): annotation archive downloaded from the publisher’s Hugging Face mirror and inspected. Annotations include head boxes, points and tracking IDs. Separate multipart videos total approximately 19 GB and were not downloaded. These annotations cannot train a detector without matching frames, and head boxes must not be silently renamed full-body boxes.
- [Hajj/Umrah](https://www.kaggle.com/datasets/ziya07/hajj-and-umrah-crowd-management-dataset): downloaded and inspected. It contains a CSV with crowd/context variables, **no detector images or bounding boxes**. It is not used to train object detection or validate danger prediction.
- The other six supplied sources are recorded with access, format and licensing details in `artifacts/crowd-dataset-catalog.json`, also visible in the guard dashboard. Roboflow exports were not obtained; UniDataPro’s public preview does not provide its full advertised annotations. The UCF legacy crowd link could not be acquired. Kaggle scene/counting archives are not claimed as downloaded or trained.

The deployed detector is **YOLO11s, vendor COCO-pretrained**, with all 80 classes enabled. No weights trained on these supplied crowd sources have been promoted. The Mall sample returned **MAE 18.875**, **RMSE 20.026**, **bias −18.875 people/frame**. This is a limited single-scene diagnostic, not a published benchmark or proof of dependable dense-crowd counting. The head-count/full-body definitions also differ. The result demonstrates why this implementation is a review-oriented prototype rather than a life-safety capacity meter.

`recording:Mall-counting-research.avi` is available in the source selector. It is an explicitly labelled slideshow of sampled real frames at artificial timing. Use it to inspect counting/overlays, **not** to measure walking speed, tracking accuracy or buildup timing. Dense-crowd deployment still needs a dedicated head/density model, properly licensed images, scene/sequence-separated training and independent validation. The existing research audit/train tools support detector experiments; count-point data needs an appropriate counting-model pipeline rather than fabricated boxes.

## Verification

```powershell
.\.venv\Scripts\python.exe -E -m unittest discover -s tests -v
node tests/mobile_transport_test.cjs
# Against a running --remote instance; uses a photo and blank fixture, not physical phones:
.\.venv\Scripts\python.exe -E tests/crowd_remote_smoke.py
```

The current 67-test suite covers crowd timing, exact capacity threshold, clearing, calibration, growth, vehicle intrusion, independent camera contexts and guard permissions, alongside previous regressions. Actual HTTPS/WSS verification exercised two simultaneous streams, separate GPS, protected guard preview and a real detector-triggered capacity event with guard acknowledgement. See `artifacts/crowd-remote-verification.json` and `artifacts/crowd-counting-evaluation.json`. Automated transport fixtures do not replace a live-device rehearsal.

## Clean crowd overlay

The dashboard starts in **Crowd dots** mode: green person-position estimates and configured shaded warning polygons, without clothing, confidence, IDs, skeletons or per-object labels obscuring the video. Switch to **Detailed tracks** to restore the earlier overlay, or select an observation in the list for its attributes. The guard overview uses the same dot presentation. Red shading requires a confirmed configured warning; it is not inferred from a screenshot or drawn over every busy scene.

Experimental learned head maps are available through `ENABLE_HEAD_MAP=true` after `scripts/setup_head_model.py`. They remain **off by default**: on the same 24 Mall frames, publisher pretrained 11n-head had MAE 26.875, v8s-head 25.833, and tiled v8s-head 30.000, versus body baseline 18.875. They do not improve this diagnostic and have not been promoted for counting or alert thresholds. The publishers report approximately 15000 training images but do not identify datasets/splits. A separate JHU-trained candidate had an unavailable weight URL. No supplied crowd dataset training is claimed. Actual model head centres, when explicitly enabled, are labelled separately from approximate person-position dots. Head points do not provide metre distances or verified identity. See `artifacts/head-counting-evaluation.json` and `artifacts/head-model-provenance.json`.
