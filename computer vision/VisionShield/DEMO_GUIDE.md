# Demo guide

**Crowd monitoring update:** See [CROWD_GUIDE.md](CROWD_GUIDE.md) for multiple cameras, guard QR access, per-camera capacity/density rules, calibration, datasets and measured counting limitations. Open `/guard` for the new camera overview.

Software checks pass, but a physical phone/webcam rehearsal is still necessary on your devices. Red means an elevated review condition under configured rules. The model cannot verify real danger, intent, authorization, ownership or hand contact.

## Start

~~~powershell
cd "C:\Users\punee\Desktop\MY-PROJECTS\COMPUTER_VISION-V2\computer vision\VisionShield"
.\.venv\Scripts\python.exe -E run.py --remote
~~~

Open http://localhost:8000. Initial state waits for a camera. Select Webcam 0 (try 1 if needed), or wait for the tunnel ready message and choose Connect phone. Do not launch a second server on the same port. Ctrl+C stops the server and tunnel. Add --fast to trade small-object detail for speed.

## Laptop and zone rehearsal

1. Fix the camera, use good lighting and close other apps using the webcam. Walk outside configured zones. Ordinary observations should remain green.
2. Choose a restricted doorway or floor footprint. Name the zone, select **Restricted floor area** and click Draw restricted zone. Click 3–20 points in boundary order, then Save. Dwell is disabled in the default immediate-warning mode. Escape cancels drawing. Remove saved zones clears the selected source's zones.
3. Draw where feet stand: the rule uses the box bottom-centre, not an arm/head overlap. Keep public walkways outside. Crossed edges and degenerate shapes are rejected.
4. A detected person or supported animal entering the zone turns red immediately and produces **RESTRICTED ZONE — CHECK IMMEDIATELY**. This means the configured zone condition was observed, not verified intent. Exit: live warnings clear on a fresh inference; history remains timestamped. ZONE_IMMEDIATE_WARNING=false restores dwell confirmation when desired.
5. For bags, create a **Bag monitoring area** around the bag location. Show the person close to the bag for >=1 second and three observations, put the bag down, let it settle and walk away. After six stationary seconds without a nearby associated person, expect a MEDIUM unattended-object candidate. Return nearby to clear it. An unassociated bag can also become a stationary candidate inside the monitor zone, explicitly without ownership evidence. A bag outside this zone does not trigger the rule.
6. Open an event to inspect evidence, rules, confidence, persistence and uncertainty. Acknowledge it, mark a false positive, filter by review status, or delete it and its evidence. Leave the siren off during setup; historical event loading does not trigger it.

Zones are normalized image polygons representing operator policy, not learned restrictions or distances in metres. They are saved separately for webcam 0, webcam 1, phone mode and each recording. Phone zones are now scoped to the paired browser/device. Redraw after changing camera angle, orientation or scene. Letterboxing is excluded from drawing coordinates. Occluded feet, camera movement and detector errors remain limitations.

## Phone on mobile data

1. Run with --remote. In Connect phone choose **Camera upload only**, then scan the private QR in a full mobile browser.
2. Allow camera permission, select the rear camera and tap Start streaming. This link permits upload only. For monitoring/evidence access, issue a separate **Guard monitoring & acknowledge** link; use **Operator** only for administrative controls.
3. Keep the page foregrounded and screen awake. Fast requests up to 640 pixels; Detail requests up to 960. Wake lock is best effort. Stop releases the camera. Switching source on the dashboard also stops the old sender.
4. Turn phone Wi-Fi off to demonstrate cellular operation. Both devices need internet; the laptop must keep running. Repeat zone/bag tests after fixing the phone position.
5. Briefly interrupt and restore connectivity. The sender should reconnect or explain expired access/source change. A second sender from the same paired browser is rejected; other paired devices get independent slots up to the configured camera limit. Restarting the server invalidates default links: scan again.

Public HTTPS/WSS transport and camera/operator separation were tested through Cloudflare from this machine. Physical phone permissions, carrier behavior, browser suspension and hardware sensor delay were not verified automatically.

## Posture, colour, speed and camera position

Use a fixed camera with the full body visible, including hips, knees and ankles. Pose enrichment runs in one background worker every 0.75 seconds by default; allow a few seconds for the first model load and two consistent pose results. Sitting/standing are joint-angle estimates from COCO keypoints. Occluded legs, crouching, camera angle or low-confidence joints show **unknown** rather than a guessed label. Posture is not a trained action or behavior classifier.

The live overlay and Observed people/animals/objects table show pose, upper/lower clothing colours or object colour, speed and x/y velocity. Colours are estimates of visible crops, affected by background and lighting. Pixel velocity uses +x right and +y down. New or reacquired tracks show measuring until two suitable observations exist. Camera motion also creates image motion.

Open **Camera coordinates and speed calibration** beneath the video. Enter this camera's name and installation latitude/longitude, then Save. For a phone, tap **Share this camera's GPS** on the phone page after streaming starts and allow the device's permission prompt. The displayed location comes from that camera device; no server/IP geolocation is used. Denial leaves video working without coordinates. GPS accuracy is shown and location can be retained with event evidence. Tap again after moving. GPS is in server session memory; installation settings persist in data/cameras.json.

Physical speed is optional: measure a known reference-plane distance, divide metres by its pixel length and enter metres per pixel. Save while that camera's live reference frame is visible. The app then shows approximate m/s (km/h is also in telemetry); image aspect changes invalidate the scale. This simple scalar estimate is not perspective-correct and is unsuitable as certified vehicle speed. Redo calibration after moving or zooming the camera.

Select a person box/row for a compact observation summary. Gender is **not inferred from appearance**. The optional Man/Woman/Unspecified dropdown records only an operator-entered session label; it does not identify a person or verify gender.

All 10 pages of the supplied PHOTOS OF UI reference were visually reviewed. The UI now follows its class-coloured boxes (people green, vehicles yellow, warnings red), restricted-zone fills, foot markers, bag-proximity connectors in pixels, clothing/object colour swatches, live alert detail panel, summary counts and sampled occupancy trend. Counts and graphs use actual observed/persisted data. The crowd update adds a multi-camera overview and configured capacity alerts. Helmet/vest/hat compliance remains unsupported. Crowded scenes can be substantially undercounted; see the actual Mall evaluation in CROWD_GUIDE.md. No unsupported PPE result or fabricated camera count is displayed.

## Person IDs

The overlay shows a separate anonymous person ID and track ID. Short occlusions and supported reappearances retain the association; a new segment can retain the same person ID. Select a box for association details and open Model insights for history. Gap matches are labelled uncertain.

Similar clothing, poor crops, changed lighting and gaps beyond the default 120-second identity memory can require a new ID. Clothing cannot verify identity. There is no face recognition or cross-camera identity claim.

## Knife capability

All 80 pretrained COCO classes are enabled, including knife (index 43), person and the three bag classes. Webcam and phone use the same server detector. Any visible knife detection above the detector's 0.40 threshold gets a red DANGER warning immediately, explicitly marked unconfirmed. A saved HIGH knife event requires >=0.55 confidence and >=3 observations over >=1 second. A person need not be nearby by default. There is no hand-contact or intent model; visual danger wording can be a false alarm.

The selected 20-image development check contains 16 knives. YOLO11s matched 7 at 0.40, with 3 unmatched knife predictions; at 0.55 it matched 6, with 3 unmatched predictions. A separate four-image local test contains two knives: the deployed model matched one at 0.40. These tiny selected checks show misses, not reliable weapon detection. Small, hidden, edge-on, blurred or distant blades can be missed.

Use a labelled dataset photo or controlled recording for the presentation, identifying it as such. A photo tests recognition only. Do not stage dangerous handling or an actual fall. Firearms, arbitrary weapons and violent actions are unsupported.

## Recorded and deterministic checks

~~~powershell
.\.venv\Scripts\python.exe -E scripts/evaluate_tracking.py
.\.venv\Scripts\python.exe -E scripts/evaluate_events.py
~~~

The first creates data/recordings/SIMULATED-transport.avi. Refresh the dashboard and choose its RECORDED source. It verifies decoding, pacing, aspect mapping and actual model execution on a labelled rectangle video. Rectangles are not expected to be detected as people. Ground-truth boxes enter only the offline tracking evaluator, never live inference. The UI labels recorded mode and shows Recording ended at EOF.

For your own controlled recording, copy an .mp4/.avi/.mov/.mkv into data/recordings, refresh and select it. Real inference runs on decoded frames. Select the recording again to replay.

## What to tell judges

> The deployed detector is official Ultralytics YOLO11s pretrained on COCO 2017. It supplies boxes and confidences for 80 classes. Our application adds ingestion, tracking, anonymous segment association, spatial/temporal review rules and protected evidence. We completed an audited one-epoch YOLO11n fine-tuning smoke experiment on 8 training images, with 4 validation and 4 local test images. It did not consistently improve its baseline and was not deployed. We have not trained a violence, intent, hand-contact, fall or learned threat model.

See [ENGINEERING_REPORT.md](ENGINEERING_REPORT.md) for exact metrics, source links, licensing and commands. Do not present synthetic scores or tiny selected COCO checks as security accuracy.

## Troubleshooting

Use **Connection diagnostics** in Demo & model insights. DNS failure, unavailable HTTPS backend, authentication failure and missing remote configuration have separate codes. Unpaired requests return 401; camera access to operator data returns 403; forbidden Origin returns 403. Readiness can return 503 before the first source/model; /health/live independently verifies the process.

For camera denial, allow site permission in an HTTPS browser and close other camera apps. For offline video, keep the sender foregrounded, verify internet and select the correct source. Webcam capture retries after failure. If the tunnel exits, restart --remote and re-pair. For slow inference, inspect separate preview/AI FPS and stage timings, close CPU-heavy apps, or try --fast. Inference skips frames deliberately to avoid accumulating latency.
