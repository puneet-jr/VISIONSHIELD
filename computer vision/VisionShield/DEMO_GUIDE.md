# Demo guide

Software checks pass, but a physical phone/webcam rehearsal is still necessary on your devices. Red means an elevated review condition under configured rules. The model cannot verify real danger, intent, authorization, ownership or hand contact.

## Start

~~~powershell
cd "C:\Users\punee\Desktop\MY-PROJECTS\COMPUTER_VISION-V2\computer vision\VisionShield"
.\.venv\Scripts\python.exe -E run.py --remote
~~~

Open http://localhost:8000. Initial state waits for a camera. Select Webcam 0 (try 1 if needed), or wait for the tunnel ready message and choose Connect phone. Do not launch a second server on the same port. Ctrl+C stops the server and tunnel. Add --fast to trade small-object detail for speed.

## Laptop and zone rehearsal

1. Fix the camera, use good lighting and close other apps using the webcam. Walk outside configured zones. Ordinary observations should remain green.
2. Choose a restricted doorway or floor footprint. Name the zone, select **Restricted floor area**, set dwell to 1.5 seconds and click Draw restricted zone. Click 3–20 points in boundary order, then Save. Escape cancels drawing. Remove saved zones clears the selected source's zones.
3. Draw where feet stand: the rule uses the box bottom-centre, not an arm/head overlap. Keep public walkways outside. Crossed edges and degenerate shapes are rejected.
4. Enter briefly and leave: entry remains pending/amber. Stay for at least two seconds and three processed observations: the zone/person become red and a review event appears. Exit: live warnings clear on a fresh inference. History remains timestamped.
5. For bags, create a **Bag monitoring area** around the bag location. Show the person close to the bag for >=1 second and three observations, put the bag down, let it settle and walk away. After six stationary seconds without a nearby associated person, expect a MEDIUM unattended-object candidate. Return nearby to clear it. An unassociated bag can also become a stationary candidate inside the monitor zone, explicitly without ownership evidence. A bag outside this zone does not trigger the rule.
6. Open an event to inspect evidence, rules, confidence, persistence and uncertainty. Acknowledge it, mark a false positive, filter by review status, or delete it and its evidence. Leave the siren off during setup; historical event loading does not trigger it.

Zones are normalized image polygons representing operator policy, not learned restrictions or distances in metres. They are saved separately for webcam 0, webcam 1, phone mode and each recording. Different phones currently share the phone-mode zone configuration: redraw after changing phone, camera angle, orientation or scene. Letterboxing is excluded from drawing coordinates. Occluded feet, camera movement and detector errors remain limitations.

## Phone on mobile data

1. Run with --remote. In Connect phone choose **Camera upload only**, then scan the private QR in a full mobile browser.
2. Allow camera permission, select the rear camera and tap Start streaming. This link permits upload only. For dashboard/evidence access on another browser, issue a separate **Operator** link.
3. Keep the page foregrounded and screen awake. Fast requests up to 640 pixels; Detail requests up to 960. Wake lock is best effort. Stop releases the camera. Switching source on the dashboard also stops the old sender.
4. Turn phone Wi-Fi off to demonstrate cellular operation. Both devices need internet; the laptop must keep running. Repeat zone/bag tests after fixing the phone position.
5. Briefly interrupt and restore connectivity. The sender should reconnect or explain expired access/source change. A second simultaneous sender is rejected. Restarting the server invalidates default links: scan again.

Public HTTPS/WSS transport and camera/operator separation were tested through Cloudflare from this machine. Physical phone permissions, carrier behavior, browser suspension and hardware sensor delay were not verified automatically.

## Person IDs

The overlay shows a separate anonymous person ID and track ID. Short occlusions and supported reappearances retain the association; a new segment can retain the same person ID. Select a box for association details and open Model insights for history. Gap matches are labelled uncertain.

Similar clothing, poor crops, changed lighting and gaps beyond the default 120-second identity memory can require a new ID. Clothing cannot verify identity. There is no face recognition or cross-camera identity claim.

## Knife capability

All 80 pretrained COCO classes are enabled, including knife (index 43), person and the three bag classes. Webcam and phone use the same server detector. Objects above 0.40 can be drawn. Warning evidence needs >=0.55; HIGH review requires the same tracked knife in >=3 observations across >=1 second near a visible person box (12% expansion). There is no hand-pose or intent model.

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
