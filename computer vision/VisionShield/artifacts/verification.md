# Final verification

- All project dependencies are installed in .venv. `python -E -m pip check`: no broken requirements.
- 13 unittest checks passed, including bounded worker event fanout, pairing/input validation, camera WebSocket acknowledgement, duplicate inference prevention, background report dispatch, temporal knife gates, class-name knife inclusion, person-only footpoint zone entry, source-specific zone persistence, malformed polygon rejection, and clearing red state after zone exit.
- Python compilation and JavaScript syntax checks passed.
- Final headless Chromium flow passed 1440 px desktop and 390 px mobile layout checks, event-hub connection, zone drawing/saving/removal, and sustained mobile sender upload. No JavaScript runtime exceptions or horizontal overflow. A synthetic feed was used for final screenshots. A separate first-frame mapping edge case was caught and fixed: drawing is blocked until the live frame loads.
- The simulated mobile sender sustained 8.6 uploads/s in the final local check. This is upload rate, not detector throughput, and the synthetic camera was 640x480.
- Default running model is official YOLO11s at 640 px; enabled classes are person, backpack, handbag, suitcase and knife. Nano at 416 px remains available via --fast.
- Small selected COCO val2017 check: 20 images, 16 labelled knives. At IoU >=0.5 and confidence >=0.4, nano matched 3 knives with 1 unmatched prediction; small matched 7 with 3 unmatched predictions. At the 0.55 warning gate, small matched 6 with 3 unmatched predictions. These results show substantial misses and do not establish reliable security detection. Person/time gates are separate rule tests.
- Median detector-call time over the last 19 sample images was 117.2 ms for nano and 279.1 ms for small, at 640 px. Inference time depends on scene and load; it is separate from camera capture and upload FPS.
- Official checkpoint provenance, hashes and recorded training metadata are in model-card.json. No custom model training has been performed.
- Public remote transport has been tested after explicit approval: unpaired API access rejected (401), pairing and authenticated access succeeded (200), ten camera frames acknowledged through public WSS, and local pipeline source reported phone_upload online. The physical phone's browser permissions and cellular scene test still need user acceptance.
- Idle zones start empty, use neutral outlines, persist separately per source and require >=1.5 seconds plus >=3 observed person frames for a confirmed breach. Knife warnings require >=0.55 confidence, >=3 observations over >=1 second and person-box proximity for HIGH severity. Red means a rule condition needing human review, not verified danger or intent.
- Loitering alerts are disabled by default to avoid treating ordinary motion/box jitter as suspicious. Historical incident briefs are explicitly archived. Offline report templates use review-oriented language.
- Gemini service calls were not tested: no API key was supplied. Offline report delivery and nonblocking dispatch were regression tested. Real-world fall/weapon accuracy is not certified.


## Performance and tracking revision (latest)

Localhost and the tunnel were stopped at the user's request and have not been restarted. Revised checks use in-process ASGI clients, synthetic scenes, offline model inference and a Node simulated transport; no live-browser or physical-phone performance claim is made for this revision.

29 Python tests pass, including short-exit ID recovery, missed detections, appearance-assisted crossing, ambiguous-match rejection, time-based expiry, bounded tracking state, independent preview, nonblocking camera opening, source-switch shutdown, stale telemetry rejection and late-frame generation protection. JavaScript syntax and the simulated transport test pass. Dependency check reports no broken requirements.

Offline pipeline benchmark: preview 20.4 FPS independently of 5.6 FPS inference, with 159.8 ms on the final inference cycle. Alternating sample benchmark: PyTorch median 241.1 ms versus ONNX CPU 208.4 ms, with 111/111 boxes matched above IoU 0.98. See preview-benchmark.json and backend-benchmark.json. These are test measurements, not guaranteed network/physical-camera rates.

The mobile sender test used 500 ms simulated RTT and sent seven frames in about 1.1 seconds, with at most three unacknowledged frames. Camera tracks were released on a source-change notification. Preview rendering is now independent of detection, with age-limited browser box overlays. All 80 COCO classes are displayed; security rules remain specific to relevant classes. IDs are session tracks assisted by torso colour, not verified identity; reacquisition is bounded to 30 seconds and may fail or be ambiguous. Intel GPU inference stalled in a trial and is not enabled; the experimental runtime was removed.
