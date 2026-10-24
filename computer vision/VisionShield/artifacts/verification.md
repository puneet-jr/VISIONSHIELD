# Verification summary — 24 October 2026

The current detailed results, scope and limitations are in [ENGINEERING_REPORT.md](../ENGINEERING_REPORT.md). This summary replaces earlier revision notes.

- Dependency check, Python compilation, Ruff formatting/lint and JavaScript syntax passed.
- 42 Python regression tests passed, including security roles, temporal gates, identity uncertainty, camera recovery, model failure and evidence lifecycle.
- Desktop/mobile Chromium: zones, authenticated preview/events, event filtering/evidence/review/deletion, simulated phone streaming and recorded mode passed with no runtime exceptions or horizontal overflow. Raw measurements: [browser-verification.json](browser-verification.json).
- Real public HTTPS/WSS tunnel: unpaired access rejected, camera-only privileges enforced, operator access accepted and ten JPEG frames acknowledged. [remote-verification.json](remote-verification.json).
- Offline 640x480 isolation: 21.6 preview FPS, 5.2 inference FPS, final inference 186.2 ms. These are CPU/workload-specific measurements, not guarantees.
- Final browser run used synthetic 320x240 uploads: 12.1 preview FPS, 9.33 average displayed FPS, 5.2 inference FPS, last inference 187.7 ms; 59.09% inference skips intentionally avoid backlog. Receive-to-display estimate 9.3 ms with ±5.5 ms clock uncertainty excludes physical sensor and synthetic HTTP upload delay.
- A real one-epoch YOLO11n smoke experiment completed on 8/4/4 local images. It did not consistently improve its baseline and was not promoted. [training-smoke.json](training-smoke.json).
- Actual deployed-model four-image smoke check: knife TP=1/FP=0/FN=1; person TP=3/FP=3/FN=5. Tiny selected COCO results do not establish security accuracy. [detector-heldout.json](detector-heldout.json).
- Tracking and event metrics are explicitly SIMULATED fixture checks: [tracking-evaluation.json](tracking-evaluation.json), [event-evaluation.json](event-evaluation.json). No real-world accuracy or camera-hour claim follows.
- Physical phone permissions/cellular operation, real webcam scenarios, sensor-to-display latency and external Gemini remain manual/untested acceptance items.
- Verification servers and tunnels are stopped after checks. Start manually using [DEMO_GUIDE.md](../DEMO_GUIDE.md).
