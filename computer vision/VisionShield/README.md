# VisionShield

A single-process surveillance research prototype: real YOLO object detection, browser phone ingestion, anonymous motion/appearance tracking, operator-defined zones, temporal review rules and protected event evidence. Detections can be wrong and require human review. Red means a confirmed configured rule, not proof of real danger.

Read [DEMO_GUIDE.md](DEMO_GUIDE.md) for rehearsal and [ENGINEERING_REPORT.md](ENGINEERING_REPORT.md) for architecture, root causes, datasets, training, measurements and limitations.

## Install and run on Windows

Python 3.12 is used. From this project directory:

~~~powershell
.\setup.ps1
.\.venv\Scripts\python.exe -E run.py --remote
~~~

Open http://localhost:8000. In **Connect phone**, select **Camera upload only**, scan the private QR, allow camera permission and tap Start streaming. The phone can use cellular or another Wi-Fi network. Both devices need internet and the laptop must stay running. A separate **Operator** link grants dashboard and evidence access.

For laptop-only webcam use:

~~~powershell
.\.venv\Scripts\python.exe -E run.py --source 0
~~~

Default inference is official YOLO11s at 640 pixels, with its verified float32 ONNX export when available. Add --fast for YOLO11n at 416 pixels; small-object recall may decrease. Default source phone_upload waits for a real sender. Live mode never substitutes fabricated detections. Ctrl+C stops the server and tunnel.

Use -E because this machine has a global PYTHONPATH that otherwise mixes environments. requirements.lock.txt records installed versions; requirements.txt declares ranges. No frontend build is required: the UI is plain HTML/CSS/JavaScript.

## Configuration and access

Settings in [app/config.py](app/config.py) come from environment variables; [.env.example](.env.example) documents them but is not loaded automatically. VISIONSHIELD_ prefixes are supported and take precedence. Use VISIONSHIELD_DEBUG to avoid conflict with IDE tooling.

Demo mode trusts requests only when both Host and peer are local/private. Public requests require pairing; the bundled tunnel forces a nonlocal upstream Host. Camera access permits upload/status/time only. Operator access protects preview, telemetry, evidence and controls. Pairing is rate limited; cookies are HttpOnly, SameSite=Strict and Secure over HTTPS. Secrets arrive in URL fragments, are exchanged for cookies and removed from the address bar; default secrets rotate on restart.

For a stable HTTPS reverse proxy, set PUBLIC_BASE_URL, DEPLOYMENT_MODE=production and an explicit random ACCESS_TOKEN of at least 32 characters. Production disables the local bypass. Keep the origin inaccessible directly from the public internet, forward WebSocket upgrades and provide TLS at the proxy. Same-origin clients need no CORS override; ALLOWED_ORIGINS accepts an explicit JSON array. This is a shared-secret prototype without multi-user accounts, per-device revocation or production availability guarantees.

Binding 0.0.0.0 accepts network connections; internet routing still requires a tunnel or public deployment. Private HTTP LAN addresses generally cannot request browser camera access. The temporary HTTPS tunnel changes URL on restart and depends on Cloudflare/network availability. See [Cloudflare quick tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/) and [browser camera requirements](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia).

## Review rules and privacy

- Track IDs describe tracking segments. Anonymous P-<session>-<number> person IDs can link segments using motion, torso colour, timestamps and similarity margins. Gap matches are explicitly uncertain. No face recognition, learned ReID or cross-camera identification is enabled. Descriptors expire from memory after 120 seconds by default and are never written to SQLite.
- Restricted zones use person footpoints, configurable dwell (default 1.5 seconds) and at least three observations. Bag monitoring zones produce MEDIUM unattended-object candidates after six stationary seconds without a nearby associated person. Bags outside monitoring zones are not automatically alerts.
- Knife warnings need confidence >=0.55, at least three observations over one second and person-box proximity. Knife recall is limited. Proximity does not prove holding, ownership or intent. Firearm/violence detection is unsupported.
- Fall and loitering heuristics are disabled by default. Narrative reports are also disabled. ENABLE_AI_DOSSIER=true enables offline summaries; additionally setting GEMINI_API_KEY enables optional Google summaries and transmits incident thumbnails. External Gemini calls were not tested here.
- Events include rules, confidence, duration, camera/track/person IDs, uncertainty and protected JPEG evidence. The UI supports filters, acknowledgement, false-positive review and deletion. Defaults retain seven days and at most 2,000 events, pruned on startup and hourly. STORE_EVENT_IMAGES=false disables saved thumbnails. Continuous live video is not recorded automatically.

## Checks

~~~powershell
.\.venv\Scripts\python.exe -E -m pip check
.\.venv\Scripts\python.exe -E -m unittest discover -s tests -v
node tests/mobile_transport_test.cjs
.\.venv\Scripts\python.exe -E scripts/evaluate_tracking.py
.\.venv\Scripts\python.exe -E scripts/evaluate_events.py
.\.venv\Scripts\python.exe -E scripts/benchmark_preview.py
~~~

With the server running, node tests/browser_smoke.cjs exercises Chromium desktop/mobile layouts, zone controls, simulated camera transport, event review/evidence/deletion and recorded playback. Set CHROME_PATH if needed. With --remote running, use the venv Python to run tests/remote_smoke.py http://localhost:8000 for public HTTPS/WSS role checks. Synthetic fixtures are labelled and do not measure real-world accuracy.

Install requirements-dev.txt for formatting/linting, then run the venv Python with -E -m ruff format --check app scripts tests run.py and -E -m ruff check app scripts tests run.py.

The deployed model remains vendor-pretrained. An audited local 8/4/4-image fine-tuning smoke experiment completed and was **not promoted** because it did not consistently beat its baseline. Commands and exact results are in the engineering report.
