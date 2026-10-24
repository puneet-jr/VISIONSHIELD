Read [DEMO_GUIDE.md](DEMO_GUIDE.md) for the full laptop/mobile rehearsal, zone design, knife limitations and truthful dataset/training explanation.

# VisionShield

A local FastAPI security dashboard with browser-based phone capture, YOLO11s detection, motion and appearance-assisted tracking, owner/bag association, heuristic fall and loitering alerts, polygon geofencing, and background incident briefs.

## Start on this Windows machine

```powershell
cd "C:\Users\punee\Desktop\MY-PROJECTS\COMPUTER_VISION-V2\computer vision\VisionShield"
.\.venv\Scripts\python.exe -E run.py --remote
```

Open http://localhost:8000, wait for the secure connection ready message, then choose **Connect phone**. Scan the private QR code, tap **Start streaming**, and allow camera access. The phone may use mobile data or another Wi-Fi network. Both devices need internet, and the laptop/server must remain running.

`start.ps1` starts remote mode. On a fresh checkout, run `setup.ps1` to create Python 3.12's local environment, install requirements, and download the official Cloudflare tunnel client. No mobile app or inbound router port forwarding is needed. Run without `--remote` for laptop-only use. This machine defines a global PYTHONPATH; use `-E` to keep dependencies isolated (run.py also relaunches with isolation).

Phone camera access requires a secure browser context. A private `http://192.168...` URL normally cannot request camera permission, even on the same Wi-Fi. The outbound Cloudflare tunnel supplies a temporary HTTPS address. It is a development/demo connection, with a new URL on restart and no availability guarantee. For permanent use, deploy behind a stable HTTPS hostname and proper user accounts. References: [browser camera security](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia), [Cloudflare quick tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/).

## Pipeline

- `app/ingestion/video_stream.py`: latest-frame capture slot, source generation counters, a separate preview encoding worker and one cached JPEG shared by dashboard viewers.
- `app/engine/pipeline.py`: processes each frame once; 640 px small-model inference (416 px nano with --fast), visible-track analysis, alert cooldowns, single background report worker.
- `app/core/event_hub.py`: bounded subscriber queues, worker-to-ASGI-loop publishing, recent-event replay.
- `app/web/routes.py`: validated REST endpoints, local QR generation, authenticated remote pairing, bounded acknowledged binary camera WebSocket, raw MJPEG preview and age-limited tracking metadata for a browser overlay.
- `app/engine/`: detector, centroid association, luggage ownership, kinematics, geofencing, incident generation.
- `app/db/database.py`: SQLite WAL with deterministic transaction/connection cleanup.
- `app/web/templates` and `static`: responsive dashboard and mobile sender with reconnect, wake-lock request, camera selection, and stop/release controls.

The sender allows at most three unacknowledged JPEGs and skips captures under congestion; mobile Detail requests up to 960 px and Fast uses 640 px. It drops delay instead of queuing old frames. Capture rate and inference time are shown separately; speed depends on CPU, scene, and network. There is no verified 30 FPS/1080p guarantee. This implements JPEG over WebSocket, not WebRTC, and motion and appearance-assisted tracking, not ByteTrack/Kalman filtering.

## Reports and review

Without a key, incident briefs are deterministic offline templates; they do not inspect image content. To enable image-based reports, set `GEMINI_API_KEY` before launch; `GEMINI_MODEL` defaults to `gemini-flash-latest`. Requests use the supported `google-genai` SDK with a timeout and offline fallback. Generation runs outside inference and only one report runs at a time. An API call sends an incident thumbnail and event details to Google.

Alerts are heuristic observations for operator review. Bounding boxes cannot establish identity, authorization, intent, a medical diagnosis, or real-world distances in metres. Unattended objects and proximity do not prove danger or theft. The notes' uniqueness and accuracy claims have not been established. Zones start empty and persist per source in data/zones.json. Entry checks people at the bottom-centre of their boxes after temporal confirmation.

## Connection and access

The local dashboard generates a per-process secret pairing link; the QR is generated locally. The fragment is exchanged for an HttpOnly cookie and removed from the address bar. Restart invalidates pairing. Keep the link private: a paired device can access dashboard data and controls. Public HTTP and WebSocket routes require that cookie, except the phone bootstrap and pairing endpoints. The bundled tunnel forces a nonlocal origin Host so public requests cannot use the local-dashboard bypass. Local/LAN dashboard access is trusted; this is a single-operator prototype, not multi-user production access control.

Troubleshooting:

- No HTTPS link: check internet and tunnel stderr/network restrictions; restart remote mode. Refresh the connection dialog after readiness.
- Camera denied: enable camera permission for the HTTPS site in browser settings and close other camera apps.
- Another phone connected: stop the first phone; disconnected senders expire after 30 seconds.
- Camera offline: keep the sender page foregrounded and the screen awake. Stop releases camera tracks.
- Detector unavailable: inspect server logs and ensure `yolo11n.pt` exists in the project root.

## Verification

```powershell
.\.venv\Scripts\python.exe -E -m pip check
.\.venv\Scripts\python.exe -E -m unittest discover -s tests -v
```

The installed versions are recorded in `requirements.lock.txt`. To repeat the browser check with the local server running, use `node tests/browser_smoke.cjs`; set `CHROME_PATH` if Chromium is elsewhere. Screenshots are saved in `artifacts/`.

Regression checks cover worker-thread event delivery and bounded subscriber queues, remote authentication, invalid input handling, binary camera acknowledgement, and duplicate-frame inference prevention. A real phone on cellular still needs a manual acceptance test: pair, stream, disconnect/reconnect, draw a zone, and verify an alert and incident brief.

For an explicitly authorized remote smoke test against an already running tunnel, use `.\.venv\Scripts\python.exe -E tests/remote_smoke.py http://localhost:8001` (adjust the port). This exchanges the pairing secret internally and prints no secret URLs or cookies. Measured results and remaining acceptance checks are in `artifacts/verification.md`.
