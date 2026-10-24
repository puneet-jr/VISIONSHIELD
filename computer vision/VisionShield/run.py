"""Local server with optional outbound HTTPS tunnel for remote phone cameras."""

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import sys
import signal

if __name__ == "__main__" and not sys.flags.ignore_environment and os.environ.get("PYTHONPATH"):
    os.execv(sys.executable, [sys.executable, "-E", *sys.argv])
import uvicorn
from app.config import config

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="VisionShield camera command center")
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Start a temporary public HTTPS phone tunnel",
    )
    parser.add_argument("--port", type=int, default=config.PORT)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use YOLO11n at 416 px for speed; small-object recall can drop",
    )
    parser.add_argument(
        "--source",
        default="phone_upload",
        help="0, 1, phone_upload, or recording:<name in RECORDINGS_DIR>",
    )
    args = parser.parse_args()
    os.environ["START_SOURCE"] = args.source
    if args.fast:
        config.YOLO_MODEL = str(ROOT / "yolo11n.pt")
        config.INPUT_SIZE = 416
    config.PORT = args.port
    os.chdir(ROOT)
    process = None
    if args.remote:
        binary = shutil.which("cloudflared") or str(ROOT / "tools" / "cloudflared.exe")
        if not Path(binary).exists():
            parser.error("cloudflared is missing. Run setup.ps1 first.")
        # Force a nonlocal Host at the origin so public traffic always requires pairing.
        process = subprocess.Popen(
            [
                binary,
                "tunnel",
                "--no-autoupdate",
                "--url",
                f"http://127.0.0.1:{args.port}",
                "--http-host-header",
                "visionshield.remote",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

        def watch_tunnel():
            for line in process.stdout:
                match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
                if match:
                    config.PUBLIC_BASE_URL = match.group(0)
                    print(
                        "Secure phone connection ready. Open Connect phone in the local dashboard.",
                        flush=True,
                    )
            config.PUBLIC_BASE_URL = ""
            print(
                "Remote tunnel stopped. Restart --remote to reconnect phones.",
                flush=True,
            )

        threading.Thread(target=watch_tunnel, daemon=True).start()
    print(
        f"Detector: {Path(config.YOLO_MODEL).name}, {config.INPUT_SIZE}px. Knife candidates require human review."
    )
    print(f"VisionShield dashboard: http://localhost:{args.port}")
    print("Phone: use Connect phone in the dashboard. Remote mode works across networks.")
    try:
        # Uvicorn replays the shutdown signal to the previous handler. Windows'
        # default SIGBREAK terminates before finally can close our tunnel.
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, signal.default_int_handler)
        uvicorn.run(
            "app.main:app",
            host=config.HOST,
            port=args.port,
            log_level="info",
            ws_max_size=512_000,
            ws_max_queue=2,
        )
    finally:
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    main()
