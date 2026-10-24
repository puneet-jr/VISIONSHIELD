"""Run only against an explicitly authorized temporary remote tunnel."""

import asyncio
import json
import sys
import time
from urllib.parse import urlsplit, parse_qs
import cv2
import httpx
import numpy as np
from websockets.asyncio.client import connect


async def main():
    local = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001"
    async with httpx.AsyncClient(timeout=15, trust_env=False) as client:
        for _ in range(60):
            info = (await client.get(local + "/api/connection")).json()
            if info["remote_ready"]:
                break
            await asyncio.sleep(0.5)
        else:
            raise RuntimeError("Tunnel did not become ready")
        parts = urlsplit(info["phone_url"])
        base = f"{parts.scheme}://{parts.netloc}"
        token = parse_qs(parts.fragment)["token"][0]
        # No secret pairing links or cookies are printed or saved.
        unauth = await client.get(base + "/api/info")
        assert unauth.status_code == 401, ("Unpaired data access", unauth.status_code)
        assert (await client.get(base + "/phone")).status_code == 200
        paired = await client.post(base + "/api/pair", json={"token": token})
        assert paired.status_code == 200, paired.status_code
        assert (await client.get(base + "/api/info")).status_code == 403
        assert (await client.get(base + "/api/camera/status")).status_code == 200
        cookie = "; ".join(
            f"{c.name}={c.value}" for c in client.cookies.jar if c.domain == parts.hostname
        )
        ok, encoded = cv2.imencode(".jpg", np.full((480, 640, 3), 70, dtype=np.uint8))
        assert ok
        started = time.perf_counter()
        async with connect(
            "wss://" + parts.netloc + "/ws/camera",
            additional_headers={"Cookie": cookie},
            open_timeout=20,
        ) as ws:
            for _ in range(10):
                await ws.send(encoded.tobytes())
                ack = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
                assert ack.get("ok"), ack
            elapsed = time.perf_counter() - started
            telemetry = (await client.get(local + "/api/info")).json()
            assert telemetry["camera_online"], telemetry
        operator_info = (await client.get(local + "/api/connection?access=operator")).json()
        operator_token = parse_qs(urlsplit(operator_info["phone_url"]).fragment)["token"][0]
        assert (
            await client.post(base + "/api/pair", json={"token": operator_token})
        ).status_code == 200
        assert (await client.get(base + "/api/insights")).status_code == 200
        print(
            json.dumps(
                {
                    "public_url": base,
                    "unpaired_status": unauth.status_code,
                    "paired_status": paired.status_code,
                    "acknowledged_frames": 10,
                    "roundtrip_upload_fps": round(10 / elapsed, 2),
                    "source_type": telemetry["source_type"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
