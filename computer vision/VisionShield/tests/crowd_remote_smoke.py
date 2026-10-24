"""Actual HTTPS/WSS multi-device/guard test; no camera secrets written to artifacts."""

import asyncio
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit, parse_qs
import cv2
import numpy as np
import httpx
from websockets.asyncio.client import connect


async def main():
    local = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    async with httpx.AsyncClient(timeout=20, trust_env=False) as operator:
        info = (await operator.get(local + "/api/connection")).json()
        parts = urlsplit(info["phone_url"])
        base = parts.scheme + "://" + parts.netloc
        assert info["remote_ready"], "Start run.py --remote"
        camera_token = parse_qs(parts.fragment)["token"][0]
        guard_info = (await operator.get(local + "/api/connection?access=guard")).json()
        guard_token = parse_qs(urlsplit(guard_info["phone_url"]).fragment)["token"][0]
        async with (
            httpx.AsyncClient(timeout=20, trust_env=False) as a,
            httpx.AsyncClient(timeout=20, trust_env=False) as b,
            httpx.AsyncClient(timeout=20, trust_env=False) as guard,
        ):
            assert (await a.get(base + "/api/streams")).status_code == 401
            for client, token in [(a, camera_token), (b, camera_token), (guard, guard_token)]:
                response = await client.post(base + "/api/pair", json={"token": token})
                assert response.status_code == 200, response.status_code
            cookies = lambda client: "; ".join(
                f"{c.name}={c.value}" for c in client.cookies.jar if c.domain == parts.hostname
            )
            picture = Path("data/pose-smoke-bus.jpg").read_bytes()
            blank = cv2.imencode(".jpg", np.full((480, 640, 3), 50, np.uint8))[1].tobytes()
            stop = asyncio.Event()
            acked = [0, 0]
            async with (
                connect(
                    "wss://" + parts.netloc + "/ws/camera",
                    additional_headers={"Cookie": cookies(a)},
                    open_timeout=20,
                ) as wa,
                connect(
                    "wss://" + parts.netloc + "/ws/camera",
                    additional_headers={"Cookie": cookies(b)},
                    open_timeout=20,
                ) as wb,
            ):

                async def feed(ws, frame, index):
                    while not stop.is_set():
                        await ws.send(frame)
                        message = json.loads(await asyncio.wait_for(ws.recv(), 15))
                        assert message.get("ok"), message
                        acked[index] += 1
                        await asyncio.sleep(0.2)

                tasks = [
                    asyncio.create_task(feed(wa, picture, 0)),
                    asyncio.create_task(feed(wb, blank, 1)),
                ]
                backup = {}
                created_events = []
                try:
                    for _ in range(80):
                        data = (await operator.get(local + "/api/streams")).json()
                        online = [s for s in data["streams"] if s["online"] and not s["stale"]]
                        if len(online) >= 2 and any(
                            s["counts"].get("person", 0) >= 2 for s in online
                        ):
                            break
                        await asyncio.sleep(0.25)
                    else:
                        raise AssertionError(
                            "Two fresh independently inferred streams not available"
                        )
                    busy = next(s for s in online if s["counts"].get("person", 0) >= 2)
                    quiet = next(s for s in online if s["counts"].get("person", 0) == 0)
                    backup[busy["id"]] = busy["zones"]
                    testzone = {
                        "id": "crowd-transport-test",
                        "name": "TEST ONLY occupancy",
                        "purpose": "crowd_monitor",
                        "max_capacity": 2,
                        "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    }
                    save = await operator.post(
                        local + "/api/streams/" + busy["id"] + "/zones", json={"zones": [testzone]}
                    )
                    assert save.status_code == 200, save.text
                    assert (
                        await b.post(
                            base + "/api/camera/location",
                            json={"latitude": 1.234, "longitude": 2.345, "accuracy_m": 10},
                        )
                    ).status_code == 200
                    data = (await guard.get(base + "/api/streams")).json()
                    assert not data["can_configure"]
                    assert (
                        await guard.post(base + "/api/source", json={"source": "0"})
                    ).status_code == 403
                    assert (await a.get(base + "/api/events")).status_code == 403
                    async with connect(
                        "wss://" + parts.netloc + "/ws/streams/" + busy["id"] + "/preview",
                        additional_headers={"Cookie": cookies(guard)},
                        open_timeout=20,
                    ) as preview:
                        jpeg = await asyncio.wait_for(preview.recv(), 15)
                        assert isinstance(jpeg, bytes) and jpeg[:2] == b"\xff\xd8"
                    for _ in range(60):
                        events = (
                            await operator.get(
                                local + "/api/events?event_type=CROWD_CAPACITY_EXCEEDED&limit=30"
                            )
                        ).json()["events"]
                        created_events = [
                            e
                            for e in events
                            if e.get("zone_id") == "crowd-transport-test"
                            and e["camera_id"] == busy["camera_id"]
                        ]
                        if created_events:
                            break
                        await asyncio.sleep(0.25)
                    assert created_events, "No real detection-driven capacity event"
                    ack = await guard.patch(
                        base + "/api/events/" + created_events[0]["event_id"],
                        json={"status": "acknowledged"},
                    )
                    assert ack.status_code == 200, ack.text
                    data = (await operator.get(local + "/api/streams")).json()
                    quiet_now = next(s for s in data["streams"] if s["id"] == quiet["id"])
                    assert quiet_now["counts"].get("person", 0) == 0
                    assert quiet_now["camera"]["location"]["latitude"] == 1.234
                    report = {
                        "test": "real HTTPS/WSS two camera transports and guard role; photo/blank fixtures, not physical phones",
                        "passed": True,
                        "public_origin": base,
                        "acknowledged_frames": acked,
                        "concurrent_fresh_cameras": len(online),
                        "photo_visible_people": busy["counts"].get("person"),
                        "blank_visible_people": quiet_now["counts"].get("person", 0),
                        "guard_preview_jpeg": True,
                        "guard_admin_forbidden": True,
                        "camera_evidence_forbidden": True,
                        "camera_specific_gps_transport": True,
                        "real_detection_capacity_event": True,
                        "guard_acknowledged": True,
                    }
                    Path("artifacts/crowd-remote-verification.json").write_text(
                        json.dumps(report, indent=2), encoding="utf-8"
                    )
                    print(json.dumps(report, indent=2))
                finally:
                    stop.set()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    for key, zones in backup.items():
                        await operator.post(
                            local + "/api/streams/" + key + "/zones", json={"zones": zones}
                        )
                    for event in created_events:
                        await operator.delete(local + "/api/events/" + event["event_id"])


if __name__ == "__main__":
    asyncio.run(main())
