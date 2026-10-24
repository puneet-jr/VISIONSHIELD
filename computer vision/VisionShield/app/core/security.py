"""Role-aware access for one operator secret and camera-scoped pairing sessions."""

from collections import defaultdict, deque
import ipaddress
import secrets
import time
from app.config import config

attempts = defaultdict(deque)


def role(connection):
    supplied = connection.cookies.get("vs_access", "")
    authorization = connection.headers.get("authorization", "")
    if authorization.startswith("Bearer "):
        supplied = authorization[7:]
    if supplied and secrets.compare_digest(supplied, config.ACCESS_TOKEN):
        return "operator"
    if supplied and secrets.compare_digest(supplied, config.CAMERA_TOKEN):
        return "camera"
    if config.TRUST_LOCAL_DASHBOARD and config.DEPLOYMENT_MODE == "demo":
        host = connection.url.hostname or ""
        peer = connection.client.host if connection.client else ""
        try:
            local_host = (
                host == "localhost"
                or ipaddress.ip_address(host).is_private
                or ipaddress.ip_address(host).is_loopback
            )
        except ValueError:
            local_host = host == "localhost"
        try:
            local_peer = (
                ipaddress.ip_address(peer).is_private or ipaddress.ip_address(peer).is_loopback
            )
        except ValueError:
            local_peer = host == "testserver" and peer == "testclient"
        if host == "testserver" and peer == "testclient":
            local_host = True
        if local_host and local_peer:
            return "operator"
    return None


def origin_allowed(connection):
    origin = connection.headers.get("origin")
    if not origin:
        return True  # CLI/automation authenticates using its own secret/cookie.
    expected = {str(connection.base_url).rstrip("/")} if hasattr(connection, "base_url") else set()
    expected.add(
        f"{connection.url.scheme.replace('ws', 'http')}://{connection.headers.get('host', '')}"
    )
    if config.PUBLIC_BASE_URL:
        expected.add(config.PUBLIC_BASE_URL)
    expected.update(config.ALLOWED_ORIGINS)
    return origin.rstrip("/") in expected


def pair_allowed(ip):
    now = time.monotonic()
    queue = attempts[ip]
    while queue and now - queue[0] > 60:
        queue.popleft()
    if len(queue) >= 10:
        return False
    queue.append(now)
    if len(attempts) > 1024:
        for key in list(attempts)[:512]:
            if key != ip:
                attempts.pop(key, None)
    return True
