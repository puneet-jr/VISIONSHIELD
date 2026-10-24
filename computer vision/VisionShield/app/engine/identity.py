"""Anonymous session identities across track segments. No facial recognition.
Descriptors stay in memory, are camera-scoped, expire, and are never persisted.
"""

from collections import deque
from uuid import uuid4
import numpy as np
from app.config import config


class IdentityManager:
    def __init__(self):
        self.session = uuid4().hex[:8]
        self.next_id = 1
        self.people = {}
        self.mapping = {}
        self.history = deque(maxlen=500)
        self.pending = []

    def associate(self, tracks, camera_id, generation, timestamp):
        expired = [
            pid
            for pid, p in self.people.items()
            if timestamp - p["last_seen"] > config.IDENTITY_RETENTION_SECONDS
        ]
        for pid in expired:
            del self.people[pid]
        self.mapping = {key: pid for key, pid in self.mapping.items() if pid in self.people}
        persons = [t for t in tracks if t.label == "person"]
        claimed = {
            self.mapping[(camera_id, generation, t.track_id)]
            for t in persons
            if (camera_id, generation, t.track_id) in self.mapping
        }
        # Known segments first, so new arrivals cannot steal an active person's identity.
        persons.sort(key=lambda t: (camera_id, generation, t.track_id) not in self.mapping)
        for track in persons:
            key = (camera_id, generation, track.track_id)
            pid = self.mapping.get(key)
            status = "segment_continuity"
            score = None
            if pid is None:
                candidates = []
                if track.appearance is not None:
                    for old_id, old in self.people.items():
                        if (
                            old_id in claimed
                            or old["camera_id"] != camera_id
                            or old["appearance"] is None
                        ):
                            continue
                        gap = timestamp - old["last_seen"]
                        if gap < 0.05:
                            continue
                        distance = 1 - float(
                            np.clip(np.dot(track.appearance, old["appearance"]), 0, 1)
                        )
                        height = max(1, track.box[3] - track.box[1])
                        ratio = height / max(1, old["height"])
                        if distance <= config.IDENTITY_DISTANCE_THRESHOLD and 0.5 <= ratio <= 2:
                            candidates.append((distance, old_id))
                candidates.sort()
                ambiguous = (
                    len(candidates) > 1
                    and candidates[1][0] - candidates[0][0] < config.IDENTITY_MATCH_MARGIN
                )
                if candidates and not ambiguous:
                    score, pid = candidates[0]
                    status = "appearance_link_uncertain"
                else:
                    pid = f"P-{self.session}-{self.next_id:04d}"
                    self.next_id += 1
                    status = "ambiguous_new_identity" if ambiguous else "new_unlinked_identity"
                self.mapping[key] = pid
                self.history.append(
                    {
                        "person_id": pid,
                        "camera_id": camera_id,
                        "generation": generation,
                        "track_id": track.track_id,
                        "timestamp": timestamp,
                        "association": status,
                        "distance": score,
                    }
                )
                self.pending.append(self.history[-1])
            claimed.add(pid)
            old = self.people.get(pid)
            signature = track.appearance
            if old and signature is not None and old["appearance"] is not None:
                signature = 0.9 * old["appearance"] + 0.1 * signature
                signature /= max(np.linalg.norm(signature), 1e-6)
            self.people[pid] = {
                "camera_id": camera_id,
                "last_seen": timestamp,
                "appearance": signature,
                "height": track.box[3] - track.box[1],
                "last_track_id": track.track_id,
            }
            track.person_id = pid
            track.identity_association = status
            track.identity_distance = score
        if len(self.people) > 128:
            for pid in sorted(self.people, key=lambda p: self.people[p]["last_seen"])[:-128]:
                del self.people[pid]
        return list(self.history)

    def drain_history(self):
        result, self.pending = self.pending, []
        return result

    def public_history(self):
        return list(self.history)

    def clear(self):
        self.people.clear()
        self.mapping.clear()
        self.history.clear()
        self.pending.clear()
