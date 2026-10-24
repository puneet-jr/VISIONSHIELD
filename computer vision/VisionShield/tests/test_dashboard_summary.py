"""Reference-dashboard counts use persisted data, not decorative sample values."""

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.config import config
from app.db.database import Database
from app.main import app


class SummaryTests(unittest.TestCase):
    def test_current_day_counts_and_review_status(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Database(str(Path(directory) / "events.sqlite"))
            store.insert_event({"event_id": "new", "occurred_at": time.time()})
            store.insert_event({"event_id": "reviewed", "occurred_at": time.time()})
            store.insert_event({"event_id": "yesterday", "occurred_at": time.time() - 86400})
            store.review_event("reviewed", "acknowledged")
            summary = store.event_summary()
            self.assertEqual(summary["events_today"], 2)
            self.assertEqual(summary["new"], 1)
            self.assertEqual(summary["acknowledged"], 1)
            store.delete_event("new")
            self.assertEqual(store.event_summary()["events_today"], 1)

    def test_summary_is_operator_only(self):
        client = TestClient(app)
        with patch.object(config, "TRUST_LOCAL_DASHBOARD", False):
            self.assertEqual(client.get("/api/summary").status_code, 401)
            self.assertEqual(
                client.get(
                    "/api/summary", headers={"Authorization": "Bearer " + config.CAMERA_TOKEN}
                ).status_code,
                403,
            )
            self.assertEqual(
                client.get(
                    "/api/summary", headers={"Authorization": "Bearer " + config.ACCESS_TOKEN}
                ).status_code,
                200,
            )


if __name__ == "__main__":
    unittest.main()
