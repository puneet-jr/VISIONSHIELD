"""High-Performance SQLite Database with WAL (Write-Ahead Logging)
Provides concurrent reading and writing without locking.
Zero-maintenance persistence for hackathons.
"""

import sqlite3
import time
from uuid import uuid4
from datetime import datetime, timezone
from app.db.migrations import migrate
from contextlib import contextmanager
import os
import json
from typing import List, Dict, Any, Optional
from app.config import config


class Database:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.DB_PATH
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high concurrent throughput
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self):
        """Create tables if they do not exist."""
        with self._get_connection() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS security_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                track_id INTEGER,
                object_label TEXT NOT NULL,
                threat_score REAL NOT NULL,
                explanation TEXT NOT NULL,
                zone_id TEXT,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS tactical_incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                incident_title TEXT NOT NULL,
                severity TEXT NOT NULL,
                executive_summary TEXT NOT NULL,
                subject_description TEXT,
                threat_level_rating INTEGER,
                recommended_action TEXT NOT NULL,
                snapshot_base64 TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_events_timestamp ON security_events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_events_severity ON security_events(severity);
            """)

            migrate(conn)

    def insert_event(self, event, evidence_base64=None):
        event_id = event.setdefault("event_id", uuid4().hex)
        timestamp = event.get("timestamp") or datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO security_events
                (timestamp,alert_type,severity,track_id,object_label,threat_score,explanation,zone_id,metadata,
                 event_id,schema_version,camera_id,occurred_at,priority,details,evidence_base64,status)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    timestamp,
                    event.get("alert_type", "OBSERVATION"),
                    event.get("severity", "LOW"),
                    event.get("track_id"),
                    event.get("object_label", "object"),
                    event.get("threat_score", 0),
                    event.get("explanation", ""),
                    event.get("zone_id"),
                    json.dumps(event.get("metadata", {})),
                    event_id,
                    event.get("schema_version", 1),
                    event.get("camera_id", "unknown"),
                    event.get("occurred_at", time.time()),
                    event.get("priority", "review_candidate"),
                    json.dumps(event),
                    evidence_base64,
                    event.get("status", "new"),
                ),
            )
            return cursor.lastrowid

    @staticmethod
    def _public_event(row):
        record = dict(row)
        event = json.loads(record.get("details") or "{}")
        if not event:
            event = {k: v for k, v in record.items() if k not in {"evidence_base64", "details"}}
            event["legacy"] = True
            event["assessment"] = {
                "type": "legacy_unversioned",
                "limitations": ["Historical record predates the current explanation policy."],
            }
        event.update(
            id=record["id"],
            event_id=record.get("event_id") or f"legacy-{record['id']}",
            status=record.get("status") or "new",
            reviewed_at=record.get("reviewed_at"),
        )
        event["has_evidence"] = bool(record.get("evidence_base64"))
        return event

    def get_event(self, event_id, include_evidence=False):
        with self._get_connection() as conn:
            if event_id.startswith("legacy-") and event_id[7:].isdigit():
                row = conn.execute(
                    "SELECT * FROM security_events WHERE id=?", (int(event_id[7:]),)
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM security_events WHERE event_id=?", (event_id,)
                ).fetchone()
            if not row:
                return None
            result = self._public_event(row)
            if include_evidence:
                result["evidence_base64"] = row["evidence_base64"]
            return result

    def review_event(self, event_id, status):
        if status not in {"acknowledged", "false_positive", "new"}:
            raise ValueError("Invalid review status")
        with self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE security_events SET status=?,reviewed_at=? WHERE event_id=?",
                (status, datetime.now(timezone.utc).isoformat(), event_id),
            )
            return cursor.rowcount > 0

    def delete_event(self, event_id):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM tactical_incidents WHERE event_id=?", (event_id,))
            return (
                conn.execute("DELETE FROM security_events WHERE event_id=?", (event_id,)).rowcount
                > 0
            )

    def record_identity_segments(self, segments):
        if not segments:
            return
        with self._get_connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO identity_segments VALUES (?,?,?,?,?,?,?)",
                [
                    (
                        s["person_id"],
                        s["camera_id"],
                        s["generation"],
                        s["track_id"],
                        s["timestamp"],
                        s["association"],
                        s.get("distance"),
                    )
                    for s in segments
                ],
            )

    def get_identity_segments(self, limit=100):
        with self._get_connection() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM identity_segments ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                )
            ]

    def prune(self, days=None, max_events=None):
        days = config.EVIDENCE_RETENTION_DAYS if days is None else days
        max_events = config.MAX_STORED_EVENTS if max_events is None else max_events
        cutoff = time.time() - days * 86400
        with self._get_connection() as conn:
            removed = conn.execute(
                "DELETE FROM security_events WHERE COALESCE(occurred_at,CAST(strftime('%s',created_at) AS REAL))<?",
                (cutoff,),
            ).rowcount
            removed += conn.execute(
                "DELETE FROM security_events WHERE id NOT IN (SELECT id FROM security_events ORDER BY id DESC LIMIT ?)",
                (max_events,),
            ).rowcount
            conn.execute(
                "DELETE FROM tactical_incidents WHERE event_id IS NOT NULL AND event_id NOT IN (SELECT event_id FROM security_events)"
            )
            conn.execute(
                "DELETE FROM tactical_incidents WHERE created_at<datetime('now',?)",
                (f"-{days} days",),
            )
            conn.execute("DELETE FROM identity_segments WHERE timestamp<?", (cutoff,))
            return removed

    def insert_incident(self, incident: Dict[str, Any]) -> int:
        """Insert a multimodal AI incident briefing."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
            INSERT INTO tactical_incidents (
                timestamp, incident_title, severity, executive_summary,
                subject_description, threat_level_rating, recommended_action, snapshot_base64, event_id, report_mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    incident.get("timestamp"),
                    incident.get("incident_title", "Security Anomaly"),
                    incident.get("severity", "HIGH"),
                    incident.get("executive_summary", ""),
                    incident.get("subject_description", ""),
                    incident.get("threat_level_rating", 7),
                    incident.get("recommended_action", ""),
                    incident.get("snapshot_base64"),
                    incident.get("event_id"),
                    incident.get("report_mode", "offline_template"),
                ),
            )
            return cur.lastrowid

    def get_recent_events(
        self,
        limit=50,
        camera=None,
        event_type=None,
        severity=None,
        status=None,
        start=None,
        end=None,
    ):
        clauses = []
        arguments = []
        for column, value in [
            ("camera_id", camera),
            ("alert_type", event_type),
            ("severity", severity),
            ("status", status),
        ]:
            if value:
                clauses.append(f"{column}=?")
                arguments.append(value)
        for operator, value in [(">=", start), ("<=", end)]:
            if value is not None:
                clauses.append(f"occurred_at{operator}?")
                arguments.append(value)
        query = (
            "SELECT * FROM security_events"
            + (" WHERE " + " AND ".join(clauses) if clauses else "")
            + " ORDER BY id DESC LIMIT ?"
        )
        with self._get_connection() as conn:
            return [self._public_event(row) for row in conn.execute(query, (*arguments, limit))]

    def get_recent_incidents(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch latest tactical incident dossiers."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM tactical_incidents ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(row) for row in cur.fetchall()]


# Global singleton database instance
db = Database()
