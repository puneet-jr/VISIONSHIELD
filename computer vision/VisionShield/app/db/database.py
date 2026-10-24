"""High-Performance SQLite Database with WAL (Write-Ahead Logging)
Provides concurrent reading and writing without locking.
Zero-maintenance persistence for hackathons.
"""
import sqlite3
from contextlib import contextmanager
import os
import json
from typing import List, Dict, Any, Optional
from app.config import config

class Database:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
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

    def insert_event(self, event: Dict[str, Any]) -> int:
        """Insert a security event and return its ID."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
            INSERT INTO security_events (
                timestamp, alert_type, severity, track_id, object_label, 
                threat_score, explanation, zone_id, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.get("timestamp"),
                event.get("alert_type"),
                event.get("severity", "MEDIUM"),
                event.get("track_id"),
                event.get("object_label"),
                event.get("threat_score", 0.0),
                event.get("explanation", ""),
                event.get("zone_id"),
                json.dumps(event.get("metadata", {}))
            ))
            return cur.lastrowid

    def insert_incident(self, incident: Dict[str, Any]) -> int:
        """Insert a multimodal AI incident briefing."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
            INSERT INTO tactical_incidents (
                timestamp, incident_title, severity, executive_summary,
                subject_description, threat_level_rating, recommended_action, snapshot_base64
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                incident.get("timestamp"),
                incident.get("incident_title", "Security Anomaly"),
                incident.get("severity", "HIGH"),
                incident.get("executive_summary", ""),
                incident.get("subject_description", ""),
                incident.get("threat_level_rating", 7),
                incident.get("recommended_action", ""),
                incident.get("snapshot_base64")
            ))
            return cur.lastrowid

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch latest security events."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM security_events ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(row) for row in cur.fetchall()]

    def get_recent_incidents(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch latest tactical incident dossiers."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM tactical_incidents ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(row) for row in cur.fetchall()]

# Global singleton database instance
db = Database()
