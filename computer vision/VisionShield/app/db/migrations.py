"""Additive migrations preserve existing operator history."""


def migrate(connection):
    columns = {row[1] for row in connection.execute("PRAGMA table_info(security_events)")}
    additions = {
        "event_id": "TEXT",
        "schema_version": "INTEGER DEFAULT 0",
        "camera_id": "TEXT",
        "occurred_at": "REAL",
        "priority": "TEXT",
        "details": "TEXT",
        "evidence_base64": "TEXT",
        "status": "TEXT DEFAULT 'new'",
        "reviewed_at": "TEXT",
    }
    for name, definition in additions.items():
        if name not in columns:
            connection.execute(f"ALTER TABLE security_events ADD COLUMN {name} {definition}")
    connection.execute("UPDATE security_events SET event_id='legacy-'||id WHERE event_id IS NULL")
    columns = {row[1] for row in connection.execute("PRAGMA table_info(tactical_incidents)")}
    for name, definition in {"event_id": "TEXT", "report_mode": "TEXT"}.items():
        if name not in columns:
            connection.execute(f"ALTER TABLE tactical_incidents ADD COLUMN {name} {definition}")
    connection.executescript("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_event_id ON security_events(event_id) WHERE event_id IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_event_camera ON security_events(camera_id,occurred_at);
        CREATE TABLE IF NOT EXISTS identity_segments (
            person_id TEXT NOT NULL, camera_id TEXT NOT NULL, generation INTEGER NOT NULL,
            track_id INTEGER NOT NULL, timestamp REAL NOT NULL, association TEXT NOT NULL,
            distance REAL, PRIMARY KEY(person_id,camera_id,generation,track_id)
        );
        PRAGMA user_version=2;
    """)
