""" SQLite persistence for inference traces, collection contracts, manifests, and events. """

import json
import os
import sqlite3
from datetime import datetime, timezone

class AuditService:
    """ AuditService groups related application behavior. """
    def __init__(self, db_path):
        """ Handle the   init   operation. """
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._initialize()

    def _connect(self):
        """ Open a configured SQLite connection with row objects and integrity settings. """
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _initialize(self):
        """ Create the audit, registry, manifest, and event tables if absent. """
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS inference_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    model TEXT,
                    collection_name TEXT,
                    question TEXT,
                    sources_json TEXT,
                    context_sent TEXT,
                    response TEXT,
                    duration_seconds REAL,
                    prompt_tokens INTEGER,
                    generated_tokens INTEGER,
                    status TEXT NOT NULL,
                    error TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS collection_registry (
                    collection_name TEXT PRIMARY KEY,
                    embedding_model TEXT NOT NULL,
                    embedding_dimension INTEGER NOT NULL,
                    distance_metric TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_manifests (
                    document_id TEXT PRIMARY KEY,
                    collection_name TEXT NOT NULL,
                    title TEXT NOT NULL,
                    point_ids_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS application_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details_json TEXT
                )
            """)

    def record_inference(self, **kwargs):
        """ Persist prompt, retrieved context, response, metrics, and status. """
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO inference_audit (
                    request_id, created_at, model, collection_name, question,
                    sources_json, context_sent, response, duration_seconds,
                    prompt_tokens, generated_tokens, status, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                kwargs["request_id"],
                datetime.now(timezone.utc).isoformat(),
                kwargs.get("model"), kwargs.get("collection"),
                kwargs.get("question"),
                json.dumps(kwargs.get("sources", []), ensure_ascii=False),
                kwargs.get("context_sent"), kwargs.get("response"),
                kwargs.get("duration_seconds"), kwargs.get("prompt_tokens"),
                kwargs.get("generated_tokens"), kwargs.get("status", "SUCCESS"),
                kwargs.get("error")
            ))

    def record_event(self, event_type, status, details):
        """ Persist an operational event for troubleshooting and audit. """
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO application_events (created_at, event_type, status, details_json)
                VALUES (?, ?, ?, ?)
            """, (
                datetime.now(timezone.utc).isoformat(), event_type, status,
                json.dumps(details, ensure_ascii=False)
            ))


    def get_collection_contract(self, collection_name):
        """ Read the registered embedding model and vector configuration. """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM collection_registry WHERE collection_name = ?",
                (collection_name,)
            ).fetchone()
        return dict(row) if row else None

    def register_collection_contract(self, collection_name, embedding_model, dimension, distance_metric="Cosine"):
        """Create the contract once; reject model/dimension changes thereafter."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT embedding_model, embedding_dimension, distance_metric "
                "FROM collection_registry WHERE collection_name = ?",
                (collection_name,)
            ).fetchone()
            if row:
                expected = (row["embedding_model"], row["embedding_dimension"], row["distance_metric"])
                actual = (embedding_model, dimension, distance_metric)
                if expected != actual:
                    raise ValueError(
                        f"Embedding contract mismatch for '{collection_name}'. "
                        f"Existing={expected}; requested={actual}. Create a new collection "
                        "or explicitly re-embed/migrate all documents."
                    )
            else:
                conn.execute(
                    """INSERT INTO collection_registry
                    (collection_name, embedding_model, embedding_dimension, distance_metric, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (collection_name, embedding_model, dimension, distance_metric, now, now)
                )

    def create_ingestion_manifest(self, document_id, collection_name, title, point_ids):
        """ Record intended point IDs before attempting a Qdrant write. """
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO ingestion_manifests
                (document_id, collection_name, title, point_ids_json, status, error, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'PREPARING', NULL, ?, ?)""",
                (document_id, collection_name, title, json.dumps(point_ids), now, now)
            )

    def update_ingestion_manifest(self, document_id, status, error=None):
        """ Update the durable state of an ingestion operation. """
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE ingestion_manifests SET status = ?, error = ?, updated_at = ? WHERE document_id = ?",
                (status, error, now, document_id)
            )

    def record_event(self, event_type, status, details):
        """ Persist an operational event for troubleshooting and audit. """
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO application_events (created_at, event_type, status, details_json)
                VALUES (?, ?, ?, ?)
            """, (
                datetime.now(timezone.utc).isoformat(), event_type, status,
                json.dumps(details, ensure_ascii=False)
            ))


    def claim_stale_ingestion_manifests(self, stale_seconds=120, limit=100):
        """
        Atomically claim old unfinished manifests.
        RECOVERING is included so a process crash during recovery is recoverable.
        """
        from datetime import datetime, timezone, timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=stale_seconds)).isoformat()
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT document_id, collection_name, title, point_ids_json, status,
                       error, created_at, updated_at
                FROM ingestion_manifests
                WHERE status IN ('PREPARING', 'ROLLBACK_FAILED', 'RECOVERING')
                  AND updated_at <= ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (cutoff, limit)
            ).fetchall()

            claimed = []
            for row in rows:
                cursor = conn.execute(
                    """
                    UPDATE ingestion_manifests
                    SET status = 'RECOVERING', updated_at = ?, error = NULL
                    WHERE document_id = ?
                      AND status IN ('PREPARING', 'ROLLBACK_FAILED', 'RECOVERING')
                      AND updated_at <= ?
                    """,
                    (now, row["document_id"], cutoff)
                )
                if cursor.rowcount == 1:
                    item = dict(row)
                    item["point_ids"] = json.loads(item.pop("point_ids_json") or "[]")
                    claimed.append(item)
            return claimed

    def list_ingestion_manifests(self, limit=100):
        """ Return recent manifest states for operational inspection. """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT document_id, collection_name, title, status, error,
                       created_at, updated_at
                FROM ingestion_manifests
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def recent(self, limit=100):
        """ Return recent inference audit rows with decoded source metadata. """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM inference_audit ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["sources"] = json.loads(item.pop("sources_json") or "[]")
            result.append(item)
        return result
