"""NovaSight dual-backend repository layer (SQLite / Neon Postgres).

Supports SQLite locally (at state/novasight.db) and PostgreSQL when DATABASE_URL is set.
Abstracts both behind a unified StorageBackend interface.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.config import REPO_ROOT
from src.runtime.memory import EnrolledClass, RuntimeMemory
from src.runtime.scoring import RuntimeScorer

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract storage backend for NovaSight metadata, memory state, and inspection logs."""

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return human-readable provider name (e.g. SQLite (local dev), Neon Postgres (production))."""
        pass

    @abstractmethod
    def init_schema(self) -> None:
        """Create tables idempotently."""
        pass

    @abstractmethod
    def save_class(self, enrolled_class: EnrolledClass) -> None:
        """Upsert an enrolled class into the classes table."""
        pass

    @abstractmethod
    def delete_class(self, name: str) -> bool:
        """Delete an enrolled class from the classes table."""
        pass

    @abstractmethod
    def get_class(self, name: str) -> EnrolledClass | None:
        """Retrieve an enrolled class by name."""
        pass

    @abstractmethod
    def list_classes(self) -> list[EnrolledClass]:
        """List all enrolled classes ordered by enrolled_at."""
        pass

    @abstractmethod
    def record_event(self, event_dict: dict[str, Any]) -> int:
        """Record an event into the events table."""
        pass

    @abstractmethod
    def list_events(self, limit: int = 1000) -> list[dict[str, Any]]:
        """List events in chronological order."""
        pass

    @abstractmethod
    def export_events_jsonl(self) -> str:
        """Export all events as a JSONL formatted string."""
        pass

    @abstractmethod
    def export_events_csv(self) -> str:
        """Export all events as a CSV formatted string."""
        pass

    @abstractmethod
    def record_inspection(self, data: dict[str, Any]) -> int:
        """Record an inspection into the inspections table."""
        pass

    @abstractmethod
    def list_inspections(self, limit: int = 100, unreviewed_only: bool = False) -> list[dict[str, Any]]:
        """List recent inspections."""
        pass

    @abstractmethod
    def update_inspection_review(self, inspection_id: int, review_label: str, review_note: str) -> bool:
        """Update inspection review status and notes."""
        pass

    @abstractmethod
    def get_inspection_stats(self) -> dict[str, Any]:
        """Compute aggregate inspection statistics."""
        pass

    def rebuild_memory(
        self,
        scorer: RuntimeScorer,
        inheritance: str = "H1",
    ) -> RuntimeMemory:
        """Reconstruct RuntimeMemory populated from the classes table."""
        memory = RuntimeMemory(scorer=scorer, inheritance=inheritance)
        classes = self.list_classes()
        for cls in classes:
            memory.enrolled[cls.name] = cls
        logger.info("Rebuilt memory from database: %d classes enrolled", len(classes))
        return memory


class SQLiteStorage(StorageBackend):
    """SQLite implementation for local development and offline environments."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is None:
            db_path = REPO_ROOT / "state" / "novasight.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def get_provider_name(self) -> str:
        return "SQLite (local dev)"

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self) -> None:
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS classes (
                    name TEXT PRIMARY KEY,
                    mu_json TEXT NOT NULL,
                    n INTEGER NOT NULL,
                    sha256s_json TEXT NOT NULL,
                    gate_report_json TEXT NOT NULL,
                    inheritance TEXT NOT NULL,
                    r_hat REAL,
                    version INTEGER NOT NULL DEFAULT 0,
                    confirmed_count INTEGER NOT NULL DEFAULT 0,
                    enrolled_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uuid TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS inspections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    image_sha256 TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    class_name TEXT,
                    novelty_score REAL NOT NULL,
                    prediction_set_json TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    flagged_for_op INTEGER NOT NULL DEFAULT 0,
                    reviewed INTEGER NOT NULL DEFAULT 0,
                    review_label TEXT,
                    review_note TEXT,
                    latency_ms REAL NOT NULL
                );
            """)

    def save_class(self, cls: EnrolledClass) -> None:
        mu_json = json.dumps(cls.mu.tolist() if isinstance(cls.mu, np.ndarray) else list(cls.mu))
        sha256s_json = json.dumps(cls.sha256s)
        gate_report_json = json.dumps(cls.gate_report)
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO classes (
                    name, mu_json, n, sha256s_json, gate_report_json,
                    inheritance, r_hat, version, confirmed_count, enrolled_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    mu_json=excluded.mu_json,
                    n=excluded.n,
                    sha256s_json=excluded.sha256s_json,
                    gate_report_json=excluded.gate_report_json,
                    inheritance=excluded.inheritance,
                    r_hat=excluded.r_hat,
                    version=excluded.version,
                    confirmed_count=excluded.confirmed_count,
                    enrolled_at=excluded.enrolled_at
            """, (
                cls.name, mu_json, cls.n, sha256s_json, gate_report_json,
                cls.inheritance, cls.r_hat, cls.version, cls.confirmed_count, cls.enrolled_at
            ))

    def delete_class(self, name: str) -> bool:
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM classes WHERE name = ?", (name,))
            return cur.rowcount > 0

    def get_class(self, name: str) -> EnrolledClass | None:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM classes WHERE name = ?", (name,))
            row = cur.fetchone()
            if not row:
                return None
            return EnrolledClass(
                name=row["name"],
                mu=np.array(json.loads(row["mu_json"]), dtype=np.float64),
                n=row["n"],
                embeddings=[],
                sha256s=json.loads(row["sha256s_json"]),
                gate_report=json.loads(row["gate_report_json"]),
                inheritance=row["inheritance"],
                r_hat=row["r_hat"],
                version=row["version"],
                confirmed_count=row["confirmed_count"],
                enrolled_at=row["enrolled_at"],
            )

    def list_classes(self) -> list[EnrolledClass]:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM classes ORDER BY enrolled_at ASC")
            classes = []
            for row in cur.fetchall():
                classes.append(EnrolledClass(
                    name=row["name"],
                    mu=np.array(json.loads(row["mu_json"]), dtype=np.float64),
                    n=row["n"],
                    embeddings=[],
                    sha256s=json.loads(row["sha256s_json"]),
                    gate_report=json.loads(row["gate_report_json"]),
                    inheritance=row["inheritance"],
                    r_hat=row["r_hat"],
                    version=row["version"],
                    confirmed_count=row["confirmed_count"],
                    enrolled_at=row["enrolled_at"],
                ))
            return classes

    def record_event(self, event_dict: dict[str, Any]) -> int:
        uuid_str = str(event_dict.get("uuid", ""))
        ts = event_dict.get("timestamp") or datetime.now(timezone.utc).isoformat()
        ver = int(event_dict.get("version", 0))
        etype = str(event_dict.get("type", "unknown"))
        payload_json = json.dumps(event_dict)

        with self._get_connection() as conn:
            cur = conn.execute(
                "INSERT INTO events (uuid, ts, version, type, payload_json) VALUES (?, ?, ?, ?, ?)",
                (uuid_str, ts, ver, etype, payload_json)
            )
            return cur.lastrowid

    def list_events(self, limit: int = 1000) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM events ORDER BY id ASC LIMIT ?", (limit,))
            return [json.loads(r["payload_json"]) for r in cur.fetchall()]

    def export_events_jsonl(self) -> str:
        events = self.list_events(limit=50000)
        return "\n".join(json.dumps(ev) for ev in events) + ("\n" if events else "")

    def export_events_csv(self) -> str:
        events = self.list_events(limit=50000)
        if not events:
            return "uuid,timestamp,version,type,class_name,operator_id\n"
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["uuid", "timestamp", "version", "type", "class_name", "operator_id", "operator_note"])
        for ev in events:
            writer.writerow([
                ev.get("uuid", ""),
                ev.get("timestamp", ""),
                ev.get("version", ""),
                ev.get("type", ""),
                ev.get("class_name", ""),
                ev.get("operator_id", ""),
                ev.get("operator_note", ""),
            ])
        return output.getvalue()

    def record_inspection(self, data: dict[str, Any]) -> int:
        ts = data.get("ts") or datetime.now(timezone.utc).isoformat()
        sha256 = str(data.get("image_sha256", ""))
        verdict = str(data.get("verdict", ""))
        class_name = data.get("class_name")
        novelty = float(data.get("novelty_score", 0.0))
        pred_set = json.dumps(data.get("prediction_set", []))
        details = json.dumps(data.get("details", {}))
        flag_op = 1 if data.get("flagged_for_op") else 0
        reviewed = 1 if data.get("reviewed") else 0
        review_lbl = data.get("review_label")
        review_nt = data.get("review_note")
        lat = float(data.get("latency_ms", 0.0))

        with self._get_connection() as conn:
            cur = conn.execute("""
                INSERT INTO inspections (
                    ts, image_sha256, verdict, class_name, novelty_score,
                    prediction_set_json, details_json, flagged_for_op,
                    reviewed, review_label, review_note, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ts, sha256, verdict, class_name, novelty,
                pred_set, details, flag_op, reviewed, review_lbl, review_nt, lat
            ))
            return cur.lastrowid

    def list_inspections(self, limit: int = 100, unreviewed_only: bool = False) -> list[dict[str, Any]]:
        query = "SELECT * FROM inspections"
        params: list[Any] = []
        if unreviewed_only:
            query += " WHERE flagged_for_op = 1 AND reviewed = 0"
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cur = conn.execute(query, params)
            rows = []
            for r in cur.fetchall():
                rows.append({
                    "id": r["id"],
                    "ts": r["ts"],
                    "image_sha256": r["image_sha256"],
                    "verdict": r["verdict"],
                    "class_name": r["class_name"],
                    "novelty_score": r["novelty_score"],
                    "prediction_set": json.loads(r["prediction_set_json"]),
                    "details": json.loads(r["details_json"]),
                    "flagged_for_op": bool(r["flagged_for_op"]),
                    "reviewed": bool(r["reviewed"]),
                    "review_label": r["review_label"],
                    "review_note": r["review_note"],
                    "latency_ms": r["latency_ms"],
                })
            return rows

    def update_inspection_review(self, inspection_id: int, review_label: str, review_note: str) -> bool:
        with self._get_connection() as conn:
            cur = conn.execute("""
                UPDATE inspections
                SET reviewed = 1, review_label = ?, review_note = ?
                WHERE id = ?
            """, (review_label, review_note, inspection_id))
            return cur.rowcount > 0

    def get_inspection_stats(self) -> dict[str, Any]:
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
            flagged = conn.execute("SELECT COUNT(*) FROM inspections WHERE flagged_for_op = 1").fetchone()[0]
            reviewed = conn.execute("SELECT COUNT(*) FROM inspections WHERE reviewed = 1").fetchone()[0]
            unreviewed = conn.execute("SELECT COUNT(*) FROM inspections WHERE flagged_for_op = 1 AND reviewed = 0").fetchone()[0]

            cur_v = conn.execute("SELECT verdict, COUNT(*) as cnt FROM inspections GROUP BY verdict")
            verdicts = {r["verdict"]: r["cnt"] for r in cur_v.fetchall()}

            cur_c = conn.execute("SELECT class_name, COUNT(*) as cnt FROM inspections WHERE class_name IS NOT NULL GROUP BY class_name")
            classes = {r["class_name"]: r["cnt"] for r in cur_c.fetchall()}

            avg_lat = conn.execute("SELECT AVG(latency_ms) FROM inspections").fetchone()[0] or 0.0

            return {
                "total_inspections": total,
                "flagged_count": flagged,
                "reviewed_count": reviewed,
                "unreviewed_queue_count": unreviewed,
                "verdict_distribution": verdicts,
                "class_distribution": classes,
                "avg_latency_ms": round(float(avg_lat), 2),
            }


class PostgresStorage(StorageBackend):
    """PostgreSQL implementation supporting Neon and standard PostgreSQL databases."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.init_schema()

    def get_provider_name(self) -> str:
        if "neon.tech" in self.database_url:
            return "Neon Postgres (production)"
        return "Postgres (production)"

    def _get_connection(self):
        import psycopg
        from psycopg.rows import dict_row
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def init_schema(self) -> None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS classes (
                        name TEXT PRIMARY KEY,
                        mu_json TEXT NOT NULL,
                        n INT NOT NULL,
                        sha256s_json TEXT NOT NULL,
                        gate_report_json TEXT NOT NULL,
                        inheritance TEXT NOT NULL,
                        r_hat REAL,
                        version INT NOT NULL DEFAULT 0,
                        confirmed_count INT NOT NULL DEFAULT 0,
                        enrolled_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS events (
                        id SERIAL PRIMARY KEY,
                        uuid TEXT NOT NULL,
                        ts TEXT NOT NULL,
                        version INT NOT NULL,
                        type TEXT NOT NULL,
                        payload_json TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS inspections (
                        id SERIAL PRIMARY KEY,
                        ts TEXT NOT NULL,
                        image_sha256 TEXT NOT NULL,
                        verdict TEXT NOT NULL,
                        class_name TEXT,
                        novelty_score REAL NOT NULL,
                        prediction_set_json TEXT NOT NULL,
                        details_json TEXT NOT NULL,
                        flagged_for_op BOOLEAN NOT NULL DEFAULT FALSE,
                        reviewed BOOLEAN NOT NULL DEFAULT FALSE,
                        review_label TEXT,
                        review_note TEXT,
                        latency_ms REAL NOT NULL
                    );
                """)
            conn.commit()

    def save_class(self, cls: EnrolledClass) -> None:
        mu_json = json.dumps(cls.mu.tolist() if isinstance(cls.mu, np.ndarray) else list(cls.mu))
        sha256s_json = json.dumps(cls.sha256s)
        gate_report_json = json.dumps(cls.gate_report)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO classes (
                        name, mu_json, n, sha256s_json, gate_report_json,
                        inheritance, r_hat, version, confirmed_count, enrolled_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT(name) DO UPDATE SET
                        mu_json=EXCLUDED.mu_json,
                        n=EXCLUDED.n,
                        sha256s_json=EXCLUDED.sha256s_json,
                        gate_report_json=EXCLUDED.gate_report_json,
                        inheritance=EXCLUDED.inheritance,
                        r_hat=EXCLUDED.r_hat,
                        version=EXCLUDED.version,
                        confirmed_count=EXCLUDED.confirmed_count,
                        enrolled_at=EXCLUDED.enrolled_at
                """, (
                    cls.name, mu_json, cls.n, sha256s_json, gate_report_json,
                    cls.inheritance, cls.r_hat, cls.version, cls.confirmed_count, cls.enrolled_at
                ))
            conn.commit()

    def delete_class(self, name: str) -> bool:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM classes WHERE name = %s", (name,))
                deleted = cur.rowcount > 0
            conn.commit()
            return deleted

    def get_class(self, name: str) -> EnrolledClass | None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM classes WHERE name = %s", (name,))
                row = cur.fetchone()
                if not row:
                    return None
                return EnrolledClass(
                    name=row["name"],
                    mu=np.array(json.loads(row["mu_json"]), dtype=np.float64),
                    n=row["n"],
                    embeddings=[],
                    sha256s=json.loads(row["sha256s_json"]),
                    gate_report=json.loads(row["gate_report_json"]),
                    inheritance=row["inheritance"],
                    r_hat=row["r_hat"],
                    version=row["version"],
                    confirmed_count=row["confirmed_count"],
                    enrolled_at=row["enrolled_at"],
                )

    def list_classes(self) -> list[EnrolledClass]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM classes ORDER BY enrolled_at ASC")
                classes = []
                for row in cur.fetchall():
                    classes.append(EnrolledClass(
                        name=row["name"],
                        mu=np.array(json.loads(row["mu_json"]), dtype=np.float64),
                        n=row["n"],
                        embeddings=[],
                        sha256s=json.loads(row["sha256s_json"]),
                        gate_report=json.loads(row["gate_report_json"]),
                        inheritance=row["inheritance"],
                        r_hat=row["r_hat"],
                        version=row["version"],
                        confirmed_count=row["confirmed_count"],
                        enrolled_at=row["enrolled_at"],
                    ))
                return classes

    def record_event(self, event_dict: dict[str, Any]) -> int:
        uuid_str = str(event_dict.get("uuid", ""))
        ts = event_dict.get("timestamp") or datetime.now(timezone.utc).isoformat()
        ver = int(event_dict.get("version", 0))
        etype = str(event_dict.get("type", "unknown"))
        payload_json = json.dumps(event_dict)

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO events (uuid, ts, version, type, payload_json) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                    (uuid_str, ts, ver, etype, payload_json)
                )
                row_id = cur.fetchone()["id"]
            conn.commit()
            return row_id

    def list_events(self, limit: int = 1000) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM events ORDER BY id ASC LIMIT %s", (limit,))
                return [json.loads(r["payload_json"]) for r in cur.fetchall()]

    def export_events_jsonl(self) -> str:
        events = self.list_events(limit=50000)
        return "\n".join(json.dumps(ev) for ev in events) + ("\n" if events else "")

    def export_events_csv(self) -> str:
        events = self.list_events(limit=50000)
        if not events:
            return "uuid,timestamp,version,type,class_name,operator_id,operator_note\n"
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["uuid", "timestamp", "version", "type", "class_name", "operator_id", "operator_note"])
        for ev in events:
            writer.writerow([
                ev.get("uuid", ""),
                ev.get("timestamp", ""),
                ev.get("version", ""),
                ev.get("type", ""),
                ev.get("class_name", ""),
                ev.get("operator_id", ""),
                ev.get("operator_note", ""),
            ])
        return output.getvalue()

    def record_inspection(self, data: dict[str, Any]) -> int:
        ts = data.get("ts") or datetime.now(timezone.utc).isoformat()
        sha256 = str(data.get("image_sha256", ""))
        verdict = str(data.get("verdict", ""))
        class_name = data.get("class_name")
        novelty = float(data.get("novelty_score", 0.0))
        pred_set = json.dumps(data.get("prediction_set", []))
        details = json.dumps(data.get("details", {}))
        flag_op = bool(data.get("flagged_for_op"))
        reviewed = bool(data.get("reviewed"))
        review_lbl = data.get("review_label")
        review_nt = data.get("review_note")
        lat = float(data.get("latency_ms", 0.0))

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO inspections (
                        ts, image_sha256, verdict, class_name, novelty_score,
                        prediction_set_json, details_json, flagged_for_op,
                        reviewed, review_label, review_note, latency_ms
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    ts, sha256, verdict, class_name, novelty,
                    pred_set, details, flag_op, reviewed, review_lbl, review_nt, lat
                ))
                ins_id = cur.fetchone()["id"]
            conn.commit()
            return ins_id

    def list_inspections(self, limit: int = 100, unreviewed_only: bool = False) -> list[dict[str, Any]]:
        query = "SELECT * FROM inspections"
        params: list[Any] = []
        if unreviewed_only:
            query += " WHERE flagged_for_op = TRUE AND reviewed = FALSE"
        query += " ORDER BY id DESC LIMIT %s"
        params.append(limit)

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = []
                for r in cur.fetchall():
                    rows.append({
                        "id": r["id"],
                        "ts": r["ts"],
                        "image_sha256": r["image_sha256"],
                        "verdict": r["verdict"],
                        "class_name": r["class_name"],
                        "novelty_score": r["novelty_score"],
                        "prediction_set": json.loads(r["prediction_set_json"]),
                        "details": json.loads(r["details_json"]),
                        "flagged_for_op": bool(r["flagged_for_op"]),
                        "reviewed": bool(r["reviewed"]),
                        "review_label": r["review_label"],
                        "review_note": r["review_note"],
                        "latency_ms": r["latency_ms"],
                    })
                return rows

    def update_inspection_review(self, inspection_id: int, review_label: str, review_note: str) -> bool:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE inspections
                    SET reviewed = TRUE, review_label = %s, review_note = %s
                    WHERE id = %s
                """, (review_label, review_note, inspection_id))
                updated = cur.rowcount > 0
            conn.commit()
            return updated

    def get_inspection_stats(self) -> dict[str, Any]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) as cnt FROM inspections")
                total = cur.fetchone()["cnt"]
                cur.execute("SELECT COUNT(*) as cnt FROM inspections WHERE flagged_for_op = TRUE")
                flagged = cur.fetchone()["cnt"]
                cur.execute("SELECT COUNT(*) as cnt FROM inspections WHERE reviewed = TRUE")
                reviewed = cur.fetchone()["cnt"]
                cur.execute("SELECT COUNT(*) as cnt FROM inspections WHERE flagged_for_op = TRUE AND reviewed = FALSE")
                unreviewed = cur.fetchone()["cnt"]

                cur.execute("SELECT verdict, COUNT(*) as cnt FROM inspections GROUP BY verdict")
                verdicts = {r["verdict"]: r["cnt"] for r in cur.fetchall()}

                cur.execute("SELECT class_name, COUNT(*) as cnt FROM inspections WHERE class_name IS NOT NULL GROUP BY class_name")
                classes = {r["class_name"]: r["cnt"] for r in cur.fetchall()}

                cur.execute("SELECT AVG(latency_ms) as avg_lat FROM inspections")
                avg_lat = cur.fetchone()["avg_lat"] or 0.0

                return {
                    "total_inspections": total,
                    "flagged_count": flagged,
                    "reviewed_count": reviewed,
                    "unreviewed_queue_count": unreviewed,
                    "verdict_distribution": verdicts,
                    "class_distribution": classes,
                    "avg_latency_ms": round(float(avg_lat), 2),
                }


def get_storage(database_url: str | None = None, db_path: Path | str | None = None) -> StorageBackend:
    """Factory to instantiate the appropriate storage backend.

    Priority:
      1. database_url parameter (if provided)
      2. DATABASE_URL environment variable
      3. SQLite file at db_path or default REPO_ROOT / 'state' / 'novasight.db'
    """
    url = database_url or os.environ.get("DATABASE_URL")
    if url and (url.startswith("postgres://") or url.startswith("postgresql://")):
        logger.info("Initializing PostgresStorage provider.")
        return PostgresStorage(url)
    return SQLiteStorage(db_path)
