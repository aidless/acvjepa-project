"""SQLite lease ledger for resumable, idempotent SimJob generation.

Use this ledger on durable shared storage (or replace with a transactional database
in production). A worker owns a job only while its lease is fresh. Completion is
accepted only with an immutable artifact/dataset commit reference and matching
hashes. Retried workers can safely observe an already-completed matching job and
skip it.
"""
from __future__ import annotations

import json
import random
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


@dataclass(frozen=True)
class Lease:
    job_key: str
    worker_id: str
    attempt: int
    lease_until_ns: int
    payload: dict


class LeaseLedger:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=30.0, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS sim_jobs (
                 job_key TEXT PRIMARY KEY,
                 payload_json TEXT NOT NULL,
                 status TEXT NOT NULL,
                 attempt INTEGER NOT NULL DEFAULT 0,
                 worker_id TEXT,
                 lease_until_ns INTEGER,
                 artifact_sha256 TEXT,
                 metadata_sha256 TEXT,
                 remote_commit_uri TEXT,
                 error TEXT,
                 updated_ns INTEGER NOT NULL
            )"""
        )

    @staticmethod
    def now_ns() -> int:
        return time.time_ns()

    def register(self, job_key: str, payload: dict) -> None:
        """Idempotently register a stable job key; mismatched payload is a conflict."""
        encoded = json.dumps(payload, sort_keys=True)
        now = self.now_ns()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            old = self.conn.execute("SELECT payload_json FROM sim_jobs WHERE job_key=?", (job_key,)).fetchone()
            if old is None:
                self.conn.execute(
                    "INSERT INTO sim_jobs(job_key,payload_json,status,updated_ns) VALUES(?,?,?,?)",
                    (job_key, encoded, "PENDING", now),
                )
            elif old["payload_json"] != encoded:
                raise RuntimeError(f"job key conflict with different payload: {job_key}")
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def acquire(self, worker_id: str, lease_seconds: float) -> Optional[Lease]:
        """Claim one pending/retry/expired job using a short transaction."""
        now = self.now_ns()
        lease_until = now + int(lease_seconds * 1e9)
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = self.conn.execute(
                """SELECT * FROM sim_jobs
                   WHERE status IN ('PENDING','RETRY')
                      OR (status='LEASED' AND lease_until_ns < ?)
                   ORDER BY updated_ns, job_key LIMIT 1""",
                (now,),
            ).fetchone()
            if row is None:
                self.conn.execute("COMMIT")
                return None
            attempt = int(row["attempt"]) + 1
            self.conn.execute(
                """UPDATE sim_jobs SET status='LEASED', attempt=?, worker_id=?,
                   lease_until_ns=?, error=NULL, updated_ns=? WHERE job_key=?""",
                (attempt, worker_id, lease_until, now, row["job_key"]),
            )
            self.conn.execute("COMMIT")
            return Lease(row["job_key"], worker_id, attempt, lease_until, json.loads(row["payload_json"]))
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def heartbeat(self, lease: Lease, lease_seconds: float) -> Lease:
        now = self.now_ns()
        renewed = now + int(lease_seconds * 1e9)
        result = self.conn.execute(
            """UPDATE sim_jobs SET lease_until_ns=?, updated_ns=?
               WHERE job_key=? AND status='LEASED' AND worker_id=? AND attempt=?""",
            (renewed, now, lease.job_key, lease.worker_id, lease.attempt),
        )
        if result.rowcount != 1:
            raise RuntimeError(f"lease lost: {lease.job_key}")
        return Lease(lease.job_key, lease.worker_id, lease.attempt, renewed, lease.payload)

    def complete(self, lease: Lease, *, artifact_sha256: str, metadata_sha256: str, remote_commit_uri: str) -> None:
        if not (artifact_sha256 and metadata_sha256 and remote_commit_uri):
            raise ValueError("complete requires verified hashes and remote commit")
        now = self.now_ns()
        result = self.conn.execute(
            """UPDATE sim_jobs SET status='COMPLETE', artifact_sha256=?, metadata_sha256=?,
               remote_commit_uri=?, lease_until_ns=NULL, updated_ns=?
               WHERE job_key=? AND status='LEASED' AND worker_id=? AND attempt=?""",
            (artifact_sha256, metadata_sha256, remote_commit_uri, now, lease.job_key, lease.worker_id, lease.attempt),
        )
        if result.rowcount != 1:
            raise RuntimeError(f"cannot complete missing/lost lease: {lease.job_key}")

    def retry(self, lease: Lease, error: str, max_attempts: int) -> None:
        now = self.now_ns()
        next_status = "RETRY" if lease.attempt < max_attempts else "QUARANTINED"
        result = self.conn.execute(
            """UPDATE sim_jobs SET status=?, lease_until_ns=NULL, error=?, updated_ns=?
               WHERE job_key=? AND status='LEASED' AND worker_id=? AND attempt=?""",
            (next_status, error[:2048], now, lease.job_key, lease.worker_id, lease.attempt),
        )
        if result.rowcount != 1:
            raise RuntimeError(f"cannot retry lost lease: {lease.job_key}")

    def summary(self) -> dict:
        rows = self.conn.execute("SELECT status, COUNT(*) AS count FROM sim_jobs GROUP BY status").fetchall()
        return {row["status"]: row["count"] for row in rows}


def retry_with_backoff(operation: Callable[[], object], *, attempts: int = 5, base_seconds: float = 0.5) -> object:
    """Retry only idempotent stages such as staging upload or commit verification."""
    last_error: Optional[Exception] = None
    for index in range(attempts):
        try:
            return operation()
        except Exception as exc:  # caller must record this in its lease ledger
            last_error = exc
            if index == attempts - 1:
                break
            time.sleep(base_seconds * (2**index) + random.uniform(0.0, base_seconds))
    assert last_error is not None
    raise last_error
