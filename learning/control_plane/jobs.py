"""Durable learning jobs: one row per job, one lease per attempt, one idempotency key.

The control plane owns this store; a GPU worker only ever holds a lease token. The token is a fence:
after a lease expires or is superseded, the old worker's completion cannot change the row.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

from learning.control_plane import canonical

JobState = Literal[
    "queued",
    "leased",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "needs_reconcile",
]

TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})
ACTIVE_STATES = frozenset({"leased", "running"})
ResolveOutcome = Literal["succeeded", "failed", "retry", "cancelled"]
SCHEMA_VERSION = 1


class JobStoreError(RuntimeError):
    """A control-plane job operation that cannot be honored."""


class JobNotFound(JobStoreError):
    pass


class JobConflict(JobStoreError):
    """The job is already in a state that forbids this transition."""


class IdempotencyConflict(JobStoreError):
    """A replayed idempotency key names a different job or payload."""


class LeaseLost(JobStoreError):
    """The caller no longer holds the active, unexpired lease."""


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    idempotency_key: str
    kind: str
    payload: dict[str, Any]
    payload_sha256: str
    state: JobState
    attempts: int
    max_attempts: int
    retry_safe: bool
    worker_id: str | None
    lease_token: str | None
    lease_expires_at: str | None
    heartbeat_at: str | None
    progress: dict[str, Any] | None
    result: dict[str, Any] | None
    result_sha256: str | None
    reason_code: str | None
    detail: str | None
    reconciled_at: str | None
    created_at: str
    updated_at: str

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def active(self) -> bool:
        return self.state in ACTIVE_STATES


@dataclass(frozen=True)
class Lease:
    job: JobRecord
    worker_id: str
    lease_token: str
    expires_at: str

    @property
    def job_id(self) -> str:
        return self.job.job_id

    @property
    def kind(self) -> str:
        return self.job.kind

    @property
    def payload(self) -> dict[str, Any]:
        return self.job.payload

    @property
    def attempt(self) -> int:
        return self.job.attempts


@dataclass(frozen=True)
class EnqueueResult:
    created: bool
    job: JobRecord


def _record(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        job_id=row["job_id"],
        idempotency_key=row["idempotency_key"],
        kind=row["kind"],
        payload=json.loads(row["payload_json"]),
        payload_sha256=row["payload_sha256"],
        state=row["state"],
        attempts=int(row["attempts"]),
        max_attempts=int(row["max_attempts"]),
        retry_safe=bool(row["retry_safe"]),
        worker_id=row["worker_id"],
        lease_token=row["lease_token"],
        lease_expires_at=row["lease_expires_at"],
        heartbeat_at=row["heartbeat_at"],
        progress=None if row["progress_json"] is None else json.loads(row["progress_json"]),
        result=None if row["result_json"] is None else json.loads(row["result_json"]),
        result_sha256=row["result_sha256"],
        reason_code=row["reason_code"],
        detail=row["detail"],
        reconciled_at=row["reconciled_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class JobStore:
    """A SQLite-backed durable job queue with leases, fencing and idempotent enqueue."""

    def __init__(self, path: Path, *, max_attempts: int = 3) -> None:
        if max_attempts < 1:
            raise canonical.ContractError("max_attempts must be at least 1")
        self.path = Path(path).expanduser().resolve()
        self.max_attempts = int(max_attempts)
        self._lock = threading.RLock()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise JobStoreError(
                    f"job database schema {version} is newer than this control plane "
                    f"({SCHEMA_VERSION})"
                )
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=FULL;

                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    state TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL,
                    retry_safe INTEGER NOT NULL DEFAULT 0,
                    worker_id TEXT,
                    lease_token TEXT,
                    lease_expires_at TEXT,
                    heartbeat_at TEXT,
                    progress_json TEXT,
                    result_json TEXT,
                    result_sha256 TEXT,
                    reason_code TEXT,
                    detail TEXT,
                    reconciled_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_jobs_state_created
                    ON jobs(state, created_at, job_id);

                CREATE INDEX IF NOT EXISTS idx_jobs_lease_expiry
                    ON jobs(state, lease_expires_at);

                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    sender TEXT,
                    received_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_messages_received
                    ON messages(received_at);
                """
            )
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA synchronous=FULL")
                yield conn
            finally:
                conn.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA synchronous=FULL")
                conn.execute("BEGIN IMMEDIATE")
                try:
                    yield conn
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
                else:
                    conn.execute("COMMIT")
            finally:
                conn.close()

    # -- enqueue and read -------------------------------------------------------------------------------

    def enqueue(self, *, job_id: str, idempotency_key: str, kind: str, payload: dict[str, Any],
                retry_safe: bool = False, max_attempts: int | None = None,
                now: datetime | None = None) -> EnqueueResult:
        """Add one job, or return the existing job for the same key and payload.

        A key that names a different job or payload is a conflict; it is never silently replaced.
        """
        canonical.require(canonical.JOB_ID, job_id, "job id")
        canonical.require(canonical.IDEMPOTENCY_KEY, idempotency_key, "idempotency key")
        canonical.require(canonical.KIND, kind, "job kind")
        if not isinstance(payload, dict):
            raise canonical.ContractError("a job payload must be an object")
        attempts = self.max_attempts if max_attempts is None else int(max_attempts)
        if attempts < 1:
            raise canonical.ContractError("max_attempts must be at least 1")
        payload_json = canonical.json_text(payload)
        payload_sha256 = canonical.digest(payload)
        stamp = canonical.iso(now)

        with self._write() as conn:
            existing = conn.execute(
                "SELECT * FROM jobs WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
            if existing is not None:
                record = _record(existing)
                if record.job_id == job_id and record.payload_sha256 == payload_sha256:
                    return EnqueueResult(created=False, job=record)
                raise IdempotencyConflict(
                    f"idempotency key {idempotency_key!r} already names another job or payload"
                )
            if conn.execute("SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)).fetchone() is not None:
                raise JobConflict(f"job {job_id!r} already exists under another idempotency key")
            conn.execute(
                """
                INSERT INTO jobs(
                    job_id, idempotency_key, kind, payload_json, payload_sha256, state,
                    attempts, max_attempts, retry_safe, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'queued', 0, ?, ?, ?, ?)
                """,
                (job_id, idempotency_key, kind, payload_json, payload_sha256, attempts,
                 int(bool(retry_safe)), stamp, stamp),
            )
            created = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            return EnqueueResult(created=True, job=_record(created))

    def get(self, job_id: str) -> JobRecord:
        canonical.require(canonical.JOB_ID, job_id, "job id")
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise JobNotFound(f"no job {job_id!r}")
        return _record(row)

    def find(self, job_id: str) -> JobRecord | None:
        canonical.require(canonical.JOB_ID, job_id, "job id")
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return None if row is None else _record(row)

    def list(self, *, state: JobState | None = None, limit: int = 100) -> list[JobRecord]:
        if limit < 1:
            raise canonical.ContractError("limit must be at least 1")
        query, params = "SELECT * FROM jobs", []
        if state is not None:
            query += " WHERE state = ?"
            params.append(state)
        query += " ORDER BY created_at, job_id LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            return [_record(row) for row in conn.execute(query, params).fetchall()]

    def counts(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute("SELECT state, COUNT(*) AS total FROM jobs GROUP BY state").fetchall()
        return {row["state"]: int(row["total"]) for row in rows}

    # -- leases -----------------------------------------------------------------------------------------

    def lease(self, *, worker_id: str, lease_seconds: float = 300.0,
              kinds: Iterable[str] | None = None, now: datetime | None = None) -> Lease | None:
        """Take the oldest queued job for this worker, after reconciling expired leases.

        Expiry is reconciled in the same transaction, so a worker that disappeared cannot hold a row
        forever. A retry-safe job returns to the queue; any other expired job needs an explicit resolve.
        """
        canonical.require(canonical.WORKER_ID, worker_id, "worker id")
        if lease_seconds <= 0:
            raise canonical.ContractError("lease_seconds must be positive")
        at = canonical.utc(now)
        if isinstance(kinds, str):
            kinds = [kinds]
        selected = [] if kinds is None else [canonical.require(canonical.KIND, kind, "job kind")
                                             for kind in kinds]
        timestamp = canonical.iso(at)
        expires_at = canonical.iso(at + timedelta(seconds=lease_seconds))

        with self._write() as conn:
            self._reconcile_expired(conn, at)
            if kinds is not None and not selected:
                return None
            query, params = "SELECT job_id FROM jobs WHERE state = 'queued'", []
            if selected:
                query += f" AND kind IN ({','.join('?' for _ in selected)})"
                params.extend(selected)
            query += " ORDER BY created_at, job_id LIMIT 1"
            candidate = conn.execute(query, params).fetchone()
            if candidate is None:
                return None
            token = str(uuid.uuid4())
            changed = conn.execute(
                """
                UPDATE jobs
                   SET state = 'leased', worker_id = ?, lease_token = ?, lease_expires_at = ?,
                       heartbeat_at = ?, attempts = attempts + 1, progress_json = NULL,
                       reason_code = NULL, detail = NULL, updated_at = ?
                 WHERE job_id = ? AND state = 'queued'
                """,
                (worker_id, token, expires_at, timestamp, timestamp, candidate["job_id"]),
            )
            if changed.rowcount != 1:
                return None
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (candidate["job_id"],)).fetchone()
            return Lease(job=_record(row), worker_id=worker_id, lease_token=token, expires_at=expires_at)

    def heartbeat(self, job_id: str, lease_token: str, *, extend_seconds: float = 300.0,
                  now: datetime | None = None) -> JobRecord:
        canonical.require(canonical.JOB_ID, job_id, "job id")
        if extend_seconds <= 0:
            raise canonical.ContractError("extend_seconds must be positive")
        at = canonical.utc(now)
        with self._write() as conn:
            self._held(conn, job_id, lease_token, at)
            expires_at = canonical.iso(at + timedelta(seconds=extend_seconds))
            stamp = canonical.iso(at)
            conn.execute(
                "UPDATE jobs SET lease_expires_at = ?, heartbeat_at = ?, updated_at = ? WHERE job_id = ?",
                (expires_at, stamp, stamp, job_id),
            )
            return _record(conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone())

    def start(self, job_id: str, lease_token: str, *, now: datetime | None = None) -> JobRecord:
        """Mark the worker's own start. A transport acknowledgement alone never reaches this state."""
        at = canonical.utc(now)
        with self._write() as conn:
            row = self._held(conn, job_id, lease_token, at)
            if row["state"] == "running":
                return _record(row)
            stamp = canonical.iso(at)
            conn.execute(
                "UPDATE jobs SET state = 'running', heartbeat_at = ?, updated_at = ? WHERE job_id = ?",
                (stamp, stamp, job_id),
            )
            return _record(conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone())

    def progress(self, job_id: str, lease_token: str, *, stage: str, fraction: float,
                 detail: str | None = None, now: datetime | None = None) -> JobRecord:
        canonical.require(canonical.STAGE, stage, "progress stage")
        if not isinstance(fraction, (int, float)) or isinstance(fraction, bool) or not 0.0 <= fraction <= 1.0:
            raise canonical.ContractError("progress fraction must be between 0 and 1")
        at = canonical.utc(now)
        payload = {"stage": stage, "fraction": float(fraction), "detail": canonical.detail(detail)}
        with self._write() as conn:
            row = self._held(conn, job_id, lease_token, at)
            if row["state"] != "running":
                raise JobConflict(f"job {job_id!r} has not started; progress follows the worker's start")
            stamp = canonical.iso(at)
            conn.execute(
                "UPDATE jobs SET progress_json = ?, heartbeat_at = ?, updated_at = ? WHERE job_id = ?",
                (canonical.json_text(payload), stamp, stamp, job_id),
            )
            return _record(conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone())

    def succeed(self, job_id: str, lease_token: str, *, result: dict[str, Any],
                now: datetime | None = None) -> JobRecord:
        """Complete with the worker's result. The same result is idempotent; another one conflicts."""
        canonical.require(canonical.JOB_ID, job_id, "job id")
        if not isinstance(result, dict):
            raise canonical.ContractError("a job result must be an object")
        result_json = canonical.json_text(result)
        result_sha256 = canonical.digest(result)
        at = canonical.utc(now)
        with self._write() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise JobNotFound(f"no job {job_id!r}")
            if row["state"] in TERMINAL_STATES:
                if row["state"] == "succeeded" and row["result_sha256"] == result_sha256:
                    return _record(row)
                raise JobConflict(f"job {job_id!r} already ended as {row['state']}")
            self._held(conn, job_id, lease_token, at, row=row)
            stamp = canonical.iso(at)
            conn.execute(
                """
                UPDATE jobs
                   SET state = 'succeeded', result_json = ?, result_sha256 = ?, lease_token = NULL,
                       lease_expires_at = NULL, progress_json = NULL, reason_code = NULL, detail = NULL,
                       updated_at = ?
                 WHERE job_id = ?
                """,
                (result_json, result_sha256, stamp, job_id),
            )
            return _record(conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone())

    def fail(self, job_id: str, lease_token: str, *, reason_code: str, detail: str | None = None,
             retryable: bool = False, now: datetime | None = None) -> JobRecord:
        """Record a known failure; only an explicitly retryable one returns to the queue."""
        canonical.require(canonical.JOB_ID, job_id, "job id")
        canonical.require(canonical.REASON_CODE, reason_code, "reason code")
        at = canonical.utc(now)
        with self._write() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise JobNotFound(f"no job {job_id!r}")
            if row["state"] in TERMINAL_STATES:
                if row["state"] == "failed" and row["reason_code"] == reason_code:
                    return _record(row)
                raise JobConflict(f"job {job_id!r} already ended as {row['state']}")
            self._held(conn, job_id, lease_token, at, row=row)
            retry = bool(retryable) and int(row["attempts"]) < int(row["max_attempts"])
            state = "queued" if retry else "failed"
            stamp = canonical.iso(at)
            conn.execute(
                """
                UPDATE jobs
                   SET state = ?, lease_token = NULL, lease_expires_at = NULL, progress_json = NULL,
                       reason_code = ?, detail = ?, updated_at = ?
                 WHERE job_id = ?
                """,
                (state, reason_code, canonical.detail(detail), stamp, job_id),
            )
            return _record(conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone())

    def cancel(self, job_id: str, *, reason_code: str = "cancelled", detail: str | None = None,
               now: datetime | None = None) -> JobRecord:
        """Stop a job that has not ended. A late worker result then loses its lease."""
        canonical.require(canonical.JOB_ID, job_id, "job id")
        canonical.require(canonical.REASON_CODE, reason_code, "reason code")
        at = canonical.utc(now)
        with self._write() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise JobNotFound(f"no job {job_id!r}")
            if row["state"] in TERMINAL_STATES:
                return _record(row)
            stamp = canonical.iso(at)
            conn.execute(
                """
                UPDATE jobs
                   SET state = 'cancelled', lease_token = NULL, lease_expires_at = NULL,
                       reason_code = ?, detail = ?, updated_at = ?
                 WHERE job_id = ?
                """,
                (reason_code, canonical.detail(detail), stamp, job_id),
            )
            return _record(conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone())

    # -- reconciliation ---------------------------------------------------------------------------------

    def reconcile_expired(self, *, now: datetime | None = None) -> list[JobRecord]:
        at = canonical.utc(now)
        with self._write() as conn:
            changed = self._reconcile_expired(conn, at)
            return [_record(conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone())
                    for job_id in changed]

    def _reconcile_expired(self, conn: sqlite3.Connection, at: datetime) -> list[str]:
        stamp = canonical.iso(at)
        rows = conn.execute(
            """
            SELECT job_id, retry_safe, attempts, max_attempts
              FROM jobs
             WHERE state IN ('leased', 'running')
               AND lease_expires_at IS NOT NULL
               AND lease_expires_at <= ?
             ORDER BY lease_expires_at, job_id
            """,
            (stamp,),
        ).fetchall()
        changed = []
        for row in rows:
            retry = bool(row["retry_safe"]) and int(row["attempts"]) < int(row["max_attempts"])
            state = "queued" if retry else "needs_reconcile"
            conn.execute(
                """
                UPDATE jobs
                   SET state = ?, lease_token = NULL, lease_expires_at = NULL,
                       heartbeat_at = NULL, progress_json = NULL, reason_code = 'lease_expired',
                       detail = ?, updated_at = ?
                 WHERE job_id = ?
                """,
                (state,
                 "the worker did not finish before its lease expired; the job is safe to retry"
                 if retry else "the worker did not finish before its lease expired",
                 stamp, row["job_id"]),
            )
            changed.append(str(row["job_id"]))
        return changed

    def resolve(self, job_id: str, *, outcome: ResolveOutcome, result: dict[str, Any] | None = None,
                reason_code: str = "reconciled", detail: str | None = None,
                now: datetime | None = None) -> JobRecord:
        """The explicit human/operator exit from ``needs_reconcile``; no socket may call it."""
        canonical.require(canonical.JOB_ID, job_id, "job id")
        canonical.require(canonical.REASON_CODE, reason_code, "reason code")
        at = canonical.utc(now)
        stamp = canonical.iso(at)
        with self._write() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise JobNotFound(f"no job {job_id!r}")
            if row["state"] != "needs_reconcile":
                raise JobConflict(f"job {job_id!r} is {row['state']}, not needs_reconcile")
            if outcome == "succeeded":
                if not isinstance(result, dict):
                    raise canonical.ContractError("a reconciled success must carry its result")
                conn.execute(
                    """
                    UPDATE jobs
                       SET state = 'succeeded', result_json = ?, result_sha256 = ?, reason_code = NULL,
                           detail = NULL, reconciled_at = ?, updated_at = ?
                     WHERE job_id = ?
                    """,
                    (canonical.json_text(result), canonical.digest(result), stamp, stamp, job_id),
                )
            elif outcome == "retry":
                if int(row["attempts"]) >= int(row["max_attempts"]):
                    raise JobConflict(f"job {job_id!r} has no attempt left")
                conn.execute(
                    """
                    UPDATE jobs
                       SET state = 'queued', progress_json = NULL, reason_code = ?, detail = ?,
                           reconciled_at = ?, updated_at = ?
                     WHERE job_id = ?
                    """,
                    (reason_code, canonical.detail(detail), stamp, stamp, job_id),
                )
            else:
                state = "failed" if outcome == "failed" else "cancelled"
                conn.execute(
                    """
                    UPDATE jobs
                       SET state = ?, reason_code = ?, detail = ?, reconciled_at = ?, updated_at = ?
                     WHERE job_id = ?
                    """,
                    (state, reason_code, canonical.detail(detail), stamp, stamp, job_id),
                )
            return _record(conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone())

    # -- replay guard -----------------------------------------------------------------------------------

    def claim_message(self, message_id: str, *, sender: str | None = None,
                      now: datetime | None = None) -> bool:
        """True the first time this message id is seen; False on a replay."""
        canonical.require(canonical.MESSAGE_ID, message_id, "message id")
        if sender is not None:
            canonical.require(canonical.SENDER, sender, "sender")
        with self._write() as conn:
            inserted = conn.execute(
                "INSERT OR IGNORE INTO messages(message_id, sender, received_at) VALUES (?, ?, ?)",
                (message_id, sender, canonical.iso(now)),
            )
            return inserted.rowcount == 1

    def prune_messages(self, *, older_than_seconds: float = 7 * 24 * 3600,
                       now: datetime | None = None) -> int:
        """Drop replay-guard rows older than the retention window.

        A pruned id can be claimed again after a long delay, which is safe because every store
        transition is fenced and idempotent; the guard only saves work for recent replays.
        """
        if older_than_seconds <= 0:
            raise canonical.ContractError("older_than_seconds must be positive")
        cutoff = canonical.iso(canonical.utc(now) - timedelta(seconds=older_than_seconds))
        with self._write() as conn:
            deleted = conn.execute("DELETE FROM messages WHERE received_at < ?", (cutoff,))
            return int(deleted.rowcount)

    # -- internals --------------------------------------------------------------------------------------

    @staticmethod
    def _held(conn: sqlite3.Connection, job_id: str, lease_token: str, at: datetime,
              *, row: sqlite3.Row | None = None) -> sqlite3.Row:
        current = row if row is not None else conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if current is None:
            raise JobNotFound(f"no job {job_id!r}")
        if current["state"] not in ACTIVE_STATES or current["lease_token"] != lease_token:
            raise LeaseLost(f"job {job_id!r} is not held by that lease")
        expires = current["lease_expires_at"]
        if expires is None or canonical.parse_iso(expires) <= at:
            raise LeaseLost(f"job {job_id!r}'s lease expired")
        return current
