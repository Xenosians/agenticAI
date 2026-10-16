"""Crash-safe continuation claims in the existing approval SQLite database."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import sqlite3


class WorkflowStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS account_ticket_continuation (
                    source_approval_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    attempt INTEGER NOT NULL CHECK(attempt >= 1),
                    state TEXT NOT NULL CHECK(state IN ('pending', 'preparing', 'ready', 'blocked')),
                    plan TEXT NOT NULL,
                    response TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

    @staticmethod
    def _encode(value: dict) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def put(self, *, source_approval_id: str, owner_id: str, job_id: str,
            attempt: int, plan: dict[str, Any]) -> None:
        if not source_approval_id or not owner_id or not job_id or type(attempt) is not int or attempt < 1:
            raise ValueError("Incomplete onboarding job identity.")
        encoded = self._encode(plan)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT * FROM account_ticket_continuation WHERE source_approval_id=?",
                (source_approval_id,),
            ).fetchone()
            if existing is not None:
                if (existing["owner_id"], existing["job_id"], existing["attempt"], existing["plan"]) != (
                    owner_id, job_id, attempt, encoded
                ):
                    raise ValueError("An approval cannot be rebound to another workflow.")
                return
            # A workflow must be durably attached before the source action can
            # execute. Never attach a newly supplied plan to an old approval.
            approval = db.execute(
                "SELECT tool, status FROM approvals WHERE approval_id=?", (source_approval_id,)
            ).fetchone()
            if approval is None or approval["tool"] != "account_create" or approval["status"] != "pending":
                raise ValueError("Onboarding requires a pending account approval.")
            db.execute("""
                INSERT INTO account_ticket_continuation
                    (source_approval_id, owner_id, job_id, attempt, state, plan)
                VALUES (?, ?, ?, ?, 'pending', ?)
            """, (source_approval_id, owner_id, job_id, attempt, encoded))

    def get(self, source_approval_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM account_ticket_continuation WHERE source_approval_id=?",
                (source_approval_id,),
            ).fetchone()
        if row is None:
            return None
        return {**dict(row), "plan": json.loads(row["plan"]),
                "response": json.loads(row["response"]) if row["response"] else None}

    def claim(self, source_approval_id: str) -> bool:
        with self.connect() as db:
            return db.execute("""
                UPDATE account_ticket_continuation SET state='preparing', updated_at=CURRENT_TIMESTAMP
                WHERE source_approval_id=? AND state='pending'
            """, (source_approval_id,)).rowcount == 1

    def finish(self, source_approval_id: str, response: dict[str, Any], *, blocked: bool = False) -> None:
        with self.connect() as db:
            count = db.execute("""
                UPDATE account_ticket_continuation SET state=?, response=?, updated_at=CURRENT_TIMESTAMP
                WHERE source_approval_id=? AND state='preparing'
            """, ("blocked" if blocked else "ready", self._encode(response), source_approval_id)).rowcount
            if count != 1:
                raise ValueError("Onboarding continuation was not claimed.")
