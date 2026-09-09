from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import sqlite3


VALID_APPROVAL_STATUSES = {
    "pending",
    "executing",
    "approved",
    "failed",
}


@dataclass(frozen=True)
class ApprovalEntry:
    approval_id: str
    tool: str
    arguments: dict[str, Any]
    risk: str
    status: str
    result: dict[str, Any] | None
    created_at: str
    updated_at: str


class ApprovalStore:
    """
    Durable local approval store.

    Approval actions are persisted before execution so that an
    AI process restart does not lose pending approvals.

    State machine:

        pending
          |
          v
        executing
         /     \\
        v       v
    approved   failed

    An approval left in "executing" after a process crash is
    intentionally NOT returned to "pending" automatically.

    The side effect may already have happened, so automatic
    re-execution would be unsafe.
    """

    def __init__(
        self,
        path: str | Path,
    ) -> None:
        self.path = Path(path)

    # ============================================================
    # Initialization
    # ============================================================

    def initialize(self) -> None:
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS approvals (
                    approval_id TEXT PRIMARY KEY,
                    tool TEXT NOT NULL,
                    arguments TEXT NOT NULL,
                    risk TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result TEXT,
                    created_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    CHECK (
                        status IN (
                            'pending',
                            'executing',
                            'approved',
                            'failed'
                        )
                    )
                )
                """
            )

    # ============================================================
    # Create
    # ============================================================

    def create(
        self,
        approval_id: str,
        tool: str,
        arguments: dict[str, Any],
        risk: str,
    ) -> ApprovalEntry:
        if not approval_id:
            raise ValueError(
                "approval_id must not be empty"
            )

        if not tool:
            raise ValueError(
                "tool must not be empty"
            )

        if not isinstance(
            arguments,
            dict,
        ):
            raise ValueError(
                "arguments must be a dict"
            )

        if not risk:
            raise ValueError(
                "risk must not be empty"
            )

        encoded_arguments = (
            self._encode_object(
                arguments
            )
        )

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO approvals (
                    approval_id,
                    tool,
                    arguments,
                    risk,
                    status,
                    result
                )
                VALUES (?, ?, ?, ?, 'pending', NULL)
                """,
                (
                    approval_id,
                    tool,
                    encoded_arguments,
                    risk,
                ),
            )

        approval = self.get(
            approval_id
        )

        if approval is None:
            raise RuntimeError(
                "approval disappeared after creation"
            )

        return approval

    # ============================================================
    # Read
    # ============================================================

    def get(
        self,
        approval_id: str,
    ) -> ApprovalEntry | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    approval_id,
                    tool,
                    arguments,
                    risk,
                    status,
                    result,
                    created_at,
                    updated_at
                FROM approvals
                WHERE approval_id = ?
                """,
                (
                    approval_id,
                ),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_entry(
            row
        )

    def list_pending(
        self,
    ) -> list[ApprovalEntry]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    approval_id,
                    tool,
                    arguments,
                    risk,
                    status,
                    result,
                    created_at,
                    updated_at
                FROM approvals
                WHERE status = 'pending'
                ORDER BY created_at ASC
                """
            ).fetchall()

        return [
            self._row_to_entry(row)
            for row in rows
        ]

    # ============================================================
    # Execution claim
    # ============================================================

    def claim_for_execution(
        self,
        approval_id: str,
    ) -> bool:
        """
        Atomically transition:

            pending -> executing

        Returns True only for the caller that successfully
        claimed the approval.

        A concurrent caller receives False and must inspect the
        persisted state instead of executing the mutation.
        """

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE approvals
                SET
                    status = 'executing',
                    updated_at = CURRENT_TIMESTAMP
                WHERE approval_id = ?
                  AND status = 'pending'
                """,
                (
                    approval_id,
                ),
            )

            return cursor.rowcount == 1

    # ============================================================
    # Terminal states
    # ============================================================

    def mark_approved(
        self,
        approval_id: str,
        result: dict[str, Any],
    ) -> ApprovalEntry:
        return self._finish_execution(
            approval_id=approval_id,
            status="approved",
            result=result,
        )

    def mark_failed(
        self,
        approval_id: str,
        result: dict[str, Any],
    ) -> ApprovalEntry:
        return self._finish_execution(
            approval_id=approval_id,
            status="failed",
            result=result,
        )

    def _finish_execution(
        self,
        approval_id: str,
        status: str,
        result: dict[str, Any],
    ) -> ApprovalEntry:
        if status not in {
            "approved",
            "failed",
        }:
            raise ValueError(
                "terminal approval status must be "
                "'approved' or 'failed'"
            )

        if not isinstance(
            result,
            dict,
        ):
            raise ValueError(
                "result must be a dict"
            )

        encoded_result = (
            self._encode_object(
                result
            )
        )

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE approvals
                SET
                    status = ?,
                    result = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE approval_id = ?
                  AND status = 'executing'
                """,
                (
                    status,
                    encoded_result,
                    approval_id,
                ),
            )

            if cursor.rowcount != 1:
                raise ValueError(
                    "approval is not in executing state: "
                    f"{approval_id}"
                )

        approval = self.get(
            approval_id
        )

        if approval is None:
            raise RuntimeError(
                "approval disappeared after update"
            )

        return approval

    # ============================================================
    # Inspection
    # ============================================================

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM approvals
                """
            ).fetchone()

        return int(
            row["count"]
        )

    # ============================================================
    # SQLite
    # ============================================================

    def _connect(
        self,
    ) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=5.0,
        )

        connection.row_factory = (
            sqlite3.Row
        )

        connection.execute(
            "PRAGMA busy_timeout = 5000"
        )

        connection.execute(
            "PRAGMA journal_mode = WAL"
        )

        connection.execute(
            "PRAGMA synchronous = FULL"
        )

        return connection

    # ============================================================
    # Serialization
    # ============================================================

    @staticmethod
    def _encode_object(
        value: dict[str, Any],
    ) -> str:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @staticmethod
    def _decode_object(
        value: str,
    ) -> dict[str, Any]:
        decoded = json.loads(
            value
        )

        if not isinstance(
            decoded,
            dict,
        ):
            raise ValueError(
                "stored JSON must decode "
                "to an object"
            )

        return decoded

    def _row_to_entry(
        self,
        row: sqlite3.Row,
    ) -> ApprovalEntry:
        status = row[
            "status"
        ]

        if (
            status
            not in VALID_APPROVAL_STATUSES
        ):
            raise ValueError(
                "invalid stored approval status: "
                f"{status}"
            )

        result_json = row[
            "result"
        ]

        result = (
            None
            if result_json is None
            else self._decode_object(
                result_json
            )
        )

        return ApprovalEntry(
            approval_id=row[
                "approval_id"
            ],
            tool=row[
                "tool"
            ],
            arguments=self._decode_object(
                row["arguments"]
            ),
            risk=row[
                "risk"
            ],
            status=status,
            result=result,
            created_at=row[
                "created_at"
            ],
            updated_at=row[
                "updated_at"
            ],
        )