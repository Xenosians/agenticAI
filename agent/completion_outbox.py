from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import sqlite3


@dataclass(frozen=True)
class OutboxEntry:
    job_id: str
    attempt: int
    payload: dict[str, Any]
    delivery_attempts: int
    last_error: str | None
    created_at: str


class CompletionOutbox:
    """
    Durable local completion outbox.

    Phoenix remains the durable owner of the public job state.
    This database only protects the Python -> Phoenix completion
    delivery boundary.

    Identity:
        (job_id, attempt)

    Once a completion payload is persisted for a job attempt,
    that payload is immutable.
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
                CREATE TABLE IF NOT EXISTS completion_outbox (
                    job_id TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    delivery_attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

                    PRIMARY KEY (
                        job_id,
                        attempt
                    )
                )
                """
            )

    # ============================================================
    # Persist completion
    # ============================================================

    def put(
        self,
        job_id: str,
        attempt: int,
        payload: dict[str, Any],
    ) -> None:
        if not job_id:
            raise ValueError(
                "job_id must not be empty"
            )

        if attempt < 1:
            raise ValueError(
                "attempt must be >= 1"
            )

        encoded_payload = self._encode_payload(
            payload
        )

        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT payload
                FROM completion_outbox
                WHERE job_id = ?
                  AND attempt = ?
                """,
                (
                    job_id,
                    attempt,
                ),
            ).fetchone()

            if existing is not None:
                existing_payload = existing[
                    "payload"
                ]

                if (
                    existing_payload
                    != encoded_payload
                ):
                    raise ValueError(
                        "completion payload changed "
                        f"for job_id={job_id} "
                        f"attempt={attempt}"
                    )

                return

            connection.execute(
                """
                INSERT INTO completion_outbox (
                    job_id,
                    attempt,
                    payload
                )
                VALUES (?, ?, ?)
                """,
                (
                    job_id,
                    attempt,
                    encoded_payload,
                ),
            )

    # ============================================================
    # Read pending deliveries
    # ============================================================

    def list_pending(
        self,
        limit: int = 100,
    ) -> list[OutboxEntry]:
        if limit < 1:
            raise ValueError(
                "limit must be >= 1"
            )

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    job_id,
                    attempt,
                    payload,
                    delivery_attempts,
                    last_error,
                    created_at
                FROM completion_outbox
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            self._row_to_entry(row)
            for row in rows
        ]

    # ============================================================
    # Delivery bookkeeping
    # ============================================================

    def mark_delivery_attempt(
        self,
        job_id: str,
        attempt: int,
        error: str | None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE completion_outbox
                SET
                    delivery_attempts =
                        delivery_attempts + 1,
                    last_error = ?
                WHERE job_id = ?
                  AND attempt = ?
                """,
                (
                    error,
                    job_id,
                    attempt,
                ),
            )

    def delete(
        self,
        job_id: str,
        attempt: int,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM completion_outbox
                WHERE job_id = ?
                  AND attempt = ?
                """,
                (
                    job_id,
                    attempt,
                ),
            )

    # ============================================================
    # Inspection helpers
    # ============================================================

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM completion_outbox
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
    def _encode_payload(
        payload: dict[str, Any],
    ) -> str:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @staticmethod
    def _decode_payload(
        payload: str,
    ) -> dict[str, Any]:
        decoded = json.loads(
            payload
        )

        if not isinstance(
            decoded,
            dict,
        ):
            raise ValueError(
                "outbox payload must decode "
                "to a JSON object"
            )

        return decoded

    def _row_to_entry(
        self,
        row: sqlite3.Row,
    ) -> OutboxEntry:
        return OutboxEntry(
            job_id=row["job_id"],
            attempt=row["attempt"],
            payload=self._decode_payload(
                row["payload"]
            ),
            delivery_attempts=
                row[
                    "delivery_attempts"
                ],
            last_error=row[
                "last_error"
            ],
            created_at=row[
                "created_at"
            ],
        )