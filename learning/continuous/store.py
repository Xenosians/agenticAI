from __future__ import annotations

import json
import sqlite3
import threading
import time

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from learning.continuous.types import (
    AdaptiveRecipe,
    ContinuousCyclePlan,
    ContinuousCycleResult,
    CorpusPage,
    CorpusSource,
    LearningEvent,
)


class ContinuousLearningStore:
    """
    Durable local control-plane store.

    SQLite is used deliberately here:
    - crash-safe enough for one host;
    - atomic claims;
    - simple deduplication;
    - no new external infrastructure requirement.

    Training artifacts remain immutable files under .runtime/learning.
    """

    def __init__(
        self,
        path: Path,
    ) -> None:
        self.path = (
            path
            .expanduser()
            .resolve()
        )
        self._lock = threading.RLock()

    def initialize(
        self,
    ) -> None:
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=FULL;

                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    trajectory_id TEXT NOT NULL,
                    job_id TEXT,
                    model_key TEXT NOT NULL,
                    hub_status TEXT,
                    had_error INTEGER NOT NULL,
                    waiting_approval INTEGER NOT NULL,
                    execution_reward REAL NOT NULL,
                    state TEXT NOT NULL,
                    claimed_cycle_id TEXT,
                    payload_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_events_state_created
                    ON events(state, created_at);

                CREATE TABLE IF NOT EXISTS corpus_sources (
                    source_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    target_role TEXT NOT NULL,
                    audience_json TEXT NOT NULL,
                    trusted INTEGER NOT NULL,
                    training_eligible INTEGER NOT NULL,
                    page_chars INTEGER NOT NULL,
                    records_per_page INTEGER NOT NULL,
                    source_sha256 TEXT,
                    cursor INTEGER NOT NULL DEFAULT 0,
                    total_units INTEGER,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS corpus_pages (
                    page_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    page_index INTEGER NOT NULL,
                    cursor_start INTEGER NOT NULL,
                    cursor_end INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    claimed_cycle_id TEXT,
                    payload_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_corpus_pages_state
                    ON corpus_pages(state, source_id, page_index);

                CREATE TABLE IF NOT EXISTS cycles (
                    cycle_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    state TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    result_json TEXT
                );

                CREATE TABLE IF NOT EXISTS kv (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )

    @contextmanager
    def _connect(
        self,
    ) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = sqlite3.connect(
                self.path,
                timeout=30.0,
                isolation_level=None,
            )
            conn.row_factory = sqlite3.Row

            try:
                yield conn
            finally:
                conn.close()

    def enqueue_event(
        self,
        event: LearningEvent,
    ) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO events(
                    event_id,
                    created_at,
                    trajectory_id,
                    job_id,
                    model_key,
                    hub_status,
                    had_error,
                    waiting_approval,
                    execution_reward,
                    state,
                    claimed_cycle_id,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', NULL, ?)
                """,
                (
                    event.event_id,
                    event.created_at,
                    event.trajectory_id,
                    event.job_id,
                    event.model_key,
                    event.hub_status,
                    int(
                        event.had_error
                    ),
                    int(
                        event.waiting_approval
                    ),
                    event.execution_reward,
                    event.model_dump_json(
                        by_alias=True
                    ),
                ),
            )

            return cursor.rowcount > 0

    def pending_event_counts(
        self,
    ) -> dict[str, int]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN had_error = 1 THEN 1 ELSE 0 END) AS failures
                FROM events
                WHERE state = 'pending'
                """
            ).fetchone()

        return {
            "total":
                int(
                    row["total"]
                    or 0
                ),

            "failures":
                int(
                    row["failures"]
                    or 0
                ),
        }

    def oldest_pending_event_age_seconds(
        self,
    ) -> float | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT created_at
                FROM events
                WHERE state = 'pending'
                ORDER BY created_at ASC
                LIMIT 1
                """
            ).fetchone()

        if row is None:
            return None

        try:
            from datetime import datetime

            created = datetime.fromisoformat(
                row["created_at"]
            )

            now = datetime.now(
                created.tzinfo
            )

            return max(
                0.0,
                (
                    now
                    - created
                ).total_seconds(),
            )

        except Exception:
            return None

    def claim_events(
        self,
        *,
        cycle_id: str,
        limit: int,
    ) -> list[LearningEvent]:
        with self._connect() as conn:
            conn.execute(
                "BEGIN IMMEDIATE"
            )

            rows = conn.execute(
                """
                SELECT event_id, payload_json
                FROM events
                WHERE state = 'pending'
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (
                    limit,
                ),
            ).fetchall()

            ids = [
                row["event_id"]
                for row in rows
            ]

            if ids:
                placeholders = ",".join(
                    "?"
                    for _ in ids
                )

                conn.execute(
                    f"""
                    UPDATE events
                    SET state = 'claimed',
                        claimed_cycle_id = ?
                    WHERE event_id IN ({placeholders})
                    """,
                    (
                        cycle_id,
                        *ids,
                    ),
                )

            conn.execute(
                "COMMIT"
            )

        return [
            LearningEvent.model_validate_json(
                row["payload_json"]
            )
            for row in rows
        ]

    def finalize_events(
        self,
        *,
        cycle_id: str,
        consumed: bool,
    ) -> None:
        state = (
            "consumed"
            if consumed
            else "pending"
        )

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE events
                SET state = ?,
                    claimed_cycle_id = NULL
                WHERE claimed_cycle_id = ?
                """,
                (
                    state,
                    cycle_id,
                ),
            )

    def upsert_corpus_source(
        self,
        source: CorpusSource,
        *,
        source_sha256: str | None = None,
        total_units: int | None = None,
    ) -> None:
        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT source_sha256, cursor
                FROM corpus_sources
                WHERE source_id = ?
                """,
                (
                    source.source_id,
                ),
            ).fetchone()

            cursor = 0

            if existing is not None:
                if (
                    source_sha256
                    and existing[
                        "source_sha256"
                    ]
                    == source_sha256
                ):
                    cursor = int(
                        existing[
                            "cursor"
                        ]
                    )

            conn.execute(
                """
                INSERT INTO corpus_sources(
                    source_id,
                    kind,
                    path,
                    target_role,
                    audience_json,
                    trusted,
                    training_eligible,
                    page_chars,
                    records_per_page,
                    source_sha256,
                    cursor,
                    total_units,
                    enabled,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    kind = excluded.kind,
                    path = excluded.path,
                    target_role = excluded.target_role,
                    audience_json = excluded.audience_json,
                    trusted = excluded.trusted,
                    training_eligible = excluded.training_eligible,
                    page_chars = excluded.page_chars,
                    records_per_page = excluded.records_per_page,
                    source_sha256 = excluded.source_sha256,
                    cursor = excluded.cursor,
                    total_units = excluded.total_units,
                    enabled = 1,
                    updated_at = excluded.updated_at
                """,
                (
                    source.source_id,
                    source.kind,
                    source.path,
                    source.target_role,
                    json.dumps(
                        source.audience,
                        sort_keys=True,
                    ),
                    int(
                        source.trusted
                    ),
                    int(
                        source.training_eligible
                    ),
                    source.page_chars,
                    source.records_per_page,
                    source_sha256,
                    cursor,
                    total_units,
                    time.time(),
                ),
            )

    def corpus_source_rows(
        self,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM corpus_sources
                WHERE enabled = 1
                  AND trusted = 1
                  AND training_eligible = 1
                ORDER BY source_id ASC
                """
            ).fetchall()

        return [
            dict(
                row
            )
            for row in rows
        ]

    def set_corpus_cursor(
        self,
        *,
        source_id: str,
        cursor: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE corpus_sources
                SET cursor = ?,
                    updated_at = ?
                WHERE source_id = ?
                """,
                (
                    cursor,
                    time.time(),
                    source_id,
                ),
            )

    def remember_page(
        self,
        page: CorpusPage,
        *,
        state: str = "available",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO corpus_pages(
                    page_id,
                    source_id,
                    page_index,
                    cursor_start,
                    cursor_end,
                    state,
                    claimed_cycle_id,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?)
                """,
                (
                    page.page_id,
                    page.source_id,
                    page.page_index,
                    page.cursor_start,
                    page.cursor_end,
                    state,
                    page.model_dump_json(
                        by_alias=True
                    ),
                ),
            )

    def claim_pages(
        self,
        *,
        cycle_id: str,
        page_ids: list[str],
    ) -> None:
        if not page_ids:
            return

        placeholders = ",".join(
            "?"
            for _ in page_ids
        )

        with self._connect() as conn:
            conn.execute(
                f"""
                UPDATE corpus_pages
                SET state = 'claimed',
                    claimed_cycle_id = ?
                WHERE page_id IN ({placeholders})
                """,
                (
                    cycle_id,
                    *page_ids,
                ),
            )

    def finalize_pages(
        self,
        *,
        cycle_id: str,
        consumed: bool,
    ) -> None:
        with self._connect() as conn:
            if consumed:
                conn.execute(
                    """
                    UPDATE corpus_pages
                    SET state = 'consumed',
                        claimed_cycle_id = NULL
                    WHERE claimed_cycle_id = ?
                    """,
                    (
                        cycle_id,
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE corpus_pages
                    SET state = 'available',
                        claimed_cycle_id = NULL
                    WHERE claimed_cycle_id = ?
                    """,
                    (
                        cycle_id,
                    ),
                )

    def available_pages(
        self,
        *,
        limit: int,
    ) -> list[CorpusPage]:
        if limit <= 0:
            return []

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload_json
                FROM corpus_pages
                WHERE state = 'available'
                ORDER BY source_id ASC, page_index ASC
                LIMIT ?
                """,
                (
                    limit,
                ),
            ).fetchall()

        return [
            CorpusPage.model_validate_json(
                row["payload_json"]
            )
            for row in rows
        ]

    def replay_pages(
        self,
        *,
        limit: int,
    ) -> list[CorpusPage]:
        if limit <= 0:
            return []

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload_json
                FROM corpus_pages
                WHERE state = 'consumed'
                ORDER BY page_index DESC, page_id ASC
                LIMIT ?
                """,
                (
                    limit,
                ),
            ).fetchall()

        result = []

        for row in rows:
            page = CorpusPage.model_validate_json(
                row["payload_json"]
            )

            result.append(
                page.model_copy(
                    update={
                        "replay":
                            True,
                    }
                )
            )

        return result

    def save_cycle_plan(
        self,
        plan: ContinuousCyclePlan,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cycles(
                    cycle_id,
                    created_at,
                    state,
                    plan_json,
                    result_json
                ) VALUES (?, ?, ?, ?, NULL)
                """,
                (
                    plan.cycle_id,
                    plan.created_at,
                    plan.state,
                    plan.model_dump_json(
                        by_alias=True
                    ),
                ),
            )

    def save_cycle_result(
        self,
        result: ContinuousCycleResult,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE cycles
                SET state = ?,
                    result_json = ?
                WHERE cycle_id = ?
                """,
                (
                    result.state,
                    result.model_dump_json(
                        by_alias=True
                    ),
                    result.cycle_id,
                ),
            )

    def cycle_count(
        self,
    ) -> int:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM cycles
                """
            ).fetchone()

        return int(
            row["count"]
            or 0
        )

    def latest_cycle(
        self,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM cycles
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).fetchone()

        return (
            dict(
                row
            )
            if row is not None
            else None
        )

    def set_json(
        self,
        key: str,
        value: Any,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO kv(
                    key,
                    value_json,
                    updated_at
                ) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (
                    key,
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    time.time(),
                ),
            )

    def get_json(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT value_json
                FROM kv
                WHERE key = ?
                """,
                (
                    key,
                ),
            ).fetchone()

        if row is None:
            return default

        return json.loads(
            row["value_json"]
        )

    def load_recipe(
        self,
        default: AdaptiveRecipe,
    ) -> AdaptiveRecipe:
        raw = self.get_json(
            "adaptive_recipe"
        )

        if raw is None:
            return default

        return AdaptiveRecipe.model_validate(
            raw
        )

    def save_recipe(
        self,
        recipe: AdaptiveRecipe,
    ) -> None:
        self.set_json(
            "adaptive_recipe",
            recipe.model_dump(
                mode="json"
            ),
        )
