from __future__ import annotations

import json
import os
import threading
import uuid

from datetime import (
    datetime,
    timezone,
)

from pathlib import (
    Path,
)

from typing import (
    Any,
)

from learning.sanitizer import (
    sanitize_value,
)

from learning.types import (
    CorrectionEvent,
    CorrectionValue,
)


VALID_CORRECTION_TYPES = {
    "route",
    "tool_selection",
    "argument",
    "repository_scope",
    "ticket_scope",
    "workspace_scope",
    "identifier",
    "answer",
    "other",
}


VALID_CORRECTION_SOURCES = {
    "explicit_user",
    "trusted_review",
    "evaluation",
}


class CorrectionRecorder:
    """
    Append-only correction evidence store.

    Corrections remain separate from raw trajectories so original
    runtime evidence is never rewritten.
    """

    def __init__(
        self,
        *,
        path: Path,
        enabled: bool,
    ) -> None:

        self.path = (
            path
            .expanduser()
            .resolve()
        )

        self.enabled = (
            enabled
        )

        self._lock = (
            threading.Lock()
        )

    def build(
        self,
        *,
        trajectory_id: str,
        correction_type: str,
        values: list[
            dict[
                str,
                Any,
            ]
        ],
        task_id: str | None = None,
        source: str = (
            "explicit_user"
        ),
        note: str | None = None,
    ) -> CorrectionEvent:

        normalized_trajectory_id = (
            trajectory_id
            .strip()
        )

        if not normalized_trajectory_id:
            raise ValueError(
                "trajectory_id must not be empty."
            )

        normalized_task_id = None

        if task_id is not None:
            normalized_task_id = (
                task_id
                .strip()
            )

            if not normalized_task_id:
                raise ValueError(
                    "task_id must not be empty "
                    "when provided."
                )

        normalized_type = (
            correction_type
            .strip()
            .lower()
        )

        if (
            normalized_type
            not in VALID_CORRECTION_TYPES
        ):
            raise ValueError(
                "Unsupported correction_type: "
                f"{correction_type}"
            )

        normalized_source = (
            source
            .strip()
            .lower()
        )

        if (
            normalized_source
            not in VALID_CORRECTION_SOURCES
        ):
            raise ValueError(
                "Unsupported correction source: "
                f"{source}"
            )

        correction_values: list[
            CorrectionValue
        ] = []

        for item in values:
            field_name = (
                str(
                    item.get(
                        "field",
                        "",
                    )
                )
                .strip()
            )

            if not field_name:
                raise ValueError(
                    "Correction value field "
                    "must not be empty."
                )

            correction_values.append(
                CorrectionValue(
                    field=(
                        field_name
                    ),

                    rejected_value=(
                        item.get(
                            "rejected_value"
                        )
                    ),

                    chosen_value=(
                        item.get(
                            "chosen_value"
                        )
                    ),
                )
            )

        if not correction_values:
            raise ValueError(
                "At least one corrected value "
                "is required."
            )

        return (
            CorrectionEvent(
                correction_id=(
                    uuid
                    .uuid4()
                    .hex
                ),

                observed_at=(
                    datetime
                    .now(
                        timezone.utc
                    )
                    .isoformat()
                ),

                trajectory_id=(
                    normalized_trajectory_id
                ),

                task_id=(
                    normalized_task_id
                ),

                correction_type=(
                    normalized_type
                ),

                source=(
                    normalized_source
                ),

                values=(
                    correction_values
                ),

                note=(
                    note
                ),

                dataset_eligible=False,
            )
        )

    def record(
        self,
        *,
        trajectory_id: str,
        correction_type: str,
        values: list[
            dict[
                str,
                Any,
            ]
        ],
        task_id: str | None = None,
        source: str = (
            "explicit_user"
        ),
        note: str | None = None,
    ) -> dict | None:

        if not self.enabled:
            return None

        correction = (
            self.build(
                trajectory_id=(
                    trajectory_id
                ),

                task_id=(
                    task_id
                ),

                correction_type=(
                    correction_type
                ),

                values=(
                    values
                ),

                source=(
                    source
                ),

                note=(
                    note
                ),
            )
        )

        raw_payload = (
            correction
            .model_dump(
                mode="json",
                by_alias=True,
            )
        )

        payload = (
            sanitize_value(
                raw_payload
            )
        )

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        serialized = (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
            )
        )

        with self._lock:

            with self.path.open(
                "a",
                encoding="utf-8",
            ) as handle:

                handle.write(
                    serialized
                )

                handle.write(
                    "\n"
                )

                handle.flush()

                os.fsync(
                    handle.fileno()
                )

        return payload