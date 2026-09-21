from __future__ import annotations

import json

from pathlib import (
    Path,
)

from typing import (
    Any,
)


def _string(
    value: Any,
) -> str | None:

    if not isinstance(
        value,
        str,
    ):

        return None

    normalized = (
        value.strip()
    )

    if not normalized:

        return None

    return normalized


def find_approval_lineage(
    *,
    path: Path,
    approval_id: str,
) -> dict[
    str,
    Any,
] | None:
    """
    Resolve one durable approval identifier back to the exact
    model trajectory step that originally requested approval.

    This is evidence lookup only.

    It does NOT:
        authorize execution
        approve anything
        alter the approval state machine
        execute providers
    """

    normalized_approval_id = (
        _string(
            approval_id
        )
    )

    if normalized_approval_id is None:

        return None

    trajectory_path = (
        path
        .expanduser()
        .resolve()
    )

    if not (
        trajectory_path.exists()
        and trajectory_path.is_file()
    ):

        return None

    match: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None

    with trajectory_path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for raw_line in handle:

            line = (
                raw_line.strip()
            )

            if not line:

                continue

            try:

                payload = (
                    json.loads(
                        line
                    )
                )

            except (
                json.JSONDecodeError,
                UnicodeDecodeError,
            ):

                # Learning lookup must never affect authoritative
                # approval execution.
                continue

            if not isinstance(
                payload,
                dict,
            ):

                continue

            trajectory_id = (
                _string(
                    payload.get(
                        "trajectory_id"
                    )
                )
            )

            job_id = (
                _string(
                    payload.get(
                        "job_id"
                    )
                )
            )

            user_request = (
                _string(
                    payload.get(
                        "user_request"
                    )
                )
            )

            if (
                trajectory_id is None
                or job_id is None
                or user_request is None
            ):

                continue

            steps = (
                payload.get(
                    "steps"
                )
            )

            if not isinstance(
                steps,
                list,
            ):

                continue

            for step in steps:

                if not isinstance(
                    step,
                    dict,
                ):

                    continue

                if (
                    _string(
                        step.get(
                            "approval_id"
                        )
                    )
                    != normalized_approval_id
                ):

                    continue

                task_id = (
                    _string(
                        step.get(
                            "task_id"
                        )
                    )
                )

                if task_id is None:

                    # New approval evidence requires exact task
                    # lineage. Do not create orphan context.
                    continue

                proposed_arguments = (
                    step.get(
                        "proposed_arguments"
                    )
                )

                if not isinstance(
                    proposed_arguments,
                    dict,
                ):

                    proposed_arguments = {}

                semantic_intent = (
                    step.get(
                        "semantic_intent"
                    )
                )

                if not isinstance(
                    semantic_intent,
                    dict,
                ):

                    semantic_intent = {}

                match = {
                    "approval_id":
                        normalized_approval_id,

                    "trajectory_id":
                        trajectory_id,

                    "job_id":
                        job_id,

                    "task_id":
                        task_id,

                    "user_request":
                        user_request,

                    "task_instructions":
                        _string(
                            step.get(
                                "task_instructions"
                            )
                        ),

                    "agent":
                        _string(
                            step.get(
                                "agent"
                            )
                        ),

                    "proposed_tool":
                        _string(
                            step.get(
                                "proposed_tool"
                            )
                        ),

                    "proposed_arguments":
                        dict(
                            proposed_arguments
                        ),

                    "semantic_intent":
                        dict(
                            semantic_intent
                        ),
                }

    return match
