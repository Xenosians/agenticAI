from __future__ import annotations

import hashlib
import json

from datetime import datetime, timezone
from typing import Any

from learning.continuous.types import (
    LearningEvent,
)


def _utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def _stable_event_id(
    payload: dict[str, Any],
) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    digest = hashlib.sha256(
        serialized.encode(
            "utf-8"
        )
    ).hexdigest()

    return (
        "learning-event-"
        + digest[:24]
    )


def learning_event_from_trajectory(
    trajectory: dict[str, Any],
) -> LearningEvent:
    trajectory_id = str(
        trajectory.get(
            "trajectory_id",
            "",
        )
    ).strip()

    if not trajectory_id:
        raise ValueError(
            "Trajectory event requires trajectory_id."
        )

    steps = trajectory.get(
        "steps",
        [],
    )

    if not isinstance(
        steps,
        list,
    ):
        steps = []

    outcome_codes = []

    for step in steps:
        if not isinstance(
            step,
            dict,
        ):
            continue

        code = step.get(
            "outcome_code"
        )

        if (
            isinstance(
                code,
                str,
            )
            and code.strip()
        ):
            outcome_codes.append(
                code.strip()
            )

    quality = trajectory.get(
        "quality"
    )

    if not isinstance(
        quality,
        dict,
    ):
        quality = {}

    failure_types = (
        quality.get(
            "failure_types",
            [],
        )
    )

    if not isinstance(
        failure_types,
        list,
    ):
        failure_types = []

    signals = trajectory.get(
        "signals"
    )

    if not isinstance(
        signals,
        dict,
    ):
        signals = {}

    reward = trajectory.get(
        "execution_reward"
    )

    if isinstance(
        reward,
        dict,
    ):
        reward_total = float(
            reward.get(
                "total",
                0.0,
            )
            or 0.0
        )
    else:
        reward_total = 0.0

    identity = {
        "trajectory_id":
            trajectory_id,

        "job_id":
            trajectory.get(
                "job_id"
            ),

        "attempt":
            trajectory.get(
                "attempt"
            ),
    }

    return LearningEvent(
        event_id=_stable_event_id(
            identity
        ),
        created_at=_utc_now(),
        trajectory_id=(
            trajectory_id
        ),
        job_id=(
            trajectory.get(
                "job_id"
            )
        ),
        attempt=(
            trajectory.get(
                "attempt"
            )
        ),
        model_key=str(
            trajectory.get(
                "hub_model",
                "hub-main",
            )
        ),
        hub_status=(
            trajectory.get(
                "hub_status"
            )
        ),
        outcome_codes=list(
            dict.fromkeys(
                outcome_codes
            )
        ),
        failure_types=[
            str(
                value
            )
            for value in (
                failure_types
            )
        ],
        execution_reward=(
            reward_total
        ),
        overall_success=bool(
            signals.get(
                "overall_success",
                False,
            )
        ),
        had_error=bool(
            signals.get(
                "had_error",
                False,
            )
        ),
        waiting_approval=bool(
            signals.get(
                "waiting_approval",
                False,
            )
        ),
        payload=(
            trajectory
        ),
        # Runtime evidence remains evidence. Trusted review /
        # deterministic ground truth is what eventually promotes
        # material into training datasets.
        training_eligible=False,
    )
