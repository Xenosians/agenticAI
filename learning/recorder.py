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

from subagents.core.types import (
    HubResult,
)

from learning.quality import (
    derive_trajectory_quality,
)

from learning.rewards import (
    derive_execution_reward,
)

from learning.sanitizer import (
    sanitize_value,
)

from learning.types import (
    LearningTrajectory,
    TrajectorySignals,
    TrajectoryStep,
)


class TrajectoryRecorder:
    """
    Durable append-only raw trajectory recorder.

    Raw trajectories are sanitized before disk persistence.

    Recorder failures must never change the authoritative outcome
    of the user's actual job.
    """

    def __init__(
        self,
        *,
        path: Path,
        enabled: bool,
        hub_model: str,
    ) -> None:

        self.path = (
            path
            .expanduser()
            .resolve()
        )

        self.enabled = (
            enabled
        )

        self.hub_model = (
            hub_model
        )

        self._lock = (
            threading.Lock()
        )

    def build(
        self,
        *,
        job_id: str,
        attempt: int,
        result: HubResult,
    ) -> LearningTrajectory:

        steps = [
            TrajectoryStep(
                task_id=(
                    item.task_id
                ),

                agent=(
                    item.agent_name
                ),

                status=(
                    item.status
                ),

                outcome_code=(
                    item.outcome_code
                ),

                proposed_tool=(
                    item.proposed_tool
                ),

                proposed_arguments=(
                    item.proposed_arguments
                ),

                tool_result=(
                    item.tool_result
                ),

                approval_id=(
                    item.approval_id
                ),

                error=(
                    item.error
                ),
            )

            for item
            in result.results
        ]

        specialist_success_count = sum(
            1

            for item
            in result.results

            if (
                item.status
                == "success"
            )
        )

        specialist_error_count = sum(
            1

            for item
            in result.results

            if (
                item.status
                == "error"
            )
        )

        approval_required_count = sum(
            1

            for item
            in result.results

            if (
                item.status
                == "approval_required"
            )
        )

        tool_proposed_count = sum(
            1

            for item
            in result.results

            if (
                item.proposed_tool
                is not None
            )
        )

        tool_success_count = sum(
            1

            for item
            in result.results

            if (
                item.status
                == "success"
                and item.proposed_tool
                is not None
                and isinstance(
                    item.tool_result,
                    dict,
                )
            )
        )

        signals = (
            TrajectorySignals(
                delegated=(
                    bool(
                        result.routes
                    )
                ),

                route_count=(
                    len(
                        result.routes
                    )
                ),

                specialist_count=(
                    len(
                        result.results
                    )
                ),

                specialist_success_count=(
                    specialist_success_count
                ),

                specialist_error_count=(
                    specialist_error_count
                ),

                approval_required_count=(
                    approval_required_count
                ),

                tool_proposed_count=(
                    tool_proposed_count
                ),

                tool_success_count=(
                    tool_success_count
                ),

                overall_success=(
                    result.status
                    == "success"
                ),

                had_error=(
                    specialist_error_count
                    > 0
                    or result.status
                    == "partial_error"
                ),

                waiting_approval=(
                    approval_required_count
                    > 0
                    or result.status
                    == "approval_required"
                ),
            )
        )

        quality = (
            derive_trajectory_quality(
                steps
            )
        )

        return (
            LearningTrajectory(
                trajectory_id=(
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

                job_id=(
                    job_id
                ),

                attempt=(
                    attempt
                ),

                user_request=(
                    result.user_request
                ),

                hub_model=(
                    self.hub_model
                ),

                hub_status=(
                    result.status
                ),

                routes=(
                    result.routes
                ),

                steps=(
                    steps
                ),

                final_answer=(
                    result.answer
                ),

                signals=(
                    signals
                ),

                execution_reward=(
                    derive_execution_reward(
                        result
                    )
                ),

                quality=(
                    quality
                ),

                # Still false.
                #
                # Runtime success alone must never promote a
                # trajectory into training data.
                dataset_eligible=False,
            )
        )

    def record(
        self,
        *,
        job_id: str,
        attempt: int,
        result: HubResult,
    ) -> dict | None:

        if not (
            self.enabled
        ):
            return None

        trajectory = (
            self.build(
                job_id=(
                    job_id
                ),

                attempt=(
                    attempt
                ),

                result=(
                    result
                ),
            )
        )

        raw_payload = (
            trajectory
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

        return (
            payload
        )