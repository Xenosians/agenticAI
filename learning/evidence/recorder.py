from __future__ import annotations

import json
import os
import threading
import uuid

from dataclasses import (
    asdict,
)

from datetime import (
    datetime,
    timezone,
)

from pathlib import (
    Path,
)

from typing import (
    TYPE_CHECKING,
)

from subagents.core.definitions.types import (
    HubResult,
)

from learning.evidence.quality import (
    derive_trajectory_quality,
)

from learning.evidence.rewards import (
    derive_execution_reward,
)

from learning.evidence.runtime_decisions import (
    extract_hub_runtime_decisions,
)

from learning.evidence.sanitizer import (
    sanitize_value,
)

from learning.evidence.types import (
    LearningTrajectory,
    TrajectorySignals,
    TrajectoryStep,
)


if TYPE_CHECKING:

    from learning.integrations.runtime_hooks import (
        ContinualLearningRuntimeHooks,
    )


class TrajectoryRecorder:
    """
    Durable append-only raw trajectory recorder.

    Raw trajectories are sanitized before disk persistence.

    Recorder failures must never change the authoritative outcome
    of the user's actual job.

    Specialist execution provenance is captured as immutable
    evidence when supplied by the runtime.

    Hub SemanticIntent is retained as descriptive learning evidence.

    Exact live SemanticGuard and ToolGateway decision metadata is
    retrieved from private task-indexed evidence attached to the
    HubResult by the orchestrator.

    That private evidence is intentionally absent from normal
    dataclass serialization and therefore does not become part of
    the public execution result.

    Context capture remains best-effort and happens only after the
    durable trajectory has been written.
    """

    def __init__(
        self,
        *,
        path: Path,
        enabled: bool,
        hub_model: str,
        context_hooks: (
            "ContinualLearningRuntimeHooks"
            | None
        ) = None,
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

        self.context_hooks = (
            context_hooks
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

        # ========================================================
        # PRIVATE LIVE DECISION EVIDENCE
        # ========================================================

        runtime_decisions_by_task_id = (
            extract_hub_runtime_decisions(
                result
            )
        )

        steps: list[
            TrajectoryStep
        ] = []

        for item in (
            result.results
        ):

            runtime_decisions = (
                runtime_decisions_by_task_id
                .get(
                    item.task_id,
                    {},
                )
            )

            if not isinstance(
                runtime_decisions,
                dict,
            ):

                runtime_decisions = {}

            semantic_guard_decision = (
                runtime_decisions.get(
                    "semantic_guard_decision"
                )
            )

            if not isinstance(
                semantic_guard_decision,
                dict,
            ):

                semantic_guard_decision = (
                    None
                )

            gateway_decision = (
                runtime_decisions.get(
                    "gateway_decision"
                )
            )

            if not isinstance(
                gateway_decision,
                dict,
            ):

                gateway_decision = (
                    None
                )

            steps.append(
                TrajectoryStep(
                    task_id=(
                        item.task_id
                    ),

                    task_instructions=(
                        item.task_instructions
                    ),

                    semantic_intent=(
                        asdict(
                            item.semantic_intent
                        )
                        if item.semantic_intent
                        is not None
                        else None
                    ),

                    semantic_guard_decision=(
                        semantic_guard_decision
                    ),

                    gateway_decision=(
                        gateway_decision
                    ),

                    execution_provenance=(
                        item.execution_provenance
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

                    raw_model_output=(
                        item.raw_model_output
                    ),

                    proposed_tool_calls=(
                        item.proposed_tool_calls
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
            )

        # ========================================================
        # TRAJECTORY SIGNALS
        # ========================================================

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
                    uuid.uuid4().hex
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

                dataset_eligible=False,
            )
        )

    def _record_context(
        self,
        payload: dict,
    ) -> None:
        """
        Best-effort trajectory -> contextual-evidence bridge.

        The durable trajectory already exists before this method
        executes.
        """

        if (
            self.context_hooks
            is None
        ):

            return

        try:

            records = (
                self.context_hooks
                .record_trajectory_context(
                    payload
                )
            )

            print(
                "[LEARNING] Contextualized trajectory "
                "trajectory_id="
                f"{payload.get('trajectory_id')} "
                f"context_records={len(records)}"
            )

        except Exception as exc:

            print(
                "[LEARNING] Context capture failed "
                "trajectory_id="
                f"{payload.get('trajectory_id')} "
                f"error={exc!r}"
            )

    def record(
        self,
        *,
        job_id: str,
        attempt: int,
        result: HubResult,
    ) -> dict | None:

        if not self.enabled:

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
            trajectory.model_dump(
                mode="json",
                by_alias=True,
            )
        )

        payload = (
            sanitize_value(
                raw_payload
            )
        )

        if not isinstance(
            payload,
            dict,
        ):

            raise ValueError(
                "Sanitized trajectory must "
                "remain a JSON object."
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

        # ========================================================
        # DURABLE TRAJECTORY FIRST
        # ========================================================

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

        # ========================================================
        # CONTEXT SECOND
        # ========================================================

        self._record_context(
            payload
        )

        return payload