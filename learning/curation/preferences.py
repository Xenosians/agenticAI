from __future__ import annotations

import uuid

from datetime import (
    datetime,
    timezone,
)

from typing import (
    Any,
)

from learning.evidence.types import (
    CorrectionEvent,
    LearningTrajectory,
    PreferenceExample,
    PreferenceOption,
    TrajectoryStep,
)


AGENT_FIELDS = {
    "agent",
    "agent_name",
    "route",
}


TOOL_FIELDS = {
    "tool",
    "tool_name",
}


TOOL_CALL_FIELDS = {
    "tool_calls",
    "proposed_tool_calls",
}


ANSWER_FIELDS = {
    "answer",
    "final_answer",
}


def _find_step(
    trajectory: LearningTrajectory,
    correction: CorrectionEvent,
) -> (
    TrajectoryStep
    | None
):
    """
    Resolve the specialist step targeted by a correction.

    task_id is authoritative when supplied.

    For older/simpler single-step trajectories, the only
    specialist/tool step remains unambiguous.
    """

    if correction.task_id is not None:

        for step in trajectory.steps:

            if (
                step.task_id
                == correction.task_id
            ):

                return step

        raise ValueError(
            "Correction task_id does not exist "
            "in the referenced trajectory."
        )

    tool_steps = [
        step

        for step in trajectory.steps

        if (
            step.proposed_tool
            is not None
            or step.proposed_tool_calls
            is not None
        )
    ]

    if len(
        tool_steps
    ) == 1:

        return (
            tool_steps[
                0
            ]
        )

    if len(
        trajectory.steps
    ) == 1:

        return (
            trajectory.steps[
                0
            ]
        )

    if not trajectory.steps:

        return None

    raise ValueError(
        "Correction is ambiguous because the trajectory "
        "contains multiple specialist steps. "
        "Provide task_id."
    )


def _assert_original_value(
    *,
    field: str,
    observed: Any,
    expected_rejected: Any,
) -> None:
    """
    Refuse to create preference evidence when the correction does
    not describe what was actually observed in the immutable raw
    trajectory.
    """

    if (
        expected_rejected
        is None
    ):

        return

    if (
        observed
        != expected_rejected
    ):

        raise ValueError(
            "Correction rejected_value does not match "
            "the original trajectory for field "
            f"'{field}'. "
            f"Observed={observed!r}, "
            f"correction={expected_rejected!r}."
        )


def _normalize_tool_calls(
    value: Any,
    *,
    field: str,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """
    Validate and copy a complete structured worker tool-call set.

    The accepted structure intentionally matches parse_tool_calls:

        [
            {
                "name": "...",
                "arguments": {...},
            }
        ]

    A preference call set is evidence only. Validation here does
    not authorize any capability or execute anything.
    """

    if not isinstance(
        value,
        list,
    ):

        raise ValueError(
            f"{field} must be a list of tool calls."
        )

    if not value:

        raise ValueError(
            f"{field} must contain at least one tool call."
        )

    normalized: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for index, call in enumerate(
        value
    ):

        if not isinstance(
            call,
            dict,
        ):

            raise ValueError(
                f"{field}[{index}] must be an object."
            )

        name = (
            call.get(
                "name"
            )
        )

        arguments = (
            call.get(
                "arguments"
            )
        )

        if (
            not isinstance(
                name,
                str,
            )
            or not name.strip()
        ):

            raise ValueError(
                f"{field}[{index}] has an invalid tool name."
            )

        if not isinstance(
            arguments,
            dict,
        ):

            raise ValueError(
                f"{field}[{index}] has invalid arguments."
            )

        normalized.append(
            {
                "name":
                    name,

                "arguments":
                    dict(
                        arguments
                    ),
            }
        )

    return normalized


def _copy_observed_tool_calls(
    step: (
        TrajectoryStep
        | None
    ),
) -> (
    list[
        dict[
            str,
            Any,
        ]
    ]
    | None
):

    if (
        step is None
        or step.proposed_tool_calls
        is None
    ):

        return None

    return (
        _normalize_tool_calls(
            step.proposed_tool_calls,
            field=(
                "proposed_tool_calls"
            ),
        )
    )


def build_preference_example(
    *,
    trajectory: LearningTrajectory,
    correction: CorrectionEvent,
) -> PreferenceExample:
    """
    Convert one immutable trajectory + correction into a canonical
    chosen/rejected preference example.

    The original specialist task context and execution provenance
    are preserved from the exact trajectory step targeted by the
    correction.

    Singular tool behavior continues to use tool / arguments.

    Abnormal complete worker call sets use tool_calls so rejected
    behavior is never silently flattened into tool=None.

    Creating this derived example does not make it
    training-eligible. Promotion remains a separate trusted step.
    """

    if (
        trajectory.trajectory_id
        != correction.trajectory_id
    ):

        raise ValueError(
            "Correction trajectory_id does not match "
            "the trajectory."
        )

    step = (
        _find_step(
            trajectory,
            correction,
        )
    )

    rejected = (
        PreferenceOption(
            agent=(
                step.agent
                if step is not None
                else (
                    trajectory.routes[
                        0
                    ]
                    if len(
                        trajectory.routes
                    ) == 1
                    else None
                )
            ),

            tool_calls=(
                _copy_observed_tool_calls(
                    step
                )
            ),

            tool=(
                step.proposed_tool
                if step is not None
                else None
            ),

            arguments=(
                dict(
                    step.proposed_arguments
                )
                if (
                    step is not None
                    and step.proposed_arguments
                    is not None
                )
                else None
            ),

            answer=(
                trajectory.final_answer
            ),
        )
    )

    chosen = (
        rejected.model_copy(
            deep=True
        )
    )

    for value in correction.values:

        field = (
            value.field
            .strip()
        )

        # ====================================================
        # AGENT / ROUTE CORRECTION
        # ====================================================

        if field in AGENT_FIELDS:

            _assert_original_value(
                field=(
                    field
                ),

                observed=(
                    rejected.agent
                ),

                expected_rejected=(
                    value.rejected_value
                ),
            )

            chosen.agent = (
                value.chosen_value
            )

            continue

        # ====================================================
        # COMPLETE TOOL-CALL-SET CORRECTION
        #
        # This is deliberately distinct from singular tool and
        # argument corrections.
        #
        # The rejected value MUST be supplied and MUST exactly
        # match the immutable observed call set.
        # ====================================================

        if field in TOOL_CALL_FIELDS:

            if step is None:

                raise ValueError(
                    "Tool-call-set correction requires "
                    "a specialist trajectory step."
                )

            if (
                rejected.tool_calls
                is None
            ):

                raise ValueError(
                    "Tool-call-set correction requires "
                    "captured proposed_tool_calls evidence."
                )

            if (
                value.rejected_value
                is None
            ):

                raise ValueError(
                    "Tool-call-set correction must include "
                    "the exact rejected_value."
                )

            normalized_rejected = (
                _normalize_tool_calls(
                    value.rejected_value,
                    field=(
                        "rejected_value"
                    ),
                )
            )

            _assert_original_value(
                field=(
                    field
                ),

                observed=(
                    rejected.tool_calls
                ),

                expected_rejected=(
                    normalized_rejected
                ),
            )

            chosen.tool_calls = (
                _normalize_tool_calls(
                    value.chosen_value,
                    field=(
                        "chosen_value"
                    ),
                )
            )

            continue

        # ====================================================
        # SINGULAR TOOL CORRECTION
        # ====================================================

        if field in TOOL_FIELDS:

            _assert_original_value(
                field=(
                    field
                ),

                observed=(
                    rejected.tool
                ),

                expected_rejected=(
                    value.rejected_value
                ),
            )

            chosen.tool = (
                value.chosen_value
            )

            continue

        # ====================================================
        # ANSWER CORRECTION
        # ====================================================

        if field in ANSWER_FIELDS:

            _assert_original_value(
                field=(
                    field
                ),

                observed=(
                    rejected.answer
                ),

                expected_rejected=(
                    value.rejected_value
                ),
            )

            chosen.answer = (
                value.chosen_value
            )

            continue

        # ====================================================
        # SINGULAR ARGUMENT CORRECTION
        # ====================================================

        if step is None:

            raise ValueError(
                "Argument correction requires "
                "a specialist trajectory step."
            )

        rejected_arguments = (
            rejected.arguments
            or {}
        )

        observed = (
            rejected_arguments.get(
                field
            )
        )

        _assert_original_value(
            field=(
                field
            ),

            observed=(
                observed
            ),

            expected_rejected=(
                value.rejected_value
            ),
        )

        chosen_arguments = (
            dict(
                chosen.arguments
                or {}
            )
        )

        chosen_arguments[
            field
        ] = (
            value.chosen_value
        )

        chosen.arguments = (
            chosen_arguments
        )

    if (
        rejected.model_dump()
        == chosen.model_dump()
    ):

        raise ValueError(
            "Correction does not change the "
            "preference example."
        )

    # ========================================================
    # CRITICAL MODEL-INPUT LINEAGE
    #
    # Both task context and execution provenance come from the
    # resolved immutable trajectory step.
    #
    # Neither value comes from the correction and neither value is
    # regenerated from current runtime configuration.
    # ========================================================

    task_instructions = (
        step.task_instructions
        if step is not None
        else None
    )

    execution_provenance = (
        step
        .execution_provenance
        .model_copy(
            deep=True
        )

        if (
            step is not None
            and step.execution_provenance
            is not None
        )

        else None
    )

    task_id = (
        correction.task_id
        or (
            step.task_id
            if step is not None
            else None
        )
    )

    return (
        PreferenceExample(
            example_id=(
                uuid.uuid4().hex
            ),

            created_at=(
                datetime
                .now(
                    timezone.utc
                )
                .isoformat()
            ),

            trajectory_id=(
                trajectory.trajectory_id
            ),

            correction_id=(
                correction.correction_id
            ),

            task_id=(
                task_id
            ),

            task_instructions=(
                task_instructions
            ),

            execution_provenance=(
                execution_provenance
            ),

            source=(
                correction.source
            ),

            correction_type=(
                correction.correction_type
            ),

            user_request=(
                trajectory.user_request
            ),

            rejected=(
                rejected
            ),

            chosen=(
                chosen
            ),

            dataset_eligible=False,
        )
    )
