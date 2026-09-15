from __future__ import annotations

import uuid

from datetime import (
    datetime,
    timezone,
)

from typing import (
    Any,
)

from learning.types import (
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

        for step
        in trajectory.steps

        if (
            step.proposed_tool
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


def build_preference_example(
    *,
    trajectory: LearningTrajectory,
    correction: CorrectionEvent,
) -> PreferenceExample:
    """
    Convert one immutable trajectory + correction into a canonical
    chosen/rejected preference example.

    The original specialist task context is preserved from the
    exact trajectory step targeted by the correction.

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
        # TOOL CORRECTION
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
        # ARGUMENT CORRECTION
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
    # task_instructions comes from the resolved immutable step,
    # not from the correction and not from regenerated context.
    # ========================================================

    task_instructions = (
        step.task_instructions
        if step is not None
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