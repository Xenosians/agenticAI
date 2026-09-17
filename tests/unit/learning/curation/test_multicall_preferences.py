import pytest

from learning.curation.preferences import (
    build_preference_example,
)

from learning.evidence.types import (
    CorrectionEvent,
    CorrectionValue,
    ExecutionReward,
    LearningTrajectory,
    TrajectorySignals,
    TrajectoryStep,
)


REJECTED_CALLS = [
    {
        "name":
            "reset_password",

        "arguments": {
            "user_id":
                "jdoe",
        },
    },

    {
        "name":
            "reset_password",

        "arguments": {
            "user_id":
                "alice",
        },
    },
]


CHOSEN_CALLS = [
    {
        "name":
            "reset_password",

        "arguments": {
            "user_id":
                "jdoe",
        },
    },
]


def multi_call_trajectory(
) -> LearningTrajectory:

    return (
        LearningTrajectory(
            trajectory_id=(
                "trajectory-multi"
            ),

            observed_at=(
                "2026-09-17T00:00:00+00:00"
            ),

            job_id=(
                "job-multi"
            ),

            attempt=1,

            user_request=(
                "Reset jdoe's password, not alice's."
            ),

            hub_model=(
                "hub-main"
            ),

            hub_status=(
                "partial_error"
            ),

            routes=[
                "account-specialist",
            ],

            steps=[
                TrajectoryStep(
                    task_id=(
                        "task-multi"
                    ),

                    task_instructions=(
                        "Reset the password for jdoe only."
                    ),

                    agent=(
                        "account-specialist"
                    ),

                    status=(
                        "error"
                    ),

                    outcome_code=(
                        "invalid_tool_call_count"
                    ),

                    raw_model_output=(
                        """
                        [
                            {
                                "name": "reset_password",
                                "arguments": {
                                    "user_id": "jdoe"
                                }
                            },
                            {
                                "name": "reset_password",
                                "arguments": {
                                    "user_id": "alice"
                                }
                            }
                        ]
                        """
                    ),

                    proposed_tool_calls=(
                        REJECTED_CALLS
                    ),

                    proposed_tool=None,

                    proposed_arguments=None,

                    error=(
                        "Worker must return exactly "
                        "one tool call."
                    ),
                )
            ],

            final_answer=(
                "Worker must return exactly "
                "one tool call."
            ),

            signals=(
                TrajectorySignals(
                    delegated=True,
                    route_count=1,
                    specialist_count=1,
                    specialist_error_count=1,
                    had_error=True,
                )
            ),

            execution_reward=(
                ExecutionReward(
                    total=-1.0,
                    quality_eligible=False,
                )
            ),

            dataset_eligible=False,
        )
    )


def multi_call_correction(
) -> CorrectionEvent:

    return (
        CorrectionEvent(
            correction_id=(
                "correction-multi"
            ),

            observed_at=(
                "2026-09-17T00:01:00+00:00"
            ),

            trajectory_id=(
                "trajectory-multi"
            ),

            task_id=(
                "task-multi"
            ),

            correction_type=(
                "tool_selection"
            ),

            source=(
                "trusted_review"
            ),

            values=[
                CorrectionValue(
                    field=(
                        "tool_calls"
                    ),

                    rejected_value=(
                        REJECTED_CALLS
                    ),

                    chosen_value=(
                        CHOSEN_CALLS
                    ),
                )
            ],

            dataset_eligible=False,
        )
    )


def test_multicall_preference_preserves_complete_rejected_behavior(
):

    example = (
        build_preference_example(
            trajectory=(
                multi_call_trajectory()
            ),

            correction=(
                multi_call_correction()
            ),
        )
    )

    assert (
        example.rejected.tool_calls
        == REJECTED_CALLS
    )

    assert (
        example.chosen.tool_calls
        == CHOSEN_CALLS
    )

    # The singular canonical proposal was never valid for this
    # execution, so do not manufacture one retrospectively.
    assert (
        example.rejected.tool
        is None
    )

    assert (
        example.rejected.arguments
        is None
    )

    assert (
        example.chosen.tool
        is None
    )

    assert (
        example.chosen.arguments
        is None
    )

    assert (
        example.task_id
        == "task-multi"
    )

    assert (
        example.task_instructions
        == "Reset the password for jdoe only."
    )


def test_multicall_preference_rejects_mismatched_observed_call_set(
):

    correction = (
        multi_call_correction()
    )

    correction.values[
        0
    ].rejected_value = (
        CHOSEN_CALLS
    )

    with pytest.raises(
        ValueError,
        match=(
            "does not match "
            "the original trajectory"
        ),
    ):

        build_preference_example(
            trajectory=(
                multi_call_trajectory()
            ),

            correction=(
                correction
            ),
        )


def test_multicall_preference_requires_exact_rejected_value(
):

    correction = (
        multi_call_correction()
    )

    correction.values[
        0
    ].rejected_value = None

    with pytest.raises(
        ValueError,
        match=(
            "must include "
            "the exact rejected_value"
        ),
    ):

        build_preference_example(
            trajectory=(
                multi_call_trajectory()
            ),

            correction=(
                correction
            ),
        )


def test_multicall_preference_rejects_malformed_chosen_call_set(
):

    correction = (
        multi_call_correction()
    )

    correction.values[
        0
    ].chosen_value = [
        {
            "name":
                "reset_password",

            "arguments":
                "jdoe",
        }
    ]

    with pytest.raises(
        ValueError,
        match=(
            "has invalid arguments"
        ),
    ):

        build_preference_example(
            trajectory=(
                multi_call_trajectory()
            ),

            correction=(
                correction
            ),
        )


def test_single_tool_preferences_remain_backward_compatible(
):

    trajectory = (
        LearningTrajectory(
            trajectory_id=(
                "trajectory-single"
            ),

            observed_at=(
                "2026-09-17T00:00:00+00:00"
            ),

            job_id=(
                "job-single"
            ),

            attempt=1,

            user_request=(
                "Is jdoe unlocked?"
            ),

            hub_model=(
                "hub-main"
            ),

            hub_status=(
                "partial_error"
            ),

            routes=[
                "account-specialist",
            ],

            steps=[
                TrajectoryStep(
                    task_id=(
                        "task-single"
                    ),

                    agent=(
                        "account-specialist"
                    ),

                    status=(
                        "error"
                    ),

                    outcome_code=(
                        "agent_tool_not_allowed"
                    ),

                    proposed_tool=(
                        "unlock_user"
                    ),

                    proposed_arguments={
                        "user_id":
                            "jdoe",
                    },
                )
            ],

            final_answer=(
                "Denied."
            ),

            signals=(
                TrajectorySignals(
                    delegated=True,
                    route_count=1,
                    specialist_count=1,
                    specialist_error_count=1,
                    tool_proposed_count=1,
                    had_error=True,
                )
            ),

            execution_reward=(
                ExecutionReward(
                    total=-1.0,
                    quality_eligible=False,
                )
            ),

            dataset_eligible=False,
        )
    )

    correction = (
        CorrectionEvent(
            correction_id=(
                "correction-single"
            ),

            observed_at=(
                "2026-09-17T00:01:00+00:00"
            ),

            trajectory_id=(
                "trajectory-single"
            ),

            task_id=(
                "task-single"
            ),

            correction_type=(
                "tool_selection"
            ),

            source=(
                "trusted_review"
            ),

            values=[
                CorrectionValue(
                    field=(
                        "tool"
                    ),

                    rejected_value=(
                        "unlock_user"
                    ),

                    chosen_value=(
                        "account_status"
                    ),
                )
            ],

            dataset_eligible=False,
        )
    )

    example = (
        build_preference_example(
            trajectory=(
                trajectory
            ),

            correction=(
                correction
            ),
        )
    )

    assert (
        example.rejected.tool_calls
        is None
    )

    assert (
        example.chosen.tool_calls
        is None
    )

    assert (
        example.rejected.tool
        == "unlock_user"
    )

    assert (
        example.chosen.tool
        == "account_status"
    )

    assert (
        example.rejected.arguments
        == {
            "user_id":
                "jdoe",
        }
    )

    assert (
        example.chosen.arguments
        == {
            "user_id":
                "jdoe",
        }
    )
