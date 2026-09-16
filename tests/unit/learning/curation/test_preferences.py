import pytest

from learning.curation.preferences import (
    build_preference_example,
)

from learning.evidence.quality import (
    apply_correction_to_quality,
)

from learning.evidence.types import (
    CorrectionEvent,
    CorrectionValue,
    ExecutionReward,
    LearningTrajectory,
    TrajectoryQuality,
    TrajectorySignals,
    TrajectoryStep,
)


def example_trajectory(
) -> LearningTrajectory:

    return (
        LearningTrajectory(
            trajectory_id=(
                "trajectory-123"
            ),

            observed_at=(
                "2026-09-15T00:00:00+00:00"
            ),

            job_id=(
                "job-123"
            ),

            attempt=1,

            user_request=(
                "Show me the frontend git status."
            ),

            hub_model=(
                "hub-main"
            ),

            hub_status=(
                "partial_error"
            ),

            routes=[
                "developer-specialist",
            ],

            steps=[
                TrajectoryStep(
                    task_id=(
                        "task-123"
                    ),

                    agent=(
                        "developer-specialist"
                    ),

                    status="error",

                    outcome_code=(
                        "grounding_failed"
                    ),

                    proposed_tool=(
                        "workspace_git_status"
                    ),

                    proposed_arguments={
                        "repository":
                            "ai",
                    },

                    error=(
                        "Grounding failed."
                    ),
                )
            ],

            final_answer=(
                "The operation could not "
                "be completed."
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

            quality=(
                TrajectoryQuality(
                    grounding_valid=False,
                    failure_types=[
                        "grounding_failure",
                    ],
                )
            ),

            dataset_eligible=False,
        )
    )


def example_correction(
) -> CorrectionEvent:

    return (
        CorrectionEvent(
            correction_id=(
                "correction-123"
            ),

            observed_at=(
                "2026-09-15T00:01:00+00:00"
            ),

            trajectory_id=(
                "trajectory-123"
            ),

            task_id=(
                "task-123"
            ),

            correction_type=(
                "repository_scope"
            ),

            source=(
                "explicit_user"
            ),

            values=[
                CorrectionValue(
                    field=(
                        "repository"
                    ),

                    rejected_value=(
                        "ai"
                    ),

                    chosen_value=(
                        "frontend"
                    ),
                )
            ],

            dataset_eligible=False,
        )
    )


def test_correction_updates_quality_without_mutating_original():
    trajectory = (
        example_trajectory()
    )

    correction = (
        example_correction()
    )

    original = (
        trajectory
        .quality
        .model_copy(
            deep=True
        )
    )

    updated = (
        apply_correction_to_quality(
            trajectory.quality,
            correction,
        )
    )

    assert (
        trajectory.quality
        == original
    )

    assert (
        updated.user_corrected
        is True
    )

    assert (
        updated.review_state
        == "corrected"
    )

    assert (
        updated.arguments_correct
        is False
    )

    assert (
        updated.route_correct
        is None
    )

    assert (
        updated.tool_correct
        is None
    )

    assert (
        updated.correction_type
        == "repository_scope"
    )

    assert (
        "argument_error"
        in updated.failure_types
    )

    assert (
        updated.quality_eligible
        is False
    )


def test_build_repository_preference_example():
    example = (
        build_preference_example(
            trajectory=(
                example_trajectory()
            ),

            correction=(
                example_correction()
            ),
        )
    )

    assert (
        example.schema_name
        == "preference-example.v1"
    )

    assert (
        example.user_request
        == (
            "Show me the frontend "
            "git status."
        )
    )

    assert (
        example.rejected.tool
        == "workspace_git_status"
    )

    assert (
        example.chosen.tool
        == "workspace_git_status"
    )

    assert (
        example.rejected.arguments
        == {
            "repository":
                "ai",
        }
    )

    assert (
        example.chosen.arguments
        == {
            "repository":
                "frontend",
        }
    )

    assert (
        example.dataset_eligible
        is False
    )


def test_preference_builder_rejects_mismatched_original_value():
    correction = (
        example_correction()
    )

    correction.values[
        0
    ].rejected_value = (
        "backend"
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
                example_trajectory()
            ),

            correction=(
                correction
            ),
        )


def test_preference_builder_requires_task_id_when_ambiguous():
    trajectory = (
        example_trajectory()
    )

    trajectory.steps.append(
        TrajectoryStep(
            task_id=(
                "task-456"
            ),

            agent=(
                "ticket-specialist"
            ),

            status="success",

            outcome_code="success",

            proposed_tool=(
                "ticket_get"
            ),

            proposed_arguments={
                "issue_key":
                    "KAN-1",
            },
        )
    )

    correction = (
        example_correction()
    )

    correction.task_id = None

    with pytest.raises(
        ValueError,
        match="ambiguous",
    ):
        build_preference_example(
            trajectory=(
                trajectory
            ),

            correction=(
                correction
            ),
        )