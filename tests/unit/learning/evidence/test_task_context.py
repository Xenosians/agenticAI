from pathlib import (
    Path,
)

from learning.datasets.dataset_loader import (
    PreferenceDatasetLoader,
)

from learning.datasets.records import (
    PreferenceDatasetBuilder,
)

from learning.curation.preferences import (
    build_preference_example,
)

from learning.evidence.recorder import (
    TrajectoryRecorder,
)

from learning.evidence.types import (
    CorrectionEvent,
    CorrectionValue,
)

from subagents.core.types import (
    AgentResult,
    HubResult,
)


TASK_CONTEXT = (
    "Inspect the frontend repository working tree state."
)


def hub_result(
) -> HubResult:

    return (
        HubResult(
            status=(
                "success"
            ),

            user_request=(
                "What is the state of my frontend workspace?"
            ),

            routes=[
                "developer-specialist"
            ],

            results=[
                AgentResult(
                    task_id=(
                        "task-1"
                    ),

                    agent_name=(
                        "developer-specialist"
                    ),

                    status=(
                        "success"
                    ),

                    task_instructions=(
                        TASK_CONTEXT
                    ),

                    outcome_code=(
                        "success"
                    ),

                    proposed_tool=(
                        "workspace_git_status"
                    ),

                    proposed_arguments={
                        "repository":
                            "ai"
                    },

                    tool_result={
                        "ok":
                            True
                    },

                    answer=(
                        "Repository status retrieved."
                    ),
                )
            ],

            answer=(
                "Repository status retrieved."
            ),
        )
    )


def build_trajectory(
):

    recorder = (
        TrajectoryRecorder(
            path=Path(
                "/tmp/unused-learning-context.jsonl"
            ),

            enabled=True,

            hub_model=(
                "hub-main"
            ),
        )
    )

    return (
        recorder.build(
            job_id=(
                "job-1"
            ),

            attempt=1,

            result=(
                hub_result()
            ),
        )
    )


def build_correction(
    trajectory_id: str,
) -> CorrectionEvent:

    return (
        CorrectionEvent(
            correction_id=(
                "correction-1"
            ),

            observed_at=(
                "2026-09-15T00:00:00+00:00"
            ),

            trajectory_id=(
                trajectory_id
            ),

            task_id=(
                "task-1"
            ),

            correction_type=(
                "repository_scope"
            ),

            source=(
                "trusted_review"
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


def test_recorder_preserves_task_instructions(
):

    trajectory = (
        build_trajectory()
    )

    assert (
        trajectory.steps[
            0
        ].task_instructions
        == TASK_CONTEXT
    )


def test_preference_example_preserves_task_instructions(
):

    trajectory = (
        build_trajectory()
    )

    example = (
        build_preference_example(
            trajectory=(
                trajectory
            ),

            correction=(
                build_correction(
                    trajectory.trajectory_id
                )
            ),
        )
    )

    assert (
        example.task_instructions
        == TASK_CONTEXT
    )

    assert (
        example.task_id
        == "task-1"
    )


def test_promoted_dataset_preserves_task_instructions(
    tmp_path: Path,
):

    trajectory = (
        build_trajectory()
    )

    example = (
        build_preference_example(
            trajectory=(
                trajectory
            ),

            correction=(
                build_correction(
                    trajectory.trajectory_id
                )
            ),
        )
    )

    dataset_root = (
        tmp_path
        / "datasets"
    )

    builder = (
        PreferenceDatasetBuilder(
            root=(
                dataset_root
            ),

            eval_paths=[],
        )
    )

    manifest = (
        builder.promote(
            examples=[
                example
            ],

            promoted_by=(
                "trusted_review"
            ),

            promotion_reason=(
                "Task-context propagation test."
            ),
        )
    )

    (
        loaded_manifest,
        records,
    ) = (
        PreferenceDatasetLoader(
            root=(
                dataset_root
            )
        )
        .load(
            manifest.version
        )
    )

    assert (
        loaded_manifest.version
        == manifest.version
    )

    assert (
        len(
            records
        )
        == 1
    )

    assert (
        records[
            0
        ].task_instructions
        == TASK_CONTEXT
    )


def test_old_trajectory_step_without_task_context_remains_valid(
):

    from learning.evidence.types import (
        TrajectoryStep,
    )

    step = (
        TrajectoryStep.model_validate(
            {
                "task_id":
                    "legacy-task",

                "agent":
                    "developer-specialist",

                "status":
                    "success",

                "outcome_code":
                    "success",
            }
        )
    )

    assert (
        step.task_instructions
        is None
    )