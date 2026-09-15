import json

from pathlib import (
    Path,
)

import pytest

from learning.curation import (
    CuratedTrajectoryReference,
    CurationReport,
    evidence_fingerprint,
)

from learning.training_pipeline import (
    TrustedTrainingPipeline,
)

from learning.types import (
    CorrectionEvent,
    CorrectionValue,
    DatasetPromotion,
    ExecutionReward,
    LearningTrajectory,
    PreferenceDatasetRecord,
    PreferenceOption,
    TrajectorySignals,
    TrajectoryStep,
)


TASK_CONTEXT = (
    "Inspect the frontend repository working tree state."
)


def write_jsonl(
    path: Path,
    items,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for item in items:

            payload = (
                item.model_dump(
                    mode="json",
                    by_alias=True,
                )

                if hasattr(
                    item,
                    "model_dump"
                )

                else item
            )

            handle.write(
                json.dumps(
                    payload,
                    sort_keys=True,
                )
            )

            handle.write(
                "\n"
            )


def build_trajectory(
    *,
    task_instructions: (
        str
        | None
    ),
) -> LearningTrajectory:

    return (
        LearningTrajectory(
            trajectory_id=(
                "trajectory-1"
            ),

            observed_at=(
                "2026-09-15T00:00:00+00:00"
            ),

            job_id=(
                "job-1"
            ),

            attempt=1,

            user_request=(
                "Check my frontend workspace."
            ),

            hub_model=(
                "hub-main"
            ),

            hub_status=(
                "success"
            ),

            routes=[
                "developer-specialist"
            ],

            steps=[
                TrajectoryStep(
                    task_id=(
                        "task-1"
                    ),

                    task_instructions=(
                        task_instructions
                    ),

                    agent=(
                        "developer-specialist"
                    ),

                    status=(
                        "success"
                    ),

                    outcome_code=(
                        "success"
                    ),

                    proposed_tool=(
                        "workspace_git_status"
                    ),

                    proposed_arguments={
                        "repository":
                            "frontend"
                    },
                )
            ],

            final_answer=(
                "Frontend repository checked."
            ),

            signals=(
                TrajectorySignals(
                    delegated=True,
                    route_count=1,
                    specialist_count=1,
                    specialist_success_count=1,
                    specialist_error_count=0,
                    approval_required_count=0,
                    tool_proposed_count=1,
                    tool_success_count=1,
                    overall_success=True,
                    had_error=False,
                    waiting_approval=False,
                )
            ),

            execution_reward=(
                ExecutionReward(
                    components={
                        "success":
                            1.0
                    },

                    total=1.0,

                    quality_eligible=False,
                )
            ),

            quality=None,

            dataset_eligible=False,
        )
    )


def build_correction(
) -> CorrectionEvent:

    return (
        CorrectionEvent(
            correction_id=(
                "correction-1"
            ),

            observed_at=(
                "2026-09-15T00:00:01+00:00"
            ),

            trajectory_id=(
                "trajectory-1"
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
                        "frontend"
                    ),

                    chosen_value=(
                        "ai"
                    ),
                )
            ],

            dataset_eligible=False,
        )
    )


def build_record(
    *,
    task_id: (
        str
        | None
    ) = "task-1",

    task_instructions: (
        str
        | None
    ) = TASK_CONTEXT,
) -> PreferenceDatasetRecord:

    return (
        PreferenceDatasetRecord(
            record_id=(
                "record-1"
            ),

            source_example_id=(
                "example-1"
            ),

            trajectory_id=(
                "trajectory-1"
            ),

            correction_id=(
                "correction-1"
            ),

            task_id=(
                task_id
            ),

            task_instructions=(
                task_instructions
            ),

            source=(
                "trusted_review"
            ),

            correction_type=(
                "repository_scope"
            ),

            user_request=(
                "Check my frontend workspace."
            ),

            rejected=(
                PreferenceOption(
                    agent=(
                        "developer-specialist"
                    ),

                    tool=(
                        "workspace_git_status"
                    ),

                    arguments={
                        "repository":
                            "frontend"
                    },
                )
            ),

            chosen=(
                PreferenceOption(
                    agent=(
                        "developer-specialist"
                    ),

                    tool=(
                        "workspace_git_status"
                    ),

                    arguments={
                        "repository":
                            "ai"
                    },
                )
            ),

            promotion=(
                DatasetPromotion(
                    promoted_at=(
                        "2026-09-15T00:00:02+00:00"
                    ),

                    promoted_by=(
                        "trusted_review"
                    ),

                    reason=(
                        "Verified test evidence."
                    ),
                )
            ),

            dataset_eligible=True,
        )
    )


def build_curation_report(
    trajectory: LearningTrajectory,
) -> CurationReport:

    return (
        CurationReport(
            curation_id=(
                "curation-test"
            ),

            generated_at=(
                "2026-09-15T00:00:03+00:00"
            ),

            candidate_trajectory_count=1,

            eligible_trajectory_count=1,

            excluded_trajectory_count=0,

            correction_count=1,

            usable_correction_count=1,

            orphan_correction_count=0,

            deduplicated_trajectory_count=0,

            held_out_contamination_count=0,

            review_count=2,

            eligible=[
                CuratedTrajectoryReference(
                    trajectory_id=(
                        trajectory.trajectory_id
                    ),

                    evidence_fingerprint=(
                        evidence_fingerprint(
                            trajectory
                        )
                    ),

                    usable_correction_ids=[
                        "correction-1"
                    ],
                )
            ],
        )
    )


def build_pipeline(
    *,
    tmp_path: Path,
    trajectory: LearningTrajectory,
) -> TrustedTrainingPipeline:

    trajectory_path = (
        tmp_path
        / "trajectories.jsonl"
    )

    correction_path = (
        tmp_path
        / "corrections.jsonl"
    )

    write_jsonl(
        trajectory_path,
        [
            trajectory
        ],
    )

    write_jsonl(
        correction_path,
        [
            build_correction()
        ],
    )

    return (
        TrustedTrainingPipeline(
            trajectory_path=(
                trajectory_path
            ),

            correction_path=(
                correction_path
            ),

            review_path=(
                tmp_path
                / "reviews.jsonl"
            ),

            dataset_root=(
                tmp_path
                / "datasets"
            ),

            output_root=(
                tmp_path
                / "exports"
            ),
        )
    )


def test_evidence_fingerprint_includes_task_context(
):

    first = (
        build_trajectory(
            task_instructions=(
                "Inspect frontend status."
            )
        )
    )

    second = (
        build_trajectory(
            task_instructions=(
                "Inspect frontend branches."
            )
        )
    )

    assert (
        evidence_fingerprint(
            first
        )
        != evidence_fingerprint(
            second
        )
    )


def test_evidence_fingerprint_normalizes_task_context_formatting(
):

    first = (
        build_trajectory(
            task_instructions=(
                "Inspect FRONTEND repository status."
            )
        )
    )

    second = (
        build_trajectory(
            task_instructions=(
                "  inspect frontend\n"
                "repository status.  "
            )
        )
    )

    assert (
        evidence_fingerprint(
            first
        )
        == evidence_fingerprint(
            second
        )
    )


def test_dataset_context_lineage_accepts_exact_source_context(
    tmp_path: Path,
):

    trajectory = (
        build_trajectory(
            task_instructions=(
                TASK_CONTEXT
            )
        )
    )

    pipeline = (
        build_pipeline(
            tmp_path=(
                tmp_path
            ),

            trajectory=(
                trajectory
            ),
        )
    )

    pipeline._assert_dataset_lineage(
        records=[
            build_record()
        ],

        curation_report=(
            build_curation_report(
                trajectory
            )
        ),
    )


def test_dataset_context_lineage_rejects_changed_context(
    tmp_path: Path,
):

    trajectory = (
        build_trajectory(
            task_instructions=(
                TASK_CONTEXT
            )
        )
    )

    pipeline = (
        build_pipeline(
            tmp_path=(
                tmp_path
            ),

            trajectory=(
                trajectory
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "task_instructions does not match"
        ),
    ):

        pipeline._assert_dataset_lineage(
            records=[
                build_record(
                    task_instructions=(
                        "Inspect the backend repository."
                    )
                )
            ],

            curation_report=(
                build_curation_report(
                    trajectory
                )
            ),
        )


def test_dataset_task_lineage_rejects_changed_task_id(
    tmp_path: Path,
):

    trajectory = (
        build_trajectory(
            task_instructions=(
                TASK_CONTEXT
            )
        )
    )

    pipeline = (
        build_pipeline(
            tmp_path=(
                tmp_path
            ),

            trajectory=(
                trajectory
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "task_id does not match"
        ),
    ):

        pipeline._assert_dataset_lineage(
            records=[
                build_record(
                    task_id=(
                        "different-task"
                    )
                )
            ],

            curation_report=(
                build_curation_report(
                    trajectory
                )
            ),
        )