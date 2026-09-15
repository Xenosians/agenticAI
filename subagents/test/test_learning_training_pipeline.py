import json

from pathlib import (
    Path,
)

import pytest

from learning.datasets import (
    PreferenceDatasetBuilder,
)

from learning.diversity_gate import (
    DiversityGatePolicy,
)

from learning.preferences import (
    build_preference_example,
)

from learning.training_pipeline import (
    TrustedTrainingPipeline,
)

from learning.types import (
    CorrectionEvent,
    CorrectionValue,
    ExecutionReward,
    LearningTrajectory,
    PreferenceExample,
    TrajectorySignals,
    TrajectoryStep,
)


# ============================================================
# JSONL
# ============================================================


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


# ============================================================
# TRAJECTORY FIXTURES
# ============================================================


def build_trajectory(
    *,
    index: int,
    request: str,
    agent: str,
    tool: str,
    arguments: dict,
) -> LearningTrajectory:

    trajectory_id = (
        f"trajectory-{index}"
    )

    return (
        LearningTrajectory(
            trajectory_id=(
                trajectory_id
            ),

            observed_at=(
                "2026-09-15T00:00:00+00:00"
            ),

            job_id=(
                f"job-{index}"
            ),

            attempt=1,

            user_request=(
                request
            ),

            hub_model=(
                "hub-main"
            ),

            hub_status=(
                "success"
            ),

            routes=[
                agent,
            ],

            steps=[
                TrajectoryStep(
                    task_id=(
                        f"task-{index}"
                    ),

                    agent=(
                        agent
                    ),

                    status=(
                        "success"
                    ),

                    outcome_code=(
                        "success"
                    ),

                    proposed_tool=(
                        tool
                    ),

                    proposed_arguments=(
                        arguments
                    ),
                )
            ],

            final_answer=(
                "Observed answer."
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

            dataset_eligible=True,
        )
    )


def build_correction(
    *,
    index: int,
) -> CorrectionEvent:

    return (
        CorrectionEvent(
            correction_id=(
                f"correction-{index}"
            ),

            observed_at=(
                "2026-09-15T00:00:01+00:00"
            ),

            trajectory_id=(
                f"trajectory-{index}"
            ),

            task_id=(
                f"task-{index}"
            ),

            correction_type=(
                "answer"
            ),

            source=(
                "trusted_review"
            ),

            values=[
                CorrectionValue(
                    field=(
                        "answer"
                    ),

                    rejected_value=(
                        "Observed answer."
                    ),

                    chosen_value=(
                        f"Reviewed answer {index}."
                    ),
                )
            ],

            dataset_eligible=True,
        )
    )


def safe_trajectories(
) -> list[
    LearningTrajectory
]:

    return [
        build_trajectory(
            index=1,

            request=(
                "Synthetic pipeline frontend status."
            ),

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
        ),

        build_trajectory(
            index=2,

            request=(
                "Synthetic pipeline backend branches."
            ),

            agent=(
                "developer-specialist"
            ),

            tool=(
                "workspace_git_branches"
            ),

            arguments={
                "repository":
                    "backend"
            },
        ),

        build_trajectory(
            index=3,

            request=(
                "Synthetic pipeline ticket state."
            ),

            agent=(
                "ticket-specialist"
            ),

            tool=(
                "ticket_get"
            ),

            arguments={
                "ticket_key":
                    "SYN-103"
            },
        ),

        build_trajectory(
            index=4,

            request=(
                "Synthetic pipeline ticket comments."
            ),

            agent=(
                "ticket-specialist"
            ),

            tool=(
                "ticket_comments"
            ),

            arguments={
                "ticket_key":
                    "SYN-104",

                "limit":
                    4,
            },
        ),
    ]


def safe_corrections(
) -> list[
    CorrectionEvent
]:

    return [
        build_correction(
            index=index
        )

        for index
        in range(
            1,
            5,
        )
    ]


# ============================================================
# POLICY
# ============================================================


def relaxed_policy(
) -> DiversityGatePolicy:

    return (
        DiversityGatePolicy(
            min_eligible_trajectories=4,
            min_unique_requests=4,
            min_unique_behavior_patterns=4,
            min_unique_domains=2,
            min_unique_capabilities=4,
            max_duplicate_request_rate=0.25,
            max_dominant_domain_share=0.75,
            max_dominant_capability_share=0.75,
        )
    )


# ============================================================
# PROJECT BUILDERS
# ============================================================


def build_raw_evidence(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    list[
        LearningTrajectory
    ],
    list[
        CorrectionEvent
    ],
]:

    trajectory_path = (
        tmp_path
        / "runtime"
        / "trajectories.jsonl"
    )

    correction_path = (
        tmp_path
        / "runtime"
        / "corrections.jsonl"
    )

    trajectories = (
        safe_trajectories()
    )

    corrections = (
        safe_corrections()
    )

    write_jsonl(
        trajectory_path,
        trajectories,
    )

    write_jsonl(
        correction_path,
        corrections,
    )

    return (
        trajectory_path,
        correction_path,
        trajectories,
        corrections,
    )


def build_examples(
    *,
    trajectories: list[
        LearningTrajectory
    ],
    corrections: list[
        CorrectionEvent
    ],
) -> list[
    PreferenceExample
]:

    correction_by_trajectory = {
        correction.trajectory_id:
            correction

        for correction
        in corrections
    }

    return [
        build_preference_example(
            trajectory=(
                trajectory
            ),

            correction=(
                correction_by_trajectory[
                    trajectory.trajectory_id
                ]
            ),
        )

        for trajectory
        in trajectories
    ]


def build_dataset(
    *,
    tmp_path: Path,
    examples: list[
        PreferenceExample
    ],
) -> Path:

    dataset_root = (
        tmp_path
        / "datasets"
    )

    builder = (
        PreferenceDatasetBuilder(
            root=(
                dataset_root
            ),

            # Synthetic test data must not depend on the real
            # repository held-out suites.
            eval_paths=[],
        )
    )

    builder.promote(
        examples=(
            examples
        ),

        promoted_by=(
            "trusted_review"
        ),

        promotion_reason=(
            "Synthetic trusted pipeline test."
        ),
    )

    return (
        dataset_root
    )


def build_pipeline(
    *,
    trajectory_path: Path,
    correction_path: Path,
    dataset_root: Path,
    tmp_path: Path,
) -> TrustedTrainingPipeline:

    return (
        TrustedTrainingPipeline(
            trajectory_path=(
                trajectory_path
            ),

            correction_path=(
                correction_path
            ),

            dataset_root=(
                dataset_root
            ),

            output_root=(
                tmp_path
                / "training-exports"
            ),
        )
    )


# ============================================================
# TESTS
# ============================================================


def test_trusted_pipeline_exports_verified_split(
    tmp_path: Path,
):

    (
        trajectory_path,
        correction_path,
        trajectories,
        corrections,
    ) = (
        build_raw_evidence(
            tmp_path
        )
    )

    examples = (
        build_examples(
            trajectories=(
                trajectories
            ),

            corrections=(
                corrections
            ),
        )
    )

    dataset_root = (
        build_dataset(
            tmp_path=(
                tmp_path
            ),

            examples=(
                examples
            ),
        )
    )

    pipeline = (
        build_pipeline(
            trajectory_path=(
                trajectory_path
            ),

            correction_path=(
                correction_path
            ),

            dataset_root=(
                dataset_root
            ),

            tmp_path=(
                tmp_path
            ),
        )
    )

    result = (
        pipeline.run(
            dataset_version=(
                "v000001"
            ),

            validation_fraction=0.25,

            diversity_policy=(
                relaxed_policy()
            ),

            eval_paths=[],
        )
    )

    assert (
        result
        .diversity
        .promotion_eligible
        is True
    )

    assert (
        result.curated_eligible_count
        == 4
    )

    assert (
        result.dataset_record_count
        == 4
    )

    assert (
        result
        .split
        .record_count
        == 4
    )

    assert (
        result
        .split
        .train_record_count
        + result
        .split
        .validation_record_count
        == 4
    )


def test_pipeline_refuses_failed_diversity_gate(
    tmp_path: Path,
):

    (
        trajectory_path,
        correction_path,
        trajectories,
        corrections,
    ) = (
        build_raw_evidence(
            tmp_path
        )
    )

    dataset_root = (
        build_dataset(
            tmp_path=(
                tmp_path
            ),

            examples=(
                build_examples(
                    trajectories=(
                        trajectories
                    ),

                    corrections=(
                        corrections
                    ),
                )
            ),
        )
    )

    pipeline = (
        build_pipeline(
            trajectory_path=(
                trajectory_path
            ),

            correction_path=(
                correction_path
            ),

            dataset_root=(
                dataset_root
            ),

            tmp_path=(
                tmp_path
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "Diversity/balance gate failed"
        ),
    ):

        pipeline.run(
            dataset_version=(
                "v000001"
            ),

            # Defaults require substantially more evidence.
            diversity_policy=(
                DiversityGatePolicy()
            ),

            eval_paths=[],
        )

    assert not (
        tmp_path
        / "training-exports"
    ).exists()


def test_pipeline_refuses_curation_contamination(
    tmp_path: Path,
):

    (
        trajectory_path,
        correction_path,
        trajectories,
        corrections,
    ) = (
        build_raw_evidence(
            tmp_path
        )
    )

    dataset_root = (
        build_dataset(
            tmp_path=(
                tmp_path
            ),

            examples=(
                build_examples(
                    trajectories=(
                        trajectories
                    ),

                    corrections=(
                        corrections
                    ),
                )
            ),
        )
    )

    eval_path = (
        tmp_path
        / "held-out.jsonl"
    )

    eval_path.write_text(
        json.dumps(
            {
                "schema":
                    "evaluation-case.v1",

                "case_id":
                    "synthetic.pipeline.frontend",

                "suite":
                    "synthetic",

                "user_request":
                    (
                        " SYNTHETIC   PIPELINE "
                        "FRONTEND STATUS. "
                    ),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    pipeline = (
        build_pipeline(
            trajectory_path=(
                trajectory_path
            ),

            correction_path=(
                correction_path
            ),

            dataset_root=(
                dataset_root
            ),

            tmp_path=(
                tmp_path
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "Curation report contains "
            "held-out evaluation contamination"
        ),
    ):

        pipeline.run(
            dataset_version=(
                "v000001"
            ),

            diversity_policy=(
                relaxed_policy()
            ),

            eval_paths=[
                eval_path,
            ],
        )

    assert not (
        tmp_path
        / "training-exports"
    ).exists()


def test_pipeline_refuses_dataset_trajectory_outside_curation(
    tmp_path: Path,
):

    (
        trajectory_path,
        correction_path,
        trajectories,
        corrections,
    ) = (
        build_raw_evidence(
            tmp_path
        )
    )

    examples = (
        build_examples(
            trajectories=(
                trajectories
            ),

            corrections=(
                corrections
            ),
        )
    )

    rogue = (
        examples[
            0
        ]
        .model_copy(
            deep=True,

            update={
                "example_id":
                    "rogue-example",

                "trajectory_id":
                    "trajectory-outside-curation",

                "correction_id":
                    "rogue-correction",
            },
        )
    )

    dataset_root = (
        build_dataset(
            tmp_path=(
                tmp_path
            ),

            examples=(
                examples
                + [
                    rogue
                ]
            ),
        )
    )

    pipeline = (
        build_pipeline(
            trajectory_path=(
                trajectory_path
            ),

            correction_path=(
                correction_path
            ),

            dataset_root=(
                dataset_root
            ),

            tmp_path=(
                tmp_path
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "not present in "
            "curation_report.eligible"
        ),
    ):

        pipeline.run(
            dataset_version=(
                "v000001"
            ),

            diversity_policy=(
                relaxed_policy()
            ),

            eval_paths=[],
        )


def test_pipeline_refuses_unapproved_correction_lineage(
    tmp_path: Path,
):

    (
        trajectory_path,
        correction_path,
        trajectories,
        corrections,
    ) = (
        build_raw_evidence(
            tmp_path
        )
    )

    examples = (
        build_examples(
            trajectories=(
                trajectories
            ),

            corrections=(
                corrections
            ),
        )
    )

    examples[
        0
    ] = (
        examples[
            0
        ]
        .model_copy(
            deep=True,

            update={
                "correction_id":
                    "not-approved-correction",
            },
        )
    )

    dataset_root = (
        build_dataset(
            tmp_path=(
                tmp_path
            ),

            examples=(
                examples
            ),
        )
    )

    pipeline = (
        build_pipeline(
            trajectory_path=(
                trajectory_path
            ),

            correction_path=(
                correction_path
            ),

            dataset_root=(
                dataset_root
            ),

            tmp_path=(
                tmp_path
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "correction_id not approved "
            "by curation"
        ),
    ):

        pipeline.run(
            dataset_version=(
                "v000001"
            ),

            diversity_policy=(
                relaxed_policy()
            ),

            eval_paths=[],
        )


def test_pipeline_refuses_dataset_request_mismatch(
    tmp_path: Path,
):

    (
        trajectory_path,
        correction_path,
        trajectories,
        corrections,
    ) = (
        build_raw_evidence(
            tmp_path
        )
    )

    examples = (
        build_examples(
            trajectories=(
                trajectories
            ),

            corrections=(
                corrections
            ),
        )
    )

    examples[
        0
    ] = (
        examples[
            0
        ]
        .model_copy(
            deep=True,

            update={
                "user_request":
                    "Completely unrelated synthetic request.",
            },
        )
    )

    dataset_root = (
        build_dataset(
            tmp_path=(
                tmp_path
            ),

            examples=(
                examples
            ),
        )
    )

    pipeline = (
        build_pipeline(
            trajectory_path=(
                trajectory_path
            ),

            correction_path=(
                correction_path
            ),

            dataset_root=(
                dataset_root
            ),

            tmp_path=(
                tmp_path
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "user_request does not match"
        ),
    ):

        pipeline.run(
            dataset_version=(
                "v000001"
            ),

            diversity_policy=(
                relaxed_policy()
            ),

            eval_paths=[],
        )


def test_pipeline_refuses_corrupt_dataset(
    tmp_path: Path,
):

    (
        trajectory_path,
        correction_path,
        trajectories,
        corrections,
    ) = (
        build_raw_evidence(
            tmp_path
        )
    )

    dataset_root = (
        build_dataset(
            tmp_path=(
                tmp_path
            ),

            examples=(
                build_examples(
                    trajectories=(
                        trajectories
                    ),

                    corrections=(
                        corrections
                    ),
                )
            ),
        )
    )

    records_path = (
        dataset_root
        / "preference"
        / "v000001"
        / "records.jsonl"
    )

    records_path.write_text(
        (
            records_path
            .read_text(
                encoding="utf-8"
            )
            + "\n"
        ),
        encoding="utf-8",
    )

    pipeline = (
        build_pipeline(
            trajectory_path=(
                trajectory_path
            ),

            correction_path=(
                correction_path
            ),

            dataset_root=(
                dataset_root
            ),

            tmp_path=(
                tmp_path
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "Dataset verification failed"
        ),
    ):

        pipeline.run(
            dataset_version=(
                "v000001"
            ),

            diversity_policy=(
                relaxed_policy()
            ),

            eval_paths=[],
        )

    assert not (
        tmp_path
        / "training-exports"
    ).exists()