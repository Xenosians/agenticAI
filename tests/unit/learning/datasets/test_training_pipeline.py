import json

from pathlib import (
    Path,
)

import pytest

from learning.datasets.records import (
    PreferenceDatasetBuilder,
)

from learning.curation.diversity_gate import (
    DiversityGatePolicy,
)

from learning.curation.preferences import (
    build_preference_example,
)

from learning.curation.reviews import (
    ReviewDecision,
)

from learning.datasets.training_pipeline import (
    TrustedTrainingPipeline,
)

from learning.evidence.types import (
    CorrectionEvent,
    CorrectionValue,
    ExecutionReward,
    LearningTrajectory,
    PreferenceExample,
    TrajectorySignals,
    TrajectoryStep,
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
    index: int,
    request: str,
    agent: str,
    tool: str,
    arguments: dict,
) -> LearningTrajectory:

    return (
        LearningTrajectory(
            trajectory_id=(
                f"trajectory-{index}"
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

                    task_instructions=(
                        f"Handle synthetic task {index}."
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

            dataset_eligible=False,
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
                "explicit_user"
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

            dataset_eligible=False,
        )
    )


def review(
    *,
    review_id: str,
    subject_type: str,
    subject_id: str,
) -> ReviewDecision:

    return (
        ReviewDecision(
            review_id=(
                review_id
            ),

            observed_at=(
                "2026-09-15T00:00:02+00:00"
            ),

            subject_type=(
                subject_type
            ),

            subject_id=(
                subject_id
            ),

            decision=(
                "approve"
            ),

            source=(
                "trusted_review"
            ),

            reason=(
                "Verified synthetic evidence."
            ),
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
            index=(
                index
            )
        )

        for index
        in range(
            1,
            5,
        )
    ]


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


def build_raw_evidence(
    tmp_path: Path,
) -> tuple[
    Path,
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

    review_path = (
        tmp_path
        / "runtime"
        / "reviews.jsonl"
    )

    trajectories = (
        safe_trajectories()
    )

    corrections = (
        safe_corrections()
    )

    reviews = []

    for index in range(
        1,
        5,
    ):

        reviews.append(
            review(
                review_id=(
                    f"trajectory-review-{index}"
                ),

                subject_type=(
                    "trajectory"
                ),

                subject_id=(
                    f"trajectory-{index}"
                ),
            )
        )

        reviews.append(
            review(
                review_id=(
                    f"correction-review-{index}"
                ),

                subject_type=(
                    "correction"
                ),

                subject_id=(
                    f"correction-{index}"
                ),
            )
        )

    write_jsonl(
        trajectory_path,
        trajectories,
    )

    write_jsonl(
        correction_path,
        corrections,
    )

    write_jsonl(
        review_path,
        reviews,
    )

    return (
        trajectory_path,
        correction_path,
        review_path,
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


def build_dataset_at_root(
    *,
    dataset_root: Path,
    examples: list[
        PreferenceExample
    ],
) -> Path:

    builder = (
        PreferenceDatasetBuilder(
            root=(
                dataset_root
            ),

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


def build_dataset(
    *,
    tmp_path: Path,
    examples: list[
        PreferenceExample
    ],
) -> Path:

    return (
        build_dataset_at_root(
            dataset_root=(
                tmp_path
                / "datasets"
            ),

            examples=(
                examples
            ),
        )
    )


def build_pipeline(
    *,
    trajectory_path: Path,
    correction_path: Path,
    review_path: Path,
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

            review_path=(
                review_path
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


def prepare(
    tmp_path: Path,
):

    (
        trajectory_path,
        correction_path,
        review_path,
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

            review_path=(
                review_path
            ),

            dataset_root=(
                dataset_root
            ),

            tmp_path=(
                tmp_path
            ),
        )
    )

    return (
        pipeline,
        trajectory_path,
        correction_path,
        review_path,
        dataset_root,
        trajectories,
        corrections,
        examples,
    )


def test_trusted_pipeline_exports_verified_split(
    tmp_path: Path,
):

    (
        pipeline,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = prepare(
        tmp_path
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
        result.diversity.promotion_eligible
        is True
    )

    # The diversity report now represents the exact source set of
    # the promoted dataset.
    assert (
        result
        .diversity
        .metrics
        .eligible_trajectory_count
        == 4
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
        result.split.record_count
        == 4
    )


def test_pipeline_refuses_failed_diversity_gate(
    tmp_path: Path,
):

    (
        pipeline,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = prepare(
        tmp_path
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

            diversity_policy=(
                DiversityGatePolicy()
            ),

            eval_paths=[],
        )


def test_pipeline_applies_diversity_to_exact_dataset_source_set(
    tmp_path: Path,
):
    """
    The complete curated corpus contains four diverse trusted
    trajectories and satisfies relaxed_policy().

    This dataset intentionally contains only two of them.

    The old pipeline incorrectly evaluated all four curated
    trajectories and therefore allowed the narrower dataset to
    inherit diversity it did not possess.

    The trusted pipeline must now fail the dataset itself.
    """

    (
        _,
        trajectory_path,
        correction_path,
        review_path,
        _,
        _,
        _,
        examples,
    ) = prepare(
        tmp_path
    )

    narrow_dataset_root = (
        tmp_path
        / "narrow-datasets"
    )

    build_dataset_at_root(
        dataset_root=(
            narrow_dataset_root
        ),

        examples=(
            examples[
                :2
            ]
        ),
    )

    pipeline = (
        build_pipeline(
            trajectory_path=(
                trajectory_path
            ),

            correction_path=(
                correction_path
            ),

            review_path=(
                review_path
            ),

            dataset_root=(
                narrow_dataset_root
            ),

            tmp_path=(
                tmp_path
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "minimum_eligible_trajectories"
        ),
    ):

        pipeline.run(
            dataset_version=(
                "v000001"
            ),

            validation_fraction=0.50,

            diversity_policy=(
                relaxed_policy()
            ),

            eval_paths=[],
        )


def test_historical_contamination_is_quarantined_not_global_poison(
    tmp_path: Path,
):

    (
        pipeline,
        trajectory_path,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = prepare(
        tmp_path
    )

    contaminated = (
        build_trajectory(
            index=99,

            request=(
                "Synthetic historical held-out request."
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
        )
    )

    with trajectory_path.open(
        "a",
        encoding="utf-8",
    ) as handle:

        handle.write(
            contaminated.model_dump_json(
                by_alias=True
            )
        )

        handle.write(
            "\n"
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
                    "historical.contamination",

                "suite":
                    "synthetic",

                "user_request":
                    (
                        " SYNTHETIC HISTORICAL "
                        "HELD-OUT REQUEST. "
                    ),
            }
        )
        + "\n",

        encoding="utf-8",
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

            eval_paths=[
                eval_path,
            ],
        )
    )

    assert (
        result
        .quarantined_contamination_count
        == 1
    )

    assert (
        result.curated_eligible_count
        == 4
    )

    assert (
        result
        .diversity
        .metrics
        .eligible_trajectory_count
        == 4
    )


def test_pipeline_refuses_dataset_trajectory_outside_curation(
    tmp_path: Path,
):

    (
        _,
        trajectory_path,
        correction_path,
        review_path,
        _,
        _,
        _,
        examples,
    ) = prepare(
        tmp_path
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
        tmp_path
        / "rogue-datasets"
    )

    builder = (
        PreferenceDatasetBuilder(
            root=(
                dataset_root
            ),

            eval_paths=[],
        )
    )

    builder.promote(
        examples=(
            examples
            + [
                rogue
            ]
        ),

        promoted_by=(
            "trusted_review"
        ),

        promotion_reason=(
            "Rogue lineage test."
        ),
    )

    pipeline = (
        build_pipeline(
            trajectory_path=(
                trajectory_path
            ),

            correction_path=(
                correction_path
            ),

            review_path=(
                review_path
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
        _,
        trajectory_path,
        correction_path,
        review_path,
        _,
        _,
        corrections,
        examples,
    ) = prepare(
        tmp_path
    )

    # Exists in immutable raw evidence but has no trusted approval.
    unapproved_correction = (
        corrections[
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

    with correction_path.open(
        "a",
        encoding="utf-8",
    ) as handle:

        handle.write(
            unapproved_correction.model_dump_json(
                by_alias=True
            )
        )

        handle.write(
            "\n"
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
        tmp_path
        / "bad-correction-datasets"
    )

    builder = (
        PreferenceDatasetBuilder(
            root=(
                dataset_root
            ),

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
            "Bad correction lineage."
        ),
    )

    pipeline = (
        build_pipeline(
            trajectory_path=(
                trajectory_path
            ),

            correction_path=(
                correction_path
            ),

            review_path=(
                review_path
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
        _,
        trajectory_path,
        correction_path,
        review_path,
        _,
        _,
        _,
        examples,
    ) = prepare(
        tmp_path
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
                    (
                        "Completely unrelated "
                        "synthetic request."
                    ),
            },
        )
    )

    dataset_root = (
        tmp_path
        / "bad-request-datasets"
    )

    builder = (
        PreferenceDatasetBuilder(
            root=(
                dataset_root
            ),

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
            "Bad request lineage."
        ),
    )

    pipeline = (
        build_pipeline(
            trajectory_path=(
                trajectory_path
            ),

            correction_path=(
                correction_path
            ),

            review_path=(
                review_path
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
        pipeline,
        _,
        _,
        _,
        dataset_root,
        _,
        _,
        _,
    ) = prepare(
        tmp_path
    )

    records_path = (
        dataset_root
        / "preference"
        / "v000001"
        / "records.jsonl"
    )

    records_path.write_text(
        (
            records_path.read_text(
                encoding="utf-8"
            )
            + "\n"
        ),

        encoding="utf-8",
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