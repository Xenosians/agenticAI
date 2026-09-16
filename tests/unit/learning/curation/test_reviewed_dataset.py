import json

from pathlib import (
    Path,
)

import pytest

from learning.curation.reviewed_dataset import (
    ReviewedPreferenceDatasetBuilder,
)

from learning.curation.reviews import (
    ReviewDecision,
)

from learning.evidence.types import (
    CorrectionEvent,
    CorrectionValue,
    ExecutionReward,
    LearningTrajectory,
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


def trajectory(
    *,
    trajectory_id: str,
    request: str,
) -> LearningTrajectory:

    return (
        LearningTrajectory(
            trajectory_id=(
                trajectory_id
            ),

            observed_at=(
                "2026-09-15T00:00:00+00:00"
            ),

            job_id=(
                f"job-{trajectory_id}"
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
                "developer-specialist",
            ],

            steps=[
                TrajectoryStep(
                    task_id=(
                        f"task-{trajectory_id}"
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
                            "ai"
                    },
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


def correction(
    *,
    correction_id: str,
    trajectory_id: str,
) -> CorrectionEvent:

    return (
        CorrectionEvent(
            correction_id=(
                correction_id
            ),

            observed_at=(
                "2026-09-15T00:00:01+00:00"
            ),

            trajectory_id=(
                trajectory_id
            ),

            task_id=(
                f"task-{trajectory_id}"
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


def review(
    *,
    review_id: str,
    subject_type: str,
    subject_id: str,
    decision: str = "approve",
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
                decision
            ),

            source=(
                "trusted_review"
            ),

            reason=(
                "Verified."
            ),
        )
    )


def test_reviewed_evidence_creates_dataset(
    tmp_path: Path,
):

    trajectory_path = (
        tmp_path
        / "trajectories.jsonl"
    )

    correction_path = (
        tmp_path
        / "corrections.jsonl"
    )

    review_path = (
        tmp_path
        / "reviews.jsonl"
    )

    dataset_root = (
        tmp_path
        / "datasets"
    )

    write_jsonl(
        trajectory_path,
        [
            trajectory(
                trajectory_id="t1",
                request=(
                    "Synthetic reviewed dataset request."
                ),
            )
        ],
    )

    write_jsonl(
        correction_path,
        [
            correction(
                correction_id="c1",
                trajectory_id="t1",
            )
        ],
    )

    write_jsonl(
        review_path,
        [
            review(
                review_id="r1",
                subject_type="trajectory",
                subject_id="t1",
            ),

            review(
                review_id="r2",
                subject_type="correction",
                subject_id="c1",
            ),
        ],
    )

    result = (
        ReviewedPreferenceDatasetBuilder(
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
        )
        .build(
            eval_paths=[],

            promotion_reason=(
                "Verified synthetic evidence."
            ),
        )
    )

    assert (
        result.preference_example_count
        == 1
    )

    assert (
        result.manifest.version
        == "v000001"
    )

    assert (
        result.source_trajectory_ids
        == [
            "t1"
        ]
    )

    assert (
        result.source_correction_ids
        == [
            "c1"
        ]
    )


def test_unreviewed_trajectory_produces_no_dataset(
    tmp_path: Path,
):

    trajectory_path = (
        tmp_path
        / "trajectories.jsonl"
    )

    correction_path = (
        tmp_path
        / "corrections.jsonl"
    )

    review_path = (
        tmp_path
        / "reviews.jsonl"
    )

    dataset_root = (
        tmp_path
        / "datasets"
    )

    write_jsonl(
        trajectory_path,
        [
            trajectory(
                trajectory_id="t1",
                request=(
                    "Synthetic unreviewed request."
                ),
            )
        ],
    )

    write_jsonl(
        correction_path,
        [
            correction(
                correction_id="c1",
                trajectory_id="t1",
            )
        ],
    )

    write_jsonl(
        review_path,
        [],
    )

    with pytest.raises(
        ValueError,
        match=(
            "No trusted reviewed preference examples"
        ),
    ):

        ReviewedPreferenceDatasetBuilder(
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
        ).build(
            eval_paths=[],

            promotion_reason=(
                "Should not promote."
            ),
        )

    assert not (
        dataset_root
        .exists()
    )


def test_unreviewed_correction_produces_no_dataset(
    tmp_path: Path,
):

    trajectory_path = (
        tmp_path
        / "trajectories.jsonl"
    )

    correction_path = (
        tmp_path
        / "corrections.jsonl"
    )

    review_path = (
        tmp_path
        / "reviews.jsonl"
    )

    dataset_root = (
        tmp_path
        / "datasets"
    )

    write_jsonl(
        trajectory_path,
        [
            trajectory(
                trajectory_id="t1",
                request=(
                    "Synthetic correction review request."
                ),
            )
        ],
    )

    write_jsonl(
        correction_path,
        [
            correction(
                correction_id="c1",
                trajectory_id="t1",
            )
        ],
    )

    write_jsonl(
        review_path,
        [
            review(
                review_id="r1",
                subject_type="trajectory",
                subject_id="t1",
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "No trusted reviewed preference examples"
        ),
    ):

        ReviewedPreferenceDatasetBuilder(
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
        ).build(
            eval_paths=[],

            promotion_reason=(
                "Should not promote."
            ),
        )


def test_contaminated_reviewed_evidence_remains_quarantined(
    tmp_path: Path,
):

    trajectory_path = (
        tmp_path
        / "trajectories.jsonl"
    )

    correction_path = (
        tmp_path
        / "corrections.jsonl"
    )

    review_path = (
        tmp_path
        / "reviews.jsonl"
    )

    dataset_root = (
        tmp_path
        / "datasets"
    )

    eval_path = (
        tmp_path
        / "held-out.jsonl"
    )

    request = (
        "Synthetic contaminated reviewed request."
    )

    write_jsonl(
        trajectory_path,
        [
            trajectory(
                trajectory_id="t1",
                request=(
                    request
                ),
            )
        ],
    )

    write_jsonl(
        correction_path,
        [
            correction(
                correction_id="c1",
                trajectory_id="t1",
            )
        ],
    )

    write_jsonl(
        review_path,
        [
            review(
                review_id="r1",
                subject_type="trajectory",
                subject_id="t1",
            ),

            review(
                review_id="r2",
                subject_type="correction",
                subject_id="c1",
            ),
        ],
    )

    write_jsonl(
        eval_path,
        [
            {
                "schema":
                    "evaluation-case.v1",

                "case_id":
                    "synthetic.contamination",

                "suite":
                    "synthetic",

                "user_request":
                    request,
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "No trusted reviewed preference examples"
        ),
    ):

        ReviewedPreferenceDatasetBuilder(
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
        ).build(
            eval_paths=[
                eval_path,
            ],

            promotion_reason=(
                "Must remain quarantined."
            ),
        )


def test_latest_rejection_prevents_dataset_promotion(
    tmp_path: Path,
):

    trajectory_path = (
        tmp_path
        / "trajectories.jsonl"
    )

    correction_path = (
        tmp_path
        / "corrections.jsonl"
    )

    review_path = (
        tmp_path
        / "reviews.jsonl"
    )

    dataset_root = (
        tmp_path
        / "datasets"
    )

    write_jsonl(
        trajectory_path,
        [
            trajectory(
                trajectory_id="t1",
                request=(
                    "Synthetic revoked review request."
                ),
            )
        ],
    )

    write_jsonl(
        correction_path,
        [
            correction(
                correction_id="c1",
                trajectory_id="t1",
            )
        ],
    )

    write_jsonl(
        review_path,
        [
            review(
                review_id="r1",
                subject_type="trajectory",
                subject_id="t1",
            ),

            review(
                review_id="r2",
                subject_type="correction",
                subject_id="c1",
            ),

            review(
                review_id="r3",
                subject_type="trajectory",
                subject_id="t1",
                decision="reject",
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "No trusted reviewed preference examples"
        ),
    ):

        ReviewedPreferenceDatasetBuilder(
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
        ).build(
            eval_paths=[],

            promotion_reason=(
                "Must remain rejected."
            ),
        )