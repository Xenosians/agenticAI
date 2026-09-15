import json

from pathlib import (
    Path,
)

from learning.curation import (
    CORRECTION_DATASET_INELIGIBLE,
    DUPLICATE_EVIDENCE,
    HELD_OUT_CONTAMINATION,
    TRAJECTORY_DATASET_INELIGIBLE,
    UNKNOWN_TRUSTED_OUTCOME,
    UNTRUSTED_CORRECTION_PROVENANCE,
    curate_corpus,
)

from learning.types import (
    CorrectionEvent,
    CorrectionValue,
    ExecutionReward,
    LearningTrajectory,
    TrajectorySignals,
    TrajectoryStep,
)

from learning.reviews import (
    ReviewDecision,
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

            if hasattr(
                item,
                "model_dump"
            ):

                payload = (
                    item.model_dump(
                        mode="json",
                        by_alias=True,
                    )
                )

            else:

                payload = item

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
    trajectory_id: str,
    user_request: str,
    dataset_eligible: bool = True,
    outcome_code: str | None = "success",
    repository: str = "frontend",
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
                user_request
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
                        outcome_code
                    ),

                    proposed_tool=(
                        "workspace_git_status"
                    ),

                    proposed_arguments={
                        "repository":
                            repository
                    },
                )
            ],

            final_answer=(
                "Done."
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

            dataset_eligible=(
                dataset_eligible
            ),
        )
    )


def build_correction(
    *,
    correction_id: str,
    trajectory_id: str,
    source: str = "trusted_review",
    dataset_eligible: bool = True,
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
                source
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

            dataset_eligible=(
                dataset_eligible
            ),
        )
    )


def test_clean_trusted_evidence_is_eligible(
    tmp_path: Path,
):

    trajectories = (
        tmp_path
        / "trajectories.jsonl"
    )

    corrections = (
        tmp_path
        / "corrections.jsonl"
    )

    write_jsonl(
        trajectories,
        [
            build_trajectory(
                trajectory_id="t1",
                user_request=(
                    "Show frontend status."
                ),
            )
        ],
    )

    write_jsonl(
        corrections,
        [
            build_correction(
                correction_id="c1",
                trajectory_id="t1",
            )
        ],
    )

    report = (
        curate_corpus(
            trajectory_path=(
                trajectories
            ),

            correction_path=(
                corrections
            ),
        )
    )

    assert (
        report.eligible_trajectory_count
        == 1
    )

    assert (
        report.excluded_trajectory_count
        == 0
    )

    assert (
        report.usable_correction_count
        == 1
    )

    assert (
        report.eligible[
            0
        ].usable_correction_ids
        == [
            "c1"
        ]
    )


def test_raw_unapproved_evidence_is_excluded(
    tmp_path: Path,
):

    trajectories = (
        tmp_path
        / "trajectories.jsonl"
    )

    corrections = (
        tmp_path
        / "corrections.jsonl"
    )

    write_jsonl(
        trajectories,
        [
            build_trajectory(
                trajectory_id="t1",
                user_request=(
                    "Show frontend status."
                ),
                dataset_eligible=False,
            )
        ],
    )

    write_jsonl(
        corrections,
        [
            build_correction(
                correction_id="c1",
                trajectory_id="t1",
                source=(
                    "explicit_user"
                ),
                dataset_eligible=False,
            )
        ],
    )

    report = (
        curate_corpus(
            trajectory_path=(
                trajectories
            ),

            correction_path=(
                corrections
            ),
        )
    )

    assert (
        report.eligible_trajectory_count
        == 0
    )

    reasons = (
        report.excluded[
            0
        ].reasons
    )

    assert (
        TRAJECTORY_DATASET_INELIGIBLE
        in reasons
    )

    assert (
        UNTRUSTED_CORRECTION_PROVENANCE
        in reasons
    )

    assert (
        CORRECTION_DATASET_INELIGIBLE
        in reasons
    )


def test_held_out_contamination_is_excluded(
    tmp_path: Path,
):

    trajectories = (
        tmp_path
        / "trajectories.jsonl"
    )

    corrections = (
        tmp_path
        / "corrections.jsonl"
    )

    eval_path = (
        tmp_path
        / "core.v2.jsonl"
    )

    write_jsonl(
        trajectories,
        [
            build_trajectory(
                trajectory_id="t1",
                user_request=(
                    "Show frontend status."
                ),
            )
        ],
    )

    write_jsonl(
        corrections,
        [],
    )

    write_jsonl(
        eval_path,
        [
            {
                "schema":
                    "evaluation-case.v1",

                "case_id":
                    "git.frontend.status",

                "suite":
                    "core.v2",

                "user_request":
                    "  SHOW FRONTEND STATUS. ",
            }
        ],
    )

    report = (
        curate_corpus(
            trajectory_path=(
                trajectories
            ),

            correction_path=(
                corrections
            ),

            eval_paths=[
                eval_path,
            ],
        )
    )

    assert (
        report.eligible_trajectory_count
        == 0
    )

    assert (
        report.held_out_contamination_count
        == 1
    )

    assert (
        HELD_OUT_CONTAMINATION
        in report.excluded[
            0
        ].reasons
    )

    assert (
        report
        .contamination_matches[
            0
        ]
        .eval_cases
        == [
            "core.v2:git.frontend.status"
        ]
    )


def test_unknown_outcome_is_excluded(
    tmp_path: Path,
):

    trajectories = (
        tmp_path
        / "trajectories.jsonl"
    )

    corrections = (
        tmp_path
        / "corrections.jsonl"
    )

    write_jsonl(
        trajectories,
        [
            build_trajectory(
                trajectory_id="t1",
                user_request=(
                    "Show frontend status."
                ),
                outcome_code=(
                    "mystery_outcome"
                ),
            )
        ],
    )

    write_jsonl(
        corrections,
        [],
    )

    report = (
        curate_corpus(
            trajectory_path=(
                trajectories
            ),

            correction_path=(
                corrections
            ),
        )
    )

    assert (
        report.eligible_trajectory_count
        == 0
    )

    assert (
        UNKNOWN_TRUSTED_OUTCOME
        in report.excluded[
            0
        ].reasons
    )


def test_duplicate_evidence_is_deduplicated(
    tmp_path: Path,
):

    trajectories = (
        tmp_path
        / "trajectories.jsonl"
    )

    corrections = (
        tmp_path
        / "corrections.jsonl"
    )

    write_jsonl(
        trajectories,
        [
            build_trajectory(
                trajectory_id="t1",
                user_request=(
                    "Show frontend status."
                ),
            ),

            build_trajectory(
                trajectory_id="t2",
                user_request=(
                    "  SHOW   FRONTEND STATUS. "
                ),
            ),
        ],
    )

    write_jsonl(
        corrections,
        [],
    )

    report = (
        curate_corpus(
            trajectory_path=(
                trajectories
            ),

            correction_path=(
                corrections
            ),
        )
    )

    assert (
        report.eligible_trajectory_count
        == 1
    )

    assert (
        report.excluded_trajectory_count
        == 1
    )

    assert (
        report.deduplicated_trajectory_count
        == 1
    )

    duplicate = (
        report.excluded[
            0
        ]
    )

    assert (
        DUPLICATE_EVIDENCE
        in duplicate.reasons
    )

    assert (
        duplicate
        .duplicate_of_trajectory_id
        == "t1"
    )


def test_orphan_correction_is_quarantined(
    tmp_path: Path,
):

    trajectories = (
        tmp_path
        / "trajectories.jsonl"
    )

    corrections = (
        tmp_path
        / "corrections.jsonl"
    )

    write_jsonl(
        trajectories,
        [
            build_trajectory(
                trajectory_id="t1",
                user_request=(
                    "Show frontend status."
                ),
            )
        ],
    )

    write_jsonl(
        corrections,
        [
            build_correction(
                correction_id="orphan-c1",
                trajectory_id="missing",
            )
        ],
    )

    report = (
        curate_corpus(
            trajectory_path=(
                trajectories
            ),

            correction_path=(
                corrections
            ),
        )
    )

    assert (
        report.orphan_correction_count
        == 1
    )

    assert (
        report.orphan_correction_ids
        == [
            "orphan-c1"
        ]
    )

    assert (
        report.usable_correction_count
        == 0
    )
    
def write_reviews(
    path: Path,
    reviews,
) -> None:

    write_jsonl(
        path,
        reviews,
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
                "Verified test evidence."
            ),
        )
    )


def test_review_ledger_can_approve_raw_trajectory(
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

    write_jsonl(
        trajectory_path,
        [
            build_trajectory(
                trajectory_id="t1",
                user_request=(
                    "Synthetic reviewed request."
                ),
                dataset_eligible=False,
            )
        ],
    )

    write_jsonl(
        correction_path,
        [],
    )

    write_reviews(
        review_path,
        [
            review(
                review_id="r1",
                subject_type="trajectory",
                subject_id="t1",
            )
        ],
    )

    report = (
        curate_corpus(
            trajectory_path=(
                trajectory_path
            ),

            correction_path=(
                correction_path
            ),

            review_path=(
                review_path
            ),
        )
    )

    assert (
        report.eligible_trajectory_count
        == 1
    )

    assert (
        report.review_count
        == 1
    )


def test_review_ledger_rejection_overrides_dataset_flag(
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

    write_jsonl(
        trajectory_path,
        [
            build_trajectory(
                trajectory_id="t1",
                user_request=(
                    "Synthetic rejected request."
                ),
                dataset_eligible=True,
            )
        ],
    )

    write_jsonl(
        correction_path,
        [],
    )

    write_reviews(
        review_path,
        [
            review(
                review_id="r1",
                subject_type="trajectory",
                subject_id="t1",
                decision="reject",
            )
        ],
    )

    report = (
        curate_corpus(
            trajectory_path=(
                trajectory_path
            ),

            correction_path=(
                correction_path
            ),

            review_path=(
                review_path
            ),
        )
    )

    assert (
        report.eligible_trajectory_count
        == 0
    )

    assert (
        "trajectory_review_not_approved"
        in report.excluded[
            0
        ].reasons
    )


def test_reviewed_explicit_user_correction_becomes_usable(
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

    write_jsonl(
        trajectory_path,
        [
            build_trajectory(
                trajectory_id="t1",
                user_request=(
                    "Synthetic corrected request."
                ),
                dataset_eligible=False,
            )
        ],
    )

    write_jsonl(
        correction_path,
        [
            build_correction(
                correction_id="c1",
                trajectory_id="t1",
            )
        ],
    )

    write_reviews(
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

    report = (
        curate_corpus(
            trajectory_path=(
                trajectory_path
            ),

            correction_path=(
                correction_path
            ),

            review_path=(
                review_path
            ),
        )
    )

    assert (
        report.eligible_trajectory_count
        == 1
    )

    assert (
        report.usable_correction_count
        == 1
    )

    assert (
        report.eligible[
            0
        ].usable_correction_ids
        == [
            "c1"
        ]
    )