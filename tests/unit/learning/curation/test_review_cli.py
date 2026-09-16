from pathlib import (
    Path,
)

import pytest

from learning.cli.run_review import (
    review_subject,
)

from learning.curation.reviews import (
    load_reviews,
)

from tests.unit.learning.curation.test_curation import (
    build_correction,
    build_trajectory,
    write_jsonl,
)


def test_review_subject_approves_existing_trajectory(
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
                    "Synthetic reviewed trajectory."
                ),
                dataset_eligible=False,
            )
        ],
    )

    write_jsonl(
        correction_path,
        [],
    )

    review = (
        review_subject(
            trajectory_path=(
                trajectory_path
            ),

            correction_path=(
                correction_path
            ),

            review_path=(
                review_path
            ),

            subject_type=(
                "trajectory"
            ),

            subject_id=(
                "t1"
            ),

            decision=(
                "approve"
            ),

            source=(
                "trusted_review"
            ),

            reason=(
                "Verified behavior."
            ),
        )
    )

    assert (
        review.subject_id
        == "t1"
    )

    assert (
        review.decision
        == "approve"
    )

    persisted = (
        load_reviews(
            review_path
        )
    )

    assert (
        len(
            persisted
        )
        == 1
    )


def test_review_subject_rejects_missing_trajectory(
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
        [],
    )

    write_jsonl(
        correction_path,
        [],
    )

    with pytest.raises(
        ValueError,
        match=(
            "Trajectory does not exist"
        ),
    ):

        review_subject(
            trajectory_path=(
                trajectory_path
            ),

            correction_path=(
                correction_path
            ),

            review_path=(
                review_path
            ),

            subject_type=(
                "trajectory"
            ),

            subject_id=(
                "missing"
            ),

            decision=(
                "approve"
            ),

            source=(
                "trusted_review"
            ),

            reason=(
                "Should fail."
            ),
        )


def test_review_subject_approves_linked_correction(
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
                    "Synthetic corrected trajectory."
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

    review = (
        review_subject(
            trajectory_path=(
                trajectory_path
            ),

            correction_path=(
                correction_path
            ),

            review_path=(
                review_path
            ),

            subject_type=(
                "correction"
            ),

            subject_id=(
                "c1"
            ),

            decision=(
                "approve"
            ),

            source=(
                "trusted_review"
            ),

            reason=(
                "Correction verified."
            ),
        )
    )

    assert (
        review.subject_id
        == "c1"
    )


def test_review_subject_rejects_orphan_correction(
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
        [],
    )

    write_jsonl(
        correction_path,
        [
            build_correction(
                correction_id="c1",
                trajectory_id="missing",
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "Correction is orphaned"
        ),
    ):

        review_subject(
            trajectory_path=(
                trajectory_path
            ),

            correction_path=(
                correction_path
            ),

            review_path=(
                review_path
            ),

            subject_type=(
                "correction"
            ),

            subject_id=(
                "c1"
            ),

            decision=(
                "approve"
            ),

            source=(
                "trusted_review"
            ),

            reason=(
                "Should fail."
            ),
        )