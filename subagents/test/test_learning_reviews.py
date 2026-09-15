import json

from pathlib import (
    Path,
)

import pytest

from learning.reviews import (
    ReviewDecision,
    ReviewDecisionRecorder,
    latest_review_index,
    load_reviews,
    subject_is_approved,
)


def test_review_recorder_builds_trusted_approval(
):

    recorder = (
        ReviewDecisionRecorder(
            path=Path(
                "/tmp/unused-reviews.jsonl"
            ),
            enabled=True,
        )
    )

    review = (
        recorder.build(
            subject_type=(
                "trajectory"
            ),

            subject_id=(
                "trajectory-1"
            ),

            decision=(
                "approve"
            ),

            source=(
                "trusted_review"
            ),

            reason=(
                "Manually verified."
            ),
        )
    )

    assert (
        review.subject_type
        == "trajectory"
    )

    assert (
        review.subject_id
        == "trajectory-1"
    )

    assert (
        review.decision
        == "approve"
    )


def test_untrusted_review_source_is_rejected(
):

    recorder = (
        ReviewDecisionRecorder(
            path=Path(
                "/tmp/unused-reviews.jsonl"
            ),
            enabled=True,
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "Unsupported review source"
        ),
    ):

        recorder.build(
            subject_type=(
                "trajectory"
            ),

            subject_id=(
                "trajectory-1"
            ),

            decision=(
                "approve"
            ),

            source=(
                "explicit_user"
            ),

            reason=(
                "User liked it."
            ),
        )


def test_review_ledger_is_append_only(
    tmp_path: Path,
):

    path = (
        tmp_path
        / "reviews.jsonl"
    )

    recorder = (
        ReviewDecisionRecorder(
            path=(
                path
            ),
            enabled=True,
        )
    )

    recorder.record(
        subject_type=(
            "trajectory"
        ),

        subject_id=(
            "trajectory-1"
        ),

        decision=(
            "approve"
        ),

        source=(
            "trusted_review"
        ),

        reason=(
            "First review."
        ),
    )

    recorder.record(
        subject_type=(
            "trajectory"
        ),

        subject_id=(
            "trajectory-2"
        ),

        decision=(
            "reject"
        ),

        source=(
            "trusted_review"
        ),

        reason=(
            "Second review."
        ),
    )

    reviews = (
        load_reviews(
            path
        )
    )

    assert (
        len(
            reviews
        )
        == 2
    )

    assert (
        reviews[
            0
        ].subject_id
        == "trajectory-1"
    )

    assert (
        reviews[
            1
        ].subject_id
        == "trajectory-2"
    )


def test_latest_review_wins(
):

    reviews = [
        ReviewDecision(
            review_id="r1",
            observed_at=(
                "2026-09-15T00:00:00+00:00"
            ),
            subject_type="trajectory",
            subject_id="trajectory-1",
            decision="approve",
            source="trusted_review",
            reason="Initially approved.",
        ),

        ReviewDecision(
            review_id="r2",
            observed_at=(
                "2026-09-15T00:00:01+00:00"
            ),
            subject_type="trajectory",
            subject_id="trajectory-1",
            decision="reject",
            source="trusted_review",
            reason="Later rejected.",
        ),
    ]

    index = (
        latest_review_index(
            reviews
        )
    )

    assert (
        subject_is_approved(
            review_index=(
                index
            ),

            subject_type=(
                "trajectory"
            ),

            subject_id=(
                "trajectory-1"
            ),
        )
        is False
    )


def test_correction_can_be_approved_separately(
):

    reviews = [
        ReviewDecision(
            review_id="r1",
            observed_at=(
                "2026-09-15T00:00:00+00:00"
            ),
            subject_type="trajectory",
            subject_id="trajectory-1",
            decision="approve",
            source="trusted_review",
            reason="Trajectory verified.",
        ),

        ReviewDecision(
            review_id="r2",
            observed_at=(
                "2026-09-15T00:00:01+00:00"
            ),
            subject_type="correction",
            subject_id="correction-1",
            decision="approve",
            source="trusted_review",
            reason="Correction verified.",
        ),
    ]

    index = (
        latest_review_index(
            reviews
        )
    )

    assert (
        subject_is_approved(
            review_index=(
                index
            ),

            subject_type=(
                "trajectory"
            ),

            subject_id=(
                "trajectory-1"
            ),
        )
        is True
    )

    assert (
        subject_is_approved(
            review_index=(
                index
            ),

            subject_type=(
                "correction"
            ),

            subject_id=(
                "correction-1"
            ),
        )
        is True
    )


def test_duplicate_review_id_is_rejected(
    tmp_path: Path,
):

    path = (
        tmp_path
        / "reviews.jsonl"
    )

    payload = {
        "schema":
            "review-decision.v1",

        "review_id":
            "duplicate",

        "observed_at":
            "2026-09-15T00:00:00+00:00",

        "subject_type":
            "trajectory",

        "subject_id":
            "trajectory-1",

        "decision":
            "approve",

        "source":
            "trusted_review",

        "reason":
            "Verified.",
    }

    path.write_text(
        json.dumps(
            payload
        )
        + "\n"
        + json.dumps(
            payload
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Duplicate review_id"
        ),
    ):

        load_reviews(
            path
        )
    