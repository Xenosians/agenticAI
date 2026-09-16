from __future__ import annotations

import argparse

from pathlib import (
    Path,
)

from learning.curation.corpus_analysis import (
    load_corrections,
    load_trajectories,
)

from learning.curation.reviews import (
    ReviewDecision,
    ReviewDecisionRecorder,
)


PROJECT_ROOT = (
    Path(
        __file__
    )
    .resolve()
    .parents[
        1
    ]
)


DEFAULT_TRAJECTORIES = (
    PROJECT_ROOT
    / ".runtime"
    / "learning"
    / "trajectories.jsonl"
)


DEFAULT_CORRECTIONS = (
    PROJECT_ROOT
    / ".runtime"
    / "learning"
    / "corrections.jsonl"
)


DEFAULT_REVIEWS = (
    PROJECT_ROOT
    / ".runtime"
    / "learning"
    / "reviews.jsonl"
)


def review_subject(
    *,
    trajectory_path: Path,
    correction_path: Path,
    review_path: Path,
    subject_type: str,
    subject_id: str,
    decision: str,
    source: str,
    reason: str,
) -> ReviewDecision:

    trajectories = (
        load_trajectories(
            trajectory_path
        )
    )

    corrections = (
        load_corrections(
            correction_path
        )
    )

    trajectory_ids = {
        trajectory.trajectory_id

        for trajectory
        in trajectories
    }

    correction_by_id = {
        correction.correction_id:
            correction

        for correction
        in corrections
    }

    normalized_subject_type = (
        subject_type
        .strip()
        .lower()
    )

    normalized_subject_id = (
        subject_id
        .strip()
    )

    if (
        normalized_subject_type
        == "trajectory"
    ):

        if (
            normalized_subject_id
            not in trajectory_ids
        ):

            raise ValueError(
                "Trajectory does not exist in "
                "the raw corpus: "
                f"{normalized_subject_id}"
            )

    elif (
        normalized_subject_type
        == "correction"
    ):

        correction = (
            correction_by_id
            .get(
                normalized_subject_id
            )
        )

        if correction is None:

            raise ValueError(
                "Correction does not exist in "
                "the raw corpus: "
                f"{normalized_subject_id}"
            )

        if (
            correction.trajectory_id
            not in trajectory_ids
        ):

            raise ValueError(
                "Correction is orphaned and cannot "
                "be approved for training evidence: "
                f"{normalized_subject_id}"
            )

    else:

        raise ValueError(
            "subject_type must be trajectory "
            "or correction."
        )

    recorder = (
        ReviewDecisionRecorder(
            path=(
                review_path
            ),

            enabled=True,
        )
    )

    payload = (
        recorder.record(
            subject_type=(
                normalized_subject_type
            ),

            subject_id=(
                normalized_subject_id
            ),

            decision=(
                decision
            ),

            source=(
                source
            ),

            reason=(
                reason
            ),
        )
    )

    if payload is None:

        raise RuntimeError(
            "Review recorder unexpectedly disabled."
        )

    return (
        ReviewDecision
        .model_validate(
            payload
        )
    )


def build_parser(
) -> argparse.ArgumentParser:

    parser = (
        argparse.ArgumentParser(
            description=(
                "Append a trusted approval or rejection "
                "decision for immutable learning evidence."
            )
        )
    )

    parser.add_argument(
        "subject_type",
        choices=[
            "trajectory",
            "correction",
        ],
    )

    parser.add_argument(
        "subject_id",
    )

    parser.add_argument(
        "decision",
        choices=[
            "approve",
            "reject",
        ],
    )

    parser.add_argument(
        "--reason",
        required=True,
    )

    parser.add_argument(
        "--source",
        choices=[
            "trusted_review",
            "evaluation",
        ],
        default=(
            "trusted_review"
        ),
    )

    parser.add_argument(
        "--trajectories",
        type=Path,
        default=(
            DEFAULT_TRAJECTORIES
        ),
    )

    parser.add_argument(
        "--corrections",
        type=Path,
        default=(
            DEFAULT_CORRECTIONS
        ),
    )

    parser.add_argument(
        "--reviews",
        type=Path,
        default=(
            DEFAULT_REVIEWS
        ),
    )

    return parser


def main(
) -> int:

    args = (
        build_parser()
        .parse_args()
    )

    try:

        review = (
            review_subject(
                trajectory_path=(
                    args.trajectories
                ),

                correction_path=(
                    args.corrections
                ),

                review_path=(
                    args.reviews
                ),

                subject_type=(
                    args.subject_type
                ),

                subject_id=(
                    args.subject_id
                ),

                decision=(
                    args.decision
                ),

                source=(
                    args.source
                ),

                reason=(
                    args.reason
                ),
            )
        )

    except Exception as exc:

        print(
            "ERROR: "
            f"{exc}"
        )

        return 1

    print(
        "Learning Review Recorded"
    )

    print(
        "========================"
    )

    print(
        f"Review ID:    "
        f"{review.review_id}"
    )

    print(
        f"Subject:      "
        f"{review.subject_type}:"
        f"{review.subject_id}"
    )

    print(
        f"Decision:     "
        f"{review.decision}"
    )

    print(
        f"Source:       "
        f"{review.source}"
    )

    print(
        f"Reason:       "
        f"{review.reason}"
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )