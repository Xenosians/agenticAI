from __future__ import annotations

import argparse
from pathlib import Path

from learning.training.hub_preferences import (
    HUB_CORRECTIONS_PATH,
    HUB_CORRECTION_REVIEWS_PATH,
    HubCorrectionReviewRecorder,
    load_hub_corrections,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Append a trusted approval/rejection decision for one Hub "
            "routing correction."
        )
    )

    parser.add_argument("correction_id")
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
        default="trusted_review",
    )
    parser.add_argument(
        "--corrections",
        type=Path,
        default=HUB_CORRECTIONS_PATH,
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=HUB_CORRECTION_REVIEWS_PATH,
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    corrections = load_hub_corrections(
        args.corrections
    )

    if args.correction_id not in {
        item.correction_id
        for item in corrections
    }:
        print(
            "ERROR: Hub correction does not exist: "
            f"{args.correction_id}"
        )
        return 1

    try:
        recorder = HubCorrectionReviewRecorder(
            path=args.reviews,
            enabled=True,
        )

        review = recorder.record(
            correction_id=args.correction_id,
            decision=args.decision,
            source=args.source,
            reason=args.reason,
        )

        if review is None:
            raise RuntimeError(
                "Hub correction review recorder unexpectedly disabled."
            )

    except Exception as exc:
        print(
            f"ERROR: {exc}"
        )
        return 1

    print("Hub Correction Review")
    print("=====================")
    print(
        f"Review:       {review.review_id}"
    )
    print(
        f"Correction:   {review.correction_id}"
    )
    print(
        f"Decision:     {review.decision}"
    )
    print(
        f"Source:       {review.source}"
    )
    print("Training:     DISABLED")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
