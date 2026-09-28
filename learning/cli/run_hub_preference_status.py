from __future__ import annotations

import argparse

from learning.continual.storage import load_jsonl_models
from learning.evidence.hub_routing import (
    DEFAULT_HUB_ROUTING_LEDGER,
    HubRoutingLedgerRecord,
)
from learning.training.hub_preferences import (
    HUB_CORRECTIONS_PATH,
    HUB_CORRECTION_REVIEWS_PATH,
    HUB_PREFERENCE_DATASET_ROOT,
    latest_hub_correction_review_index,
    load_hub_corrections,
    load_hub_correction_reviews,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect Hub preference-learning readiness without training."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.limit < 1:
        raise SystemExit(
            "--limit must be positive"
        )

    attempts = load_jsonl_models(
        DEFAULT_HUB_ROUTING_LEDGER,
        HubRoutingLedgerRecord,
    )
    corrections = load_hub_corrections(
        HUB_CORRECTIONS_PATH
    )
    reviews = load_hub_correction_reviews(
        HUB_CORRECTION_REVIEWS_PATH
    )
    latest_reviews = latest_hub_correction_review_index(
        reviews
    )

    approved = sum(
        1
        for correction in corrections
        if (
            correction.correction_id
            in latest_reviews
            and latest_reviews[
                correction.correction_id
            ].decision
            == "approve"
        )
    )

    versions = (
        sorted(
            child.name
            for child in HUB_PREFERENCE_DATASET_ROOT.glob(
                "v*"
            )
            if child.is_dir()
        )
        if HUB_PREFERENCE_DATASET_ROOT.is_dir()
        else []
    )

    print("Hub Preference Status")
    print("=====================")
    print(
        f"Routing attempts: {len(attempts)}"
    )
    print(
        f"Corrections:      {len(corrections)}"
    )
    print(
        f"Approved corr.:   {approved}"
    )
    print(
        f"Dataset versions: {len(versions)}"
    )
    print()

    for record in attempts[-args.limit:]:
        attempt = record.attempt

        print(
            f"{attempt.attempt_id} "
            f"trajectory={record.trajectory_id} "
            f"mode={attempt.mode} "
            f"validation={attempt.validation_status} "
            f"model_complete={attempt.model_identity.complete} "
            f"raw_exact={attempt.raw_response_exact} "
            "prompt_exact="
            f"{getattr(attempt, 'messages_exact', False)}"
        )

    print()
    print("Training:        DISABLED")
    print("Activation:      DISABLED")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
