from __future__ import annotations

import argparse
from pathlib import Path

from learning.paths import EVALUATION_SUITE_ROOT
from learning.training.hub_preferences import (
    HUB_PREFERENCE_DATASET_ROOT,
    HubPreferenceDatasetBuilder,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Promote trusted reviewed Hub routing corrections into an "
            "immutable provenance-locked Hub preference dataset."
        )
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
        "--dataset-root",
        type=Path,
        default=HUB_PREFERENCE_DATASET_ROOT,
    )
    parser.add_argument(
        "--eval",
        dest="eval_paths",
        type=Path,
        action="append",
        default=None,
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    eval_paths = (
        sorted(
            EVALUATION_SUITE_ROOT.glob(
                "*.jsonl"
            )
        )
        if args.eval_paths is None
        else args.eval_paths
    )

    try:
        builder = HubPreferenceDatasetBuilder(
            dataset_root=args.dataset_root,
        )

        result = builder.build(
            eval_paths=eval_paths,
            promoted_by=args.source,
            promotion_reason=args.reason,
        )

    except Exception as exc:
        print(
            f"ERROR: {exc}"
        )
        return 1

    manifest = result.manifest

    print("Hub Preference Dataset")
    print("======================")
    print(
        f"Version:       {manifest.version}"
    )
    print(
        f"Records:       {manifest.record_count}"
    )
    print(
        f"SHA-256:       {manifest.content_sha256}"
    )
    print(
        f"Output:        {result.output_directory}"
    )
    print("Training:      DISABLED")
    print("Promotion:     DATASET ONLY")
    print("Activation:    DISABLED")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
