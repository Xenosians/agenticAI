from __future__ import annotations

import argparse

from pathlib import (
    Path,
)

from learning.reviewed_dataset import (
    ReviewedPreferenceDatasetBuilder,
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


DEFAULT_DATASETS = (
    PROJECT_ROOT
    / ".runtime"
    / "learning"
    / "datasets"
)


DEFAULT_EVAL_DIRECTORY = (
    PROJECT_ROOT
    / "learning"
    / "evals"
)


def build_parser(
) -> argparse.ArgumentParser:

    parser = (
        argparse.ArgumentParser(
            description=(
                "Promote trusted reviewed correction evidence "
                "into an immutable preference dataset version."
            )
        )
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

    parser.add_argument(
        "--datasets",
        type=Path,
        default=(
            DEFAULT_DATASETS
        ),
    )

    parser.add_argument(
        "--eval",
        dest="eval_paths",
        type=Path,
        action="append",
        default=None,
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

    return parser


def main(
) -> int:

    args = (
        build_parser()
        .parse_args()
    )

    if (
        args.eval_paths
        is None
    ):

        eval_paths = (
            sorted(
                DEFAULT_EVAL_DIRECTORY
                .glob(
                    "*.jsonl"
                )
            )
        )

    else:

        eval_paths = (
            args.eval_paths
        )

    try:

        builder = (
            ReviewedPreferenceDatasetBuilder(
                trajectory_path=(
                    args.trajectories
                ),

                correction_path=(
                    args.corrections
                ),

                review_path=(
                    args.reviews
                ),

                dataset_root=(
                    args.datasets
                ),
            )
        )

        result = (
            builder.build(
                eval_paths=(
                    eval_paths
                ),

                promoted_by=(
                    args.source
                ),

                promotion_reason=(
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
        "Reviewed Preference Dataset"
    )

    print(
        "==========================="
    )

    print(
        f"Version:             "
        f"{result.manifest.version}"
    )

    print(
        f"Preference examples: "
        f"{result.preference_example_count}"
    )

    print(
        f"Trajectories:        "
        f"{len(set(result.source_trajectory_ids))}"
    )

    print(
        f"Corrections:         "
        f"{len(set(result.source_correction_ids))}"
    )

    print(
        f"SHA-256:             "
        f"{result.manifest.content_sha256}"
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )