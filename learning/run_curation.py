from __future__ import annotations

import argparse

from pathlib import (
    Path,
)

from learning.curation import (
    curate_corpus,
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
                "Curate captured runtime learning evidence "
                "through deterministic quarantine rules and "
                "trusted review decisions."
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
        help=(
            "Append-only trusted review ledger. "
            "Defaults to "
            ".runtime/learning/reviews.jsonl."
        ),
    )

    parser.add_argument(
        "--eval",

        dest=(
            "eval_paths"
        ),

        type=Path,

        action="append",

        default=None,

        help=(
            "Held-out evaluation JSONL. "
            "May be supplied multiple times. "
            "Defaults to all learning/evals/*.jsonl."
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
    )

    parser.add_argument(
        "--fail-on-contamination",
        action="store_true",
        help=(
            "Exit non-zero when held-out "
            "evaluation contamination is detected."
        ),
    )

    return parser


def _print_counts(
    *,
    title: str,
    values: dict[
        str,
        int,
    ],
) -> None:

    print(
        "\n"
        f"{title}"
    )

    print(
        "-" * len(
            title
        )
    )

    if not values:

        print(
            "(none)"
        )

        return

    for (
        name,
        count,
    ) in values.items():

        print(
            f"{name:<40} "
            f"{count}"
        )


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

        report = (
            curate_corpus(
                trajectory_path=(
                    args.trajectories
                ),

                correction_path=(
                    args.corrections
                ),

                review_path=(
                    args.reviews
                ),

                eval_paths=(
                    eval_paths
                ),
            )
        )

    except Exception as exc:

        print(
            "ERROR: "
            f"{exc}"
        )

        return 1

    if args.json:

        print(
            report
            .model_dump_json(
                by_alias=True,
                indent=2,
            )
        )

    else:

        print(
            "Agentic AI Experience Curation"
        )

        print(
            "=============================="
        )

        print(
            f"Candidates:              "
            f"{report.candidate_trajectory_count}"
        )

        print(
            f"Eligible:                "
            f"{report.eligible_trajectory_count}"
        )

        print(
            f"Excluded:                "
            f"{report.excluded_trajectory_count}"
        )

        print(
            f"Corrections:             "
            f"{report.correction_count}"
        )

        print(
            f"Reviews:                 "
            f"{report.review_count}"
        )

        print(
            f"Usable corrections:      "
            f"{report.usable_correction_count}"
        )

        print(
            f"Orphan corrections:      "
            f"{report.orphan_correction_count}"
        )

        print(
            f"Deduplicated:            "
            f"{report.deduplicated_trajectory_count}"
        )

        print(
            f"Eval contamination:      "
            f"{report.held_out_contamination_count}"
        )

        _print_counts(
            title=(
                "Exclusion Reasons"
            ),

            values=(
                report.exclusion_reason_counts
            ),
        )

        print(
            "\n"
            "Eligible Trajectories"
        )

        print(
            "---------------------"
        )

        if not report.eligible:

            print(
                "(none)"
            )

        else:

            for item in (
                report.eligible
            ):

                print(
                    f"{item.trajectory_id} "
                    f"{item.evidence_fingerprint}"
                )

        print(
            "\n"
            "Excluded Trajectories"
        )

        print(
            "---------------------"
        )

        if not report.excluded:

            print(
                "(none)"
            )

        else:

            for item in (
                report.excluded
            ):

                reasons = (
                    ", ".join(
                        item.reasons
                    )
                )

                print(
                    f"{item.trajectory_id}: "
                    f"{reasons}"
                )

                if (
                    item
                    .contamination_references
                ):

                    print(
                        "  held-out: "
                        + ", ".join(
                            item
                            .contamination_references
                        )
                    )

                if (
                    item
                    .duplicate_of_trajectory_id
                    is not None
                ):

                    print(
                        "  duplicate_of: "
                        f"{item.duplicate_of_trajectory_id}"
                    )

        print(
            "\n"
            "Orphan Corrections"
        )

        print(
            "------------------"
        )

        if not (
            report
            .orphan_correction_ids
        ):

            print(
                "(none)"
            )

        else:

            for correction_id in (
                report
                .orphan_correction_ids
            ):

                print(
                    correction_id
                )

        print(
            "\n"
            "Held-out Contamination Guard"
        )

        print(
            "----------------------------"
        )

        if (
            report
            .held_out_contamination_count
            == 0
        ):

            print(
                "PASS"
            )

        else:

            print(
                "FAIL"
            )

            for match in (
                report
                .contamination_matches
            ):

                print(
                    f"{match.trajectory_id}: "
                    + ", ".join(
                        match.eval_cases
                    )
                )

    if (
        args.fail_on_contamination
        and report
        .held_out_contamination_count
        > 0
    ):

        return 2

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )