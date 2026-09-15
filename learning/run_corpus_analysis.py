from __future__ import annotations

import argparse

from pathlib import (
    Path,
)

from learning.corpus_analysis import (
    analyze_corpus,
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
                "Analyze captured learning "
                "experience for diversity, "
                "corrections, imbalance, and "
                "held-out evaluation contamination."
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
            "Exit non-zero when exact normalized "
            "held-out evaluation contamination "
            "is detected."
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
            f"{name:<36} "
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
            analyze_corpus(
                trajectory_path=(
                    args.trajectories
                ),

                correction_path=(
                    args.corrections
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
            "Agentic AI Experience Corpus Analysis"
        )

        print(
            "====================================="
        )

        print(
            f"Trajectories:           "
            f"{report.trajectory_count}"
        )

        print(
            f"Corrections:            "
            f"{report.correction_count}"
        )

        print(
            f"Corrected trajectories: "
            f"{report.corrected_trajectory_count}"
        )

        print(
            f"Orphan corrections:     "
            f"{report.orphan_correction_count}"
        )

        print(
            f"Unique requests:        "
            f"{report.unique_request_count}"
        )

        print(
            f"Duplicate requests:     "
            f"{report.duplicate_request_count}"
        )

        print(
            f"Duplicate rate:         "
            f"{report.duplicate_request_rate * 100:.1f}%"
        )

        print(
            f"Behavior patterns:      "
            f"{report.unique_behavior_pattern_count}"
        )

        print(
            f"Held-out eval cases:    "
            f"{report.held_out_eval_case_count}"
        )

        print(
            f"Eval contamination:     "
            f"{report.held_out_contamination_count}"
        )

        _print_counts(
            title=(
                "Domains"
            ),

            values=(
                report.domain_counts
            ),
        )

        _print_counts(
            title=(
                "Capabilities"
            ),

            values=(
                report.tool_counts
            ),
        )

        _print_counts(
            title=(
                "Outcomes"
            ),

            values=(
                report.outcome_counts
            ),
        )

        _print_counts(
            title=(
                "Correction Types"
            ),

            values=(
                report.correction_type_counts
            ),
        )

        _print_counts(
            title=(
                "Correction Sources"
            ),

            values=(
                report.correction_source_counts
            ),
        )

        _print_counts(
            title=(
                "Review States"
            ),

            values=(
                report
                .quality_review_state_counts
            ),
        )

        _print_counts(
            title=(
                "Failure Types"
            ),

            values=(
                report.failure_type_counts
            ),
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

                cases = (
                    ", ".join(
                        match.eval_cases
                    )
                )

                print(
                    "  "
                    f"{match.trajectory_id}: "
                    f"{cases}"
                )

        print(
            "\n"
            "Warnings"
        )

        print(
            "--------"
        )

        if not report.warnings:

            print(
                "(none)"
            )

        else:

            for warning in (
                report.warnings
            ):

                print(
                    f"[{warning.severity.upper()}] "
                    f"{warning.code}: "
                    f"{warning.message}"
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