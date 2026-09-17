from __future__ import annotations

import argparse
import json
import sys

from pathlib import (
    Path,
)

from learning.evaluation.eval_reports import (
    EvaluationReportStore,
    compare_evaluation_reports,
)

from learning.paths import (
    EVALUATIONS_ROOT,
)


DEFAULT_ROOT = (
    EVALUATIONS_ROOT
)


# ============================================================
# CLI
# ============================================================


def build_parser(
) -> argparse.ArgumentParser:

    parser = (
        argparse.ArgumentParser(
            description=(
                "Inspect and compare persisted "
                "learning evaluation reports."
            )
        )
    )

    parser.add_argument(
        "--root",

        type=Path,

        default=(
            DEFAULT_ROOT
        ),
    )

    subparsers = (
        parser.add_subparsers(
            dest="command",
            required=True,
        )
    )

    list_parser = (
        subparsers.add_parser(
            "list"
        )
    )

    list_parser.add_argument(
        "--suite",
        required=True,
    )

    list_parser.add_argument(
        "--target",
        required=True,
        choices=[
            "orchestrator",
            "tool_gateway",
        ],
    )

    show_parser = (
        subparsers.add_parser(
            "show"
        )
    )

    show_parser.add_argument(
        "--suite",
        required=True,
    )

    show_parser.add_argument(
        "--target",
        required=True,
        choices=[
            "orchestrator",
            "tool_gateway",
        ],
    )

    show_parser.add_argument(
        "--report-id",
        required=True,
    )

    compare_parser = (
        subparsers.add_parser(
            "compare"
        )
    )

    compare_parser.add_argument(
        "--suite",
        required=True,
    )

    compare_parser.add_argument(
        "--target",
        required=True,
        choices=[
            "orchestrator",
            "tool_gateway",
        ],
    )

    compare_parser.add_argument(
        "--baseline",
        required=True,
    )

    compare_parser.add_argument(
        "--candidate",
        required=True,
    )

    return parser


# ============================================================
# COMMANDS
# ============================================================


def command_list(
    store: EvaluationReportStore,
    *,
    suite: str,
    target: str,
) -> int:

    reports = (
        store.list_reports(
            suite=(
                suite
            ),

            target=(
                target
            ),
        )
    )

    if not reports:

        print(
            "No evaluation reports."
        )

        return 0

    for artifact in reports:

        report = (
            artifact.report
        )

        print(
            f"{artifact.report_id} "
            f"label={artifact.label!r} "
            f"model={artifact.model_key} "
            f"cases="
            f"{report.passed_cases}/"
            f"{report.case_count} "
            f"pass_rate="
            f"{report.pass_rate * 100:.1f}% "
            f"sha256="
            f"{artifact.report_sha256}"
        )

    return 0


def command_show(
    store: EvaluationReportStore,
    *,
    suite: str,
    target: str,
    report_id: str,
) -> int:

    artifact = (
        store.load(
            suite=(
                suite
            ),

            target=(
                target
            ),

            report_id=(
                report_id
            ),
        )
    )

    print(
        artifact.model_dump_json(
            by_alias=True,
            indent=2,
        )
    )

    return 0


def command_compare(
    store: EvaluationReportStore,
    *,
    suite: str,
    target: str,
    baseline_id: str,
    candidate_id: str,
) -> int:

    baseline = (
        store.load(
            suite=(
                suite
            ),

            target=(
                target
            ),

            report_id=(
                baseline_id
            ),
        )
    )

    candidate = (
        store.load(
            suite=(
                suite
            ),

            target=(
                target
            ),

            report_id=(
                candidate_id
            ),
        )
    )

    comparison = (
        compare_evaluation_reports(
            baseline=(
                baseline
            ),

            candidate=(
                candidate
            ),
        )
    )

    print(
        comparison.model_dump_json(
            by_alias=True,
            indent=2,
        )
    )

    return (
        0
        if comparison.promotion_eligible
        else 1
    )


# ============================================================
# ENTRY POINT
# ============================================================


def main(
) -> int:

    parser = (
        build_parser()
    )

    args = (
        parser.parse_args()
    )

    store = (
        EvaluationReportStore(
            root=(
                args.root
            )
        )
    )

    try:

        if (
            args.command
            == "list"
        ):

            return (
                command_list(
                    store,

                    suite=(
                        args.suite
                    ),

                    target=(
                        args.target
                    ),
                )
            )

        if (
            args.command
            == "show"
        ):

            return (
                command_show(
                    store,

                    suite=(
                        args.suite
                    ),

                    target=(
                        args.target
                    ),

                    report_id=(
                        args.report_id
                    ),
                )
            )

        if (
            args.command
            == "compare"
        ):

            return (
                command_compare(
                    store,

                    suite=(
                        args.suite
                    ),

                    target=(
                        args.target
                    ),

                    baseline_id=(
                        args.baseline
                    ),

                    candidate_id=(
                        args.candidate
                    ),
                )
            )

    except Exception as exc:

        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )

        return 1

    return 2


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
