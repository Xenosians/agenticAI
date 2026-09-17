from __future__ import annotations

import argparse
import asyncio
import sys

from pathlib import (
    Path,
)

from learning.evaluation.eval_reports import (
    EvaluationReportStore,
)

from learning.evaluation.eval_suite import (
    load_evaluation_cases,
)

from learning.evaluation.gateway_evaluation import (
    GatewayEvaluationRunner,
)

from learning.paths import (
    EVALUATIONS_ROOT,
    EVALUATION_SUITE_ROOT,
)


DEFAULT_SUITE = (
    EVALUATION_SUITE_ROOT
    / "core.v1.jsonl"
)


DEFAULT_REPORT_ROOT = (
    EVALUATIONS_ROOT
)


def build_parser(
) -> argparse.ArgumentParser:

    parser = (
        argparse.ArgumentParser(
            description=(
                "Run deterministic "
                "ToolGateway safety "
                "evaluation cases."
            )
        )
    )

    parser.add_argument(
        "--suite",

        type=Path,

        default=(
            DEFAULT_SUITE
        ),
    )

    parser.add_argument(
        "--case",

        action="append",

        dest="case_ids",

        default=None,
    )

    parser.add_argument(
        "--label",

        default=(
            "gateway-baseline"
        ),
    )

    parser.add_argument(
        "--report-root",

        type=Path,

        default=(
            DEFAULT_REPORT_ROOT
        ),
    )

    parser.add_argument(
        "--json",

        action="store_true",
    )

    return parser


async def run(
    args,
) -> int:

    cases = (
        load_evaluation_cases(
            args.suite
        )
    )

    gateway_cases = [
        case

        for case
        in cases

        if (
            case.target
            == "tool_gateway"
        )
    ]

    if args.case_ids:

        requested_ids = (
            set(
                args.case_ids
            )
        )

        known_ids = {
            case.case_id

            for case
            in cases
        }

        unknown_ids = (
            requested_ids
            - known_ids
        )

        if unknown_ids:

            raise ValueError(
                "Unknown evaluation case(s): "
                + ", ".join(
                    sorted(
                        unknown_ids
                    )
                )
            )

        wrong_target = [
            case.case_id

            for case
            in cases

            if (
                case.case_id
                in requested_ids
                and case.target
                != "tool_gateway"
            )
        ]

        if wrong_target:

            raise ValueError(
                "This runner only supports "
                "target='tool_gateway'."
            )

        gateway_cases = [
            case

            for case
            in gateway_cases

            if (
                case.case_id
                in requested_ids
            )
        ]

    if not gateway_cases:

        raise ValueError(
            "No ToolGateway evaluation "
            "cases were selected."
        )

    print(
        "Agentic AI ToolGateway "
        "Safety Evaluation"
    )

    print(
        "================================"
    )

    print(
        "Suite: "
        f"{gateway_cases[0].suite}"
    )

    print(
        "Cases: "
        f"{len(gateway_cases)}"
    )

    print(
        "Model inference: disabled"
    )

    print(
        "Real MCP execution: blocked"
    )

    runner = (
        GatewayEvaluationRunner()
    )

    report = (
        await runner.run(
            gateway_cases
        )
    )

    store = (
        EvaluationReportStore(
            root=(
                args.report_root
            )
        )
    )

    artifact = (
        store.save(
            report=(
                report
            ),

            label=(
                args.label
            ),
        )
    )

    print(
        "\n"
        "Safety Evaluation Summary"
    )

    print(
        "========================="
    )

    print(
        "Cases:       "
        f"{report.passed_cases}/"
        f"{report.case_count}"
    )

    print(
        "Pass rate:   "
        f"{report.pass_rate * 100:.1f}%"
    )

    print(
        "Mean score:  "
        f"{report.mean_score * 100:.1f}%"
    )

    print(
        "Duration:    "
        f"{report.total_duration_seconds:.4f}s"
    )

    metric = (
        report.outcome_metric
    )

    print(
        "\n"
        "Gateway Decision Metric"
    )

    print(
        "-----------------------"
    )

    if (
        metric.accuracy
        is None
    ):

        rendered = (
            "n/a"
        )

    else:

        rendered = (
            f"{metric.passed}/"
            f"{metric.checked} "
            f"({metric.accuracy * 100:.1f}%)"
        )

    print(
        "Decision     "
        f"{rendered}"
    )

    print(
        "\n"
        "Persisted Report"
    )

    print(
        "----------------"
    )

    print(
        "Report ID: "
        f"{artifact.report_id}"
    )

    print(
        "SHA-256:   "
        f"{artifact.report_sha256}"
    )

    if args.json:

        print(
            "\n"
            "JSON Report"
        )

        print(
            "-----------"
        )

        print(
            artifact.model_dump_json(
                by_alias=True,
                indent=2,
            )
        )

    return (
        0
        if report.failed_cases
        == 0
        else 1
    )


def main(
) -> int:

    parser = (
        build_parser()
    )

    args = (
        parser.parse_args()
    )

    try:

        return (
            asyncio.run(
                run(
                    args
                )
            )
        )

    except KeyboardInterrupt:

        return 130

    except Exception as exc:

        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )

        return 1


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
