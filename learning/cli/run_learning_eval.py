from __future__ import annotations

import argparse
import asyncio
import sys
import uuid

from pathlib import (
    Path,
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


if (
    str(
        PROJECT_ROOT
    )
    not in sys.path
):
    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
    )


from agent.mcp_client import (
    MCPRuntime,
)

from config import (
    Settings,
)

from learning.evaluation.eval_reports import (
    EvaluationReportStore,
)

from learning.evaluation.eval_suite import (
    load_evaluation_cases,
)

from learning.evaluation.live_evaluation import (
    LiveOrchestratorEvaluationRunner,
)

from subagents.core.tooling.gateway import (
    ToolGateway,
)

from subagents.hub import (
    build_hub,
)

from subagents.llm.runtime.inference import (
    InferenceCoordinator,
)

from subagents.llm.runtime.model_manager import (
    ModelManager,
)

from subagents.llm.runtime.scheduler import (
    GpuScheduler,
)


# ============================================================
# DEFAULT PATHS
# ============================================================


DEFAULT_SUITE = (
    PROJECT_ROOT
    / "learning"
    / "evals"
    / "core.v1.jsonl"
)


DEFAULT_REPORT_ROOT = (
    PROJECT_ROOT
    / ".runtime"
    / "learning"
    / "evaluations"
)


# ============================================================
# EVALUATION-SAFE APPROVAL PLACEHOLDER
# ============================================================


def create_eval_approval(
    tool_name: str,
    arguments: dict,
    *,
    risk: str,
) -> dict:
    """
    Evaluation must never create production approval state.

    If a model proposes an approval-required capability, the
    ToolGateway may still return approval_required, but the
    approval object remains synthetic and isolated.
    """

    return {
        "id":
            (
                "eval-approval-"
                f"{uuid.uuid4().hex}"
            ),

        "tool":
            tool_name,

        "arguments":
            arguments,

        "risk":
            risk,
    }


# ============================================================
# CLI
# ============================================================


def build_parser(
) -> argparse.ArgumentParser:

    parser = (
        argparse.ArgumentParser(
            description=(
                "Run live Main -> Specialist -> "
                "ToolGateway learning evaluation."
            )
        )
    )

    parser.add_argument(
        "--suite",

        type=Path,

        default=(
            DEFAULT_SUITE
        ),

        help=(
            "Evaluation suite JSONL."
        ),
    )

    parser.add_argument(
        "--case",

        action="append",

        dest="case_ids",

        default=None,

        help=(
            "Run only this case ID. "
            "May be supplied multiple times."
        ),
    )

    parser.add_argument(
        "--label",

        default=(
            "orchestrator-baseline"
        ),

        help=(
            "Immutable report label."
        ),
    )

    parser.add_argument(
        "--report-root",

        type=Path,

        default=(
            DEFAULT_REPORT_ROOT
        ),

        help=(
            "Evaluation report storage root."
        ),
    )

    parser.add_argument(
        "--json",

        action="store_true",

        help=(
            "Print the persisted report "
            "artifact as JSON."
        ),
    )

    return parser


# ============================================================
# CASE SELECTION
# ============================================================


def select_orchestrator_cases(
    *,
    cases,
    case_ids: list[
        str
    ] | None,
):
    orchestrator_cases = [
        case

        for case
        in cases

        if (
            case.target
            == "orchestrator"
        )
    ]

    skipped_cases = [
        case

        for case
        in cases

        if (
            case.target
            != "orchestrator"
        )
    ]

    if not case_ids:
        return (
            orchestrator_cases,
            skipped_cases,
        )

    requested_ids = (
        set(
            case_ids
        )
    )

    known_ids = {
        case.case_id

        for case
        in cases
    }

    missing = (
        requested_ids
        - known_ids
    )

    if missing:
        raise ValueError(
            "Unknown evaluation case(s): "
            + ", ".join(
                sorted(
                    missing
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
            != "orchestrator"
        )
    ]

    if wrong_target:
        raise ValueError(
            "This runner only supports "
            "target='orchestrator'. "
            "Unsupported selected case(s): "
            + ", ".join(
                wrong_target
            )
        )

    orchestrator_cases = [
        case

        for case
        in orchestrator_cases

        if (
            case.case_id
            in requested_ids
        )
    ]

    return (
        orchestrator_cases,
        skipped_cases,
    )


# ============================================================
# MAIN ASYNC RUN
# ============================================================


async def run(
    args,
) -> int:

    settings = (
        Settings()
    )

    all_cases = (
        load_evaluation_cases(
            args.suite
        )
    )

    (
        orchestrator_cases,
        skipped_cases,
    ) = (
        select_orchestrator_cases(
            cases=(
                all_cases
            ),

            case_ids=(
                args.case_ids
            ),
        )
    )

    if not orchestrator_cases:
        raise ValueError(
            "No orchestrator evaluation "
            "cases were selected."
        )

    # ========================================================
    # HEADER
    # ========================================================

    print(
        "Agentic AI Live Evaluation"
    )

    print(
        "=========================="
    )

    print(
        "Suite: "
        f"{orchestrator_cases[0].suite}"
    )

    print(
        "Model: "
        f"{settings.hub_model_key}"
    )

    print(
        "Cases: "
        f"{len(orchestrator_cases)}"
    )

    if (
        skipped_cases
        and not args.case_ids
    ):
        print(
            "Skipped non-orchestrator cases: "
            f"{len(skipped_cases)}"
        )

        for case in (
            skipped_cases
        ):
            print(
                "  - "
                f"{case.case_id} "
                f"({case.target})"
            )

    # ========================================================
    # RUNTIME COMPONENTS
    # ========================================================

    mcp = (
        MCPRuntime()
    )

    model_manager = (
        ModelManager(
            settings=(
                settings
            )
        )
    )

    scheduler = (
        GpuScheduler()
    )

    inference = (
        InferenceCoordinator(
            model_manager=(
                model_manager
            ),

            scheduler=(
                scheduler
            ),
        )
    )

    tool_gateway = (
        ToolGateway(
            approval_creator=(
                create_eval_approval
            ),

            mcp=(
                mcp
            ),
        )
    )

    report = None

    # ========================================================
    # LIVE EXECUTION
    # ========================================================

    try:

        await mcp.start()

        print(
            "\n"
            "[EVAL] Warming hub model..."
        )

        await inference.warm(
            settings.hub_model_key
        )

        hub = (
            build_hub(
                settings=(
                    settings
                ),

                model_manager=(
                    model_manager
                ),

                inference=(
                    inference
                ),

                tool_gateway=(
                    tool_gateway
                ),
            )
        )

        runner = (
            LiveOrchestratorEvaluationRunner(
                hub=(
                    hub
                ),

                hub_model=(
                    settings
                    .hub_model_key
                ),
            )
        )

        report = (
            await runner.run(
                orchestrator_cases
            )
        )

    finally:

        # ----------------------------------------------------
        # MCP
        # ----------------------------------------------------

        try:

            await mcp.stop()

        except Exception as exc:

            print(
                "[EVAL] MCP shutdown failed "
                f"error={exc!r}"
            )

        # ----------------------------------------------------
        # MODELS
        # ----------------------------------------------------

        loaded_models = (
            model_manager
            .list_loaded_models()
        )

        for model_key in (
            loaded_models
        ):

            try:

                model_manager.unload(
                    model_key
                )

            except Exception as exc:

                print(
                    "[EVAL] Model unload failed "
                    f"model='{model_key}' "
                    f"error={exc!r}"
                )

    if report is None:
        raise RuntimeError(
            "Evaluation completed without "
            "producing a report."
        )

    # ========================================================
    # PERSIST IMMUTABLE REPORT
    # ========================================================

    report_store = (
        EvaluationReportStore(
            root=(
                args.report_root
            )
        )
    )

    artifact = (
        report_store.save(
            report=(
                report
            ),

            label=(
                args.label
            ),
        )
    )

    # ========================================================
    # HUMAN SUMMARY
    # ========================================================

    print(
        "\n"
        "Evaluation Summary"
    )

    print(
        "=================="
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
        f"{report.total_duration_seconds:.2f}s"
    )

    # ========================================================
    # SEMANTIC METRICS
    # ========================================================

    print(
        "\n"
        "Semantic Metrics"
    )

    print(
        "----------------"
    )

    metrics = [
        (
            "Routing",
            report.route_metric,
        ),

        (
            "Tool",
            report.tool_metric,
        ),

        (
            "Arguments",
            report.arguments_metric,
        ),

        (
            "Outcome",
            report.outcome_metric,
        ),
    ]

    for (
        label,
        metric,
    ) in metrics:

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
                f"("
                f"{metric.accuracy * 100:.1f}%"
                f")"
            )

        print(
            f"{label:<12}"
            f"{rendered}"
        )

    # ========================================================
    # PERSISTED ARTIFACT
    # ========================================================

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

    # ========================================================
    # OPTIONAL JSON
    # ========================================================

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

    # ========================================================
    # PROCESS RESULT
    # ========================================================

    return (
        0
        if report.failed_cases
        == 0
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

    try:

        return (
            asyncio.run(
                run(
                    args
                )
            )
        )

    except KeyboardInterrupt:

        print(
            "\n"
            "Evaluation interrupted.",
            file=sys.stderr,
        )

        return 130

    except Exception as exc:

        print(
            "ERROR: "
            f"{exc}",
            file=sys.stderr,
        )

        return 1


if __name__ == "__main__":

    raise SystemExit(
        main()
    )