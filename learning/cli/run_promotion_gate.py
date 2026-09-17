from __future__ import annotations

import argparse
import sys

from pathlib import (
    Path,
)

from learning.evaluation.eval_reports import (
    EvaluationReportStore,
)

from learning.curation.promotion_gate import (
    PromotionGateStore,
    build_model_promotion_decision,
)

from learning.paths import (
    EVALUATIONS_ROOT,
    PROMOTIONS_ROOT,
)


DEFAULT_REPORT_ROOT = (
    EVALUATIONS_ROOT
)


DEFAULT_PROMOTION_ROOT = (
    PROMOTIONS_ROOT
)


# ============================================================
# CLI
# ============================================================


def build_parser(
) -> argparse.ArgumentParser:

    parser = (
        argparse.ArgumentParser(
            description=(
                "Evaluate a model candidate "
                "against persisted intelligence "
                "and safety baselines."
            )
        )
    )

    parser.add_argument(
        "--suite",

        default=(
            "core.v1"
        ),
    )

    parser.add_argument(
        "--baseline-intelligence",

        required=True,
    )

    parser.add_argument(
        "--candidate-intelligence",

        required=True,
    )

    parser.add_argument(
        "--baseline-safety",

        required=True,
    )

    parser.add_argument(
        "--candidate-safety",

        required=True,
    )

    parser.add_argument(
        "--label",

        default=(
            "candidate-promotion-gate"
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
        "--promotion-root",

        type=Path,

        default=(
            DEFAULT_PROMOTION_ROOT
        ),
    )

    parser.add_argument(
        "--json",

        action="store_true",
    )

    return parser


# ============================================================
# MAIN
# ============================================================


def main(
) -> int:

    parser = (
        build_parser()
    )

    args = (
        parser.parse_args()
    )

    report_store = (
        EvaluationReportStore(
            root=(
                args.report_root
            )
        )
    )

    promotion_store = (
        PromotionGateStore(
            root=(
                args.promotion_root
            )
        )
    )

    try:

        # ====================================================
        # LOAD INTELLIGENCE REPORTS
        # ====================================================

        baseline_intelligence = (
            report_store.load(
                suite=(
                    args.suite
                ),

                target=(
                    "orchestrator"
                ),

                report_id=(
                    args.baseline_intelligence
                ),
            )
        )

        candidate_intelligence = (
            report_store.load(
                suite=(
                    args.suite
                ),

                target=(
                    "orchestrator"
                ),

                report_id=(
                    args.candidate_intelligence
                ),
            )
        )

        # ====================================================
        # LOAD SAFETY REPORTS
        # ====================================================

        baseline_safety = (
            report_store.load(
                suite=(
                    args.suite
                ),

                target=(
                    "tool_gateway"
                ),

                report_id=(
                    args.baseline_safety
                ),
            )
        )

        candidate_safety = (
            report_store.load(
                suite=(
                    args.suite
                ),

                target=(
                    "tool_gateway"
                ),

                report_id=(
                    args.candidate_safety
                ),
            )
        )

        # ====================================================
        # BUILD DECISION
        # ====================================================

        decision = (
            build_model_promotion_decision(
                baseline_intelligence=(
                    baseline_intelligence
                ),

                candidate_intelligence=(
                    candidate_intelligence
                ),

                baseline_safety=(
                    baseline_safety
                ),

                candidate_safety=(
                    candidate_safety
                ),

                label=(
                    args.label
                ),
            )
        )

        artifact = (
            promotion_store.save(
                decision
            )
        )

    except Exception as exc:

        print(
            "ERROR: "
            f"{exc}",
            file=sys.stderr,
        )

        return 1

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "Agentic AI Model Promotion Gate"
    )

    print(
        "==============================="
    )

    print(
        "Suite: "
        f"{decision.suite}"
    )

    print(
        "Baseline model: "
        f"{decision.baseline_model_key}"
    )

    print(
        "Candidate model: "
        f"{decision.candidate_model_key}"
    )

    # ========================================================
    # INTELLIGENCE
    # ========================================================

    intelligence = (
        decision.intelligence_comparison
    )

    print(
        "\n"
        "Intelligence"
    )

    print(
        "------------"
    )

    print(
        "Baseline pass rate: "
        f"{intelligence.baseline_pass_rate * 100:.1f}%"
    )

    print(
        "Candidate pass rate: "
        f"{intelligence.candidate_pass_rate * 100:.1f}%"
    )

    print(
        "Pass-rate delta:     "
        f"{intelligence.pass_rate_delta * 100:+.1f}%"
    )

    print(
        "No regression:       "
        f"{intelligence.promotion_eligible}"
    )

    # ========================================================
    # SAFETY
    # ========================================================

    safety = (
        decision.safety_comparison
    )

    print(
        "\n"
        "Safety"
    )

    print(
        "------"
    )

    print(
        "Baseline pass rate: "
        f"{safety.baseline_pass_rate * 100:.1f}%"
    )

    print(
        "Candidate pass rate: "
        f"{safety.candidate_pass_rate * 100:.1f}%"
    )

    print(
        "Pass-rate delta:     "
        f"{safety.pass_rate_delta * 100:+.1f}%"
    )

    print(
        "No regression:       "
        f"{safety.promotion_eligible}"
    )

    # ========================================================
    # CHECKS
    # ========================================================

    print(
        "\n"
        "Gate Checks"
    )

    print(
        "-----------"
    )

    for check in (
        decision.checks
    ):

        state = (
            "PASS"
            if check.passed
            else "FAIL"
        )

        print(
            f"{state:<5} "
            f"{check.name}"
        )

    # ========================================================
    # DECISION
    # ========================================================

    print(
        "\n"
        "Decision"
    )

    print(
        "--------"
    )

    if (
        decision.promotion_eligible
    ):

        print(
            "PROMOTE"
        )

    else:

        print(
            "REJECT"
        )

        print(
            "Reasons:"
        )

        for reason in (
            decision.rejection_reasons
        ):

            print(
                "  - "
                f"{reason}"
            )

    print(
        "Improvement observed: "
        f"{decision.improvement_observed}"
    )

    # ========================================================
    # ARTIFACT
    # ========================================================

    print(
        "\n"
        "Persisted Decision"
    )

    print(
        "------------------"
    )

    print(
        "Decision ID: "
        f"{decision.decision_id}"
    )

    print(
        "SHA-256:    "
        f"{artifact.decision_sha256}"
    )

    if args.json:

        print(
            "\n"
            "JSON"
        )

        print(
            "----"
        )

        print(
            artifact.model_dump_json(
                by_alias=True,
                indent=2,
            )
        )

    return (
        0
        if decision.promotion_eligible
        else 1
    )


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
