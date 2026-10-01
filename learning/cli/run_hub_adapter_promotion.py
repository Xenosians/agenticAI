from __future__ import annotations

import argparse

from learning.continual.checkpoints import AdapterCheckpointStore
from learning.curation.promotion_gate import (
    PromotionGateStore,
    build_model_promotion_decision,
)
from learning.evaluation.eval_reports import (
    EvaluationReportStore,
)
from learning.paths import (
    EVALUATIONS_ROOT,
    PROMOTIONS_ROOT,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Hub adapter promotion decision from held-out "
            "orchestrator + ToolGateway safety reports, then "
            "optionally activate the registered checkpoint."
        )
    )

    parser.add_argument(
        "--checkpoint-id",
        required=True,
    )
    parser.add_argument(
        "--suite",
        required=True,
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
        required=True,
    )
    parser.add_argument(
        "--activate",
        action="store_true",
    )

    args = parser.parse_args()

    reports = EvaluationReportStore(
        root=EVALUATIONS_ROOT
    )

    baseline_intelligence = reports.load(
        suite=args.suite,
        target="orchestrator",
        report_id=args.baseline_intelligence,
    )
    candidate_intelligence = reports.load(
        suite=args.suite,
        target="orchestrator",
        report_id=args.candidate_intelligence,
    )
    baseline_safety = reports.load(
        suite=args.suite,
        target="tool_gateway",
        report_id=args.baseline_safety,
    )
    candidate_safety = reports.load(
        suite=args.suite,
        target="tool_gateway",
        report_id=args.candidate_safety,
    )

    checkpoint_store = AdapterCheckpointStore()
    checkpoint = checkpoint_store.verify_adapter(
        args.checkpoint_id
    )

    if (
        candidate_intelligence.model_key
        != checkpoint.target_model_key
    ):
        raise SystemExit(
            "Candidate orchestrator report model key does not "
            "match checkpoint target_model_key."
        )

    decision = build_model_promotion_decision(
        baseline_intelligence=(
            baseline_intelligence
        ),
        candidate_intelligence=(
            candidate_intelligence
        ),
        baseline_safety=baseline_safety,
        candidate_safety=candidate_safety,
        label=args.label,
    )

    PromotionGateStore(
        root=PROMOTIONS_ROOT
    ).save(
        decision
    )

    print("Hub Adapter Promotion Gate")
    print("======================")
    print(f"Decision:     {decision.decision_id}")
    print(
        f"Eligible:     {decision.promotion_eligible}"
    )
    print(
        f"Improvement:  {decision.improvement_observed}"
    )
    print(
        "Rejected by:  "
        + (
            ", ".join(
                decision.rejection_reasons
            )
            if decision.rejection_reasons
            else "none"
        )
    )

    if not args.activate:
        print("Activation:   NOT REQUESTED")
        return (
            0
            if decision.promotion_eligible
            else 1
        )

    if not decision.promotion_eligible:
        print("Activation:   REFUSED")
        return 1

    pointer = checkpoint_store.promote(
        checkpoint_id=args.checkpoint_id,
        suite=args.suite,
        decision_id=decision.decision_id,
    )

    print(
        f"Activation:   {pointer.checkpoint_id}"
    )
    print(
        "Restart/unload the Hub model before expecting "
        "the new adapter to be used."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
