import json

from pathlib import (
    Path,
)

import pytest

from learning.eval_reports import (
    EvaluationReportStore,
)

from learning.eval_types import (
    EvaluationMetric,
    EvaluationSuiteReport,
)

from learning.promotion_gate import (
    PromotionGateStore,
    build_model_promotion_decision,
)


# ============================================================
# REPORT HELPERS
# ============================================================


def intelligence_report(
    *,
    model_key: str,
    pass_rate: float = 1.0,
    mean_score: float = 1.0,
    arguments_accuracy: float = 1.0,
) -> EvaluationSuiteReport:

    passed_cases = (
        4
        if pass_rate
        == 1.0
        else 3
    )

    return (
        EvaluationSuiteReport(
            suite=(
                "core.v1"
            ),

            target=(
                "orchestrator"
            ),

            model_key=(
                model_key
            ),

            started_at=(
                "2026-09-15T00:00:00+00:00"
            ),

            completed_at=(
                "2026-09-15T00:01:00+00:00"
            ),

            case_count=4,

            passed_cases=(
                passed_cases
            ),

            failed_cases=(
                4
                - passed_cases
            ),

            pass_rate=(
                pass_rate
            ),

            mean_score=(
                mean_score
            ),

            total_duration_seconds=(
                10.0
            ),

            route_metric=(
                EvaluationMetric(
                    checked=4,
                    passed=4,
                    accuracy=1.0,
                )
            ),

            tool_metric=(
                EvaluationMetric(
                    checked=4,
                    passed=4,
                    accuracy=1.0,
                )
            ),

            arguments_metric=(
                EvaluationMetric(
                    checked=4,

                    passed=(
                        4
                        if arguments_accuracy
                        == 1.0
                        else 3
                    ),

                    accuracy=(
                        arguments_accuracy
                    ),
                )
            ),

            outcome_metric=(
                EvaluationMetric(
                    checked=4,
                    passed=4,
                    accuracy=1.0,
                )
            ),

            results=[],
        )
    )


def safety_report(
    *,
    pass_rate: float = 1.0,
) -> EvaluationSuiteReport:

    passed_cases = (
        4
        if pass_rate
        == 1.0
        else 3
    )

    return (
        EvaluationSuiteReport(
            suite=(
                "core.v1"
            ),

            target=(
                "tool_gateway"
            ),

            model_key=(
                "deterministic-tool-gateway"
            ),

            started_at=(
                "2026-09-15T00:00:00+00:00"
            ),

            completed_at=(
                "2026-09-15T00:00:01+00:00"
            ),

            case_count=4,

            passed_cases=(
                passed_cases
            ),

            failed_cases=(
                4
                - passed_cases
            ),

            pass_rate=(
                pass_rate
            ),

            mean_score=(
                pass_rate
            ),

            total_duration_seconds=(
                0.001
            ),

            route_metric=(
                EvaluationMetric()
            ),

            tool_metric=(
                EvaluationMetric()
            ),

            arguments_metric=(
                EvaluationMetric()
            ),

            outcome_metric=(
                EvaluationMetric(
                    checked=4,

                    passed=(
                        passed_cases
                    ),

                    accuracy=(
                        pass_rate
                    ),
                )
            ),

            results=[],
        )
    )


def persisted_reports(
    tmp_path: Path,
):

    store = (
        EvaluationReportStore(
            root=(
                tmp_path
                / "evaluations"
            )
        )
    )

    baseline_intelligence = (
        store.save(
            report=(
                intelligence_report(
                    model_key=(
                        "hub-main"
                    )
                )
            ),

            label=(
                "baseline"
            ),
        )
    )

    candidate_intelligence = (
        store.save(
            report=(
                intelligence_report(
                    model_key=(
                        "candidate-model"
                    )
                )
            ),

            label=(
                "candidate"
            ),
        )
    )

    baseline_safety = (
        store.save(
            report=(
                safety_report()
            ),

            label=(
                "baseline-safety"
            ),
        )
    )

    candidate_safety = (
        store.save(
            report=(
                safety_report()
            ),

            label=(
                "candidate-safety"
            ),
        )
    )

    return (
        store,
        baseline_intelligence,
        candidate_intelligence,
        baseline_safety,
        candidate_safety,
    )


# ============================================================
# PASSING GATE
# ============================================================


def test_equal_candidate_is_promotion_eligible(
    tmp_path: Path,
):

    (
        _,
        baseline_intelligence,
        candidate_intelligence,
        baseline_safety,
        candidate_safety,
    ) = (
        persisted_reports(
            tmp_path
        )
    )

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
                "candidate"
            ),
        )
    )

    assert (
        decision.promotion_eligible
        is True
    )

    assert (
        decision.rejection_reasons
        == []
    )


# ============================================================
# INTELLIGENCE REGRESSION
# ============================================================


def test_intelligence_regression_blocks_promotion(
    tmp_path: Path,
):

    (
        store,
        baseline_intelligence,
        _,
        baseline_safety,
        candidate_safety,
    ) = (
        persisted_reports(
            tmp_path
        )
    )

    bad_candidate = (
        store.save(
            report=(
                intelligence_report(
                    model_key=(
                        "candidate-model"
                    ),

                    pass_rate=0.75,

                    mean_score=0.95,

                    arguments_accuracy=0.75,
                )
            ),

            label=(
                "bad-candidate"
            ),
        )
    )

    decision = (
        build_model_promotion_decision(
            baseline_intelligence=(
                baseline_intelligence
            ),

            candidate_intelligence=(
                bad_candidate
            ),

            baseline_safety=(
                baseline_safety
            ),

            candidate_safety=(
                candidate_safety
            ),

            label=(
                "bad-candidate"
            ),
        )
    )

    assert (
        decision.promotion_eligible
        is False
    )

    assert (
        "intelligence_no_regression"
        in decision.rejection_reasons
    )


# ============================================================
# SAFETY REGRESSION
# ============================================================


def test_safety_failure_blocks_promotion(
    tmp_path: Path,
):

    (
        store,
        baseline_intelligence,
        candidate_intelligence,
        baseline_safety,
        _,
    ) = (
        persisted_reports(
            tmp_path
        )
    )

    bad_safety = (
        store.save(
            report=(
                safety_report(
                    pass_rate=0.75
                )
            ),

            label=(
                "bad-safety"
            ),
        )
    )

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
                bad_safety
            ),

            label=(
                "candidate"
            ),
        )
    )

    assert (
        decision.promotion_eligible
        is False
    )

    assert (
        "safety_no_regression"
        in decision.rejection_reasons
    )

    assert (
        "candidate_safety_full_pass"
        in decision.rejection_reasons
    )


# ============================================================
# ROLE VALIDATION
# ============================================================


def test_wrong_report_role_is_rejected(
    tmp_path: Path,
):

    (
        _,
        baseline_intelligence,
        candidate_intelligence,
        baseline_safety,
        candidate_safety,
    ) = (
        persisted_reports(
            tmp_path
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "Intelligence promotion reports"
        ),
    ):

        build_model_promotion_decision(
            baseline_intelligence=(
                baseline_safety
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
                "invalid"
            ),
        )


# ============================================================
# IMMUTABLE STORE / HASH
# ============================================================


def test_promotion_store_detects_tampering(
    tmp_path: Path,
):

    (
        _,
        baseline_intelligence,
        candidate_intelligence,
        baseline_safety,
        candidate_safety,
    ) = (
        persisted_reports(
            tmp_path
        )
    )

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
                "candidate"
            ),
        )
    )

    root = (
        tmp_path
        / "promotions"
    )

    store = (
        PromotionGateStore(
            root=(
                root
            )
        )
    )

    artifact = (
        store.save(
            decision
        )
    )

    loaded = (
        store.load(
            suite=(
                "core.v1"
            ),

            decision_id=(
                decision.decision_id
            ),
        )
    )

    assert (
        loaded.decision_sha256
        == artifact.decision_sha256
    )

    path = (
        root
        / "core.v1"
        / (
            f"{decision.decision_id}"
            ".json"
        )
    )

    raw = (
        json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    )

    raw[
        "decision"
    ][
        "promotion_eligible"
    ] = False

    path.write_text(
        json.dumps(
            raw,
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "SHA-256 verification failed"
        ),
    ):

        store.load(
            suite=(
                "core.v1"
            ),

            decision_id=(
                decision.decision_id
            ),
        )