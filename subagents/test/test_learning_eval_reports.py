from pathlib import (
    Path,
)

import pytest

from learning.eval_reports import (
    EvaluationReportStore,
    compare_evaluation_reports,
)

from learning.eval_types import (
    EvaluationMetric,
    EvaluationSuiteReport,
)


def report(
    *,
    pass_rate: float = 1.0,
    mean_score: float = 1.0,
    argument_accuracy: float = 1.0,
) -> EvaluationSuiteReport:

    passed_cases = (
        4
        if pass_rate == 1.0
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
                "hub-main"
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
                        if argument_accuracy
                        == 1.0
                        else 3
                    ),

                    accuracy=(
                        argument_accuracy
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


def test_report_store_round_trip(
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

    artifact = (
        store.save(
            report=(
                report()
            ),

            label=(
                "baseline"
            ),
        )
    )

    loaded = (
        store.load(
            suite=(
                "core.v1"
            ),

            target=(
                "orchestrator"
            ),

            report_id=(
                artifact.report_id
            ),
        )
    )

    assert (
        loaded.report_id
        == artifact.report_id
    )

    assert (
        loaded.report_sha256
        == artifact.report_sha256
    )

    assert (
        loaded.report.pass_rate
        == 1.0
    )


def test_report_store_is_immutable(
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

    first = (
        store.save(
            report=(
                report()
            ),

            label=(
                "baseline"
            ),
        )
    )

    second = (
        store.save(
            report=(
                report()
            ),

            label=(
                "baseline"
            ),
        )
    )

    assert (
        first.report_id
        != second.report_id
    )


def test_identical_candidate_is_promotable(
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

    baseline = (
        store.save(
            report=(
                report()
            ),

            label=(
                "baseline"
            ),
        )
    )

    candidate = (
        store.save(
            report=(
                report()
            ),

            label=(
                "candidate"
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

    assert (
        comparison.promotion_eligible
        is True
    )

    assert (
        comparison.regressions
        == []
    )


def test_argument_regression_blocks_promotion(
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

    baseline = (
        store.save(
            report=(
                report()
            ),

            label=(
                "baseline"
            ),
        )
    )

    candidate = (
        store.save(
            report=(
                report(
                    pass_rate=0.75,
                    mean_score=0.95,
                    argument_accuracy=0.75,
                )
            ),

            label=(
                "candidate"
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

    assert (
        comparison.promotion_eligible
        is False
    )

    assert (
        "pass_rate"
        in comparison.regressions
    )

    assert (
        "mean_score"
        in comparison.regressions
    )

    assert (
        "arguments"
        in comparison.regressions
    )


def test_hash_tampering_is_detected(
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

    artifact = (
        store.save(
            report=(
                report()
            ),

            label=(
                "baseline"
            ),
        )
    )

    path = (
        tmp_path
        / "evaluations"
        / "core.v1"
        / "orchestrator"
        / f"{artifact.report_id}.json"
    )

    text = (
        path.read_text(
            encoding="utf-8"
        )
    )

    text = (
        text.replace(
            '"pass_rate": 1.0',
            '"pass_rate": 0.5',
        )
    )

    path.write_text(
        text,
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

            target=(
                "orchestrator"
            ),

            report_id=(
                artifact.report_id
            ),
        )