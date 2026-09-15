from __future__ import annotations

import hashlib
import json
import uuid

from datetime import (
    datetime,
    timezone,
)

from pathlib import (
    Path,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from learning.eval_types import (
    EvaluationMetric,
    EvaluationSuiteReport,
)

from learning.sanitizer import (
    sanitize_value,
)


# ============================================================
# REPORT ARTIFACT
# ============================================================


class EvaluationReportArtifact(
    BaseModel
):
    """
    Immutable persisted evaluation report.

    The nested EvaluationSuiteReport contains the actual metrics.

    report_sha256 is calculated from the canonical serialized
    report payload so later comparison code can verify exactly
    which evaluation evidence was used.
    """

    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    schema_name: str = Field(
        default=(
            "evaluation-report-artifact.v1"
        ),
        alias="schema",
    )

    report_id: str

    created_at: str

    label: str

    suite: str

    target: str

    model_key: str

    report_sha256: str

    report: EvaluationSuiteReport


# ============================================================
# METRIC COMPARISON
# ============================================================


class EvaluationMetricComparison(
    BaseModel
):
    metric: str

    baseline_checked: int
    candidate_checked: int

    baseline_accuracy: (
        float | None
    ) = None

    candidate_accuracy: (
        float | None
    ) = None

    delta: (
        float | None
    ) = None

    regression: bool = False


class EvaluationComparison(
    BaseModel
):
    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    schema_name: str = Field(
        default=(
            "evaluation-comparison.v1"
        ),
        alias="schema",
    )

    comparison_id: str

    created_at: str

    baseline_report_id: str

    candidate_report_id: str

    suite: str

    target: str

    baseline_model_key: str

    candidate_model_key: str

    baseline_pass_rate: float

    candidate_pass_rate: float

    pass_rate_delta: float

    baseline_mean_score: float

    candidate_mean_score: float

    mean_score_delta: float

    metric_comparisons: list[
        EvaluationMetricComparison
    ] = Field(
        default_factory=list
    )

    regressions: list[
        str
    ] = Field(
        default_factory=list
    )

    promotion_eligible: bool = False


# ============================================================
# HELPERS
# ============================================================


def _utc_now(
) -> str:

    return (
        datetime
        .now(
            timezone.utc
        )
        .isoformat()
    )


def _canonical_json(
    value: dict,
) -> str:

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )
    )


def _report_hash(
    report: EvaluationSuiteReport,
) -> str:

    payload = (
        sanitize_value(
            report.model_dump(
                mode="json",
                by_alias=True,
            )
        )
    )

    serialized = (
        _canonical_json(
            payload
        )
    )

    return (
        hashlib
        .sha256(
            serialized.encode(
                "utf-8"
            )
        )
        .hexdigest()
    )


def _compare_metric(
    *,
    name: str,
    baseline: EvaluationMetric,
    candidate: EvaluationMetric,
) -> EvaluationMetricComparison:

    baseline_accuracy = (
        baseline.accuracy
    )

    candidate_accuracy = (
        candidate.accuracy
    )

    delta = None

    regression = False

    if (
        baseline_accuracy
        is not None
        and candidate_accuracy
        is not None
    ):
        delta = (
            candidate_accuracy
            - baseline_accuracy
        )

        regression = (
            delta
            < 0
        )

    elif (
        baseline_accuracy
        is not None
        and candidate_accuracy
        is None
    ):
        # Candidate stopped evaluating a metric that the baseline
        # covered. Treat that as a regression rather than silently
        # accepting reduced benchmark coverage.
        regression = True

    return (
        EvaluationMetricComparison(
            metric=(
                name
            ),

            baseline_checked=(
                baseline.checked
            ),

            candidate_checked=(
                candidate.checked
            ),

            baseline_accuracy=(
                baseline_accuracy
            ),

            candidate_accuracy=(
                candidate_accuracy
            ),

            delta=(
                delta
            ),

            regression=(
                regression
            ),
        )
    )


# ============================================================
# REPORT STORE
# ============================================================


class EvaluationReportStore:
    """
    Immutable local evaluation evidence store.

    Existing reports are never overwritten.
    """

    def __init__(
        self,
        *,
        root: Path,
    ) -> None:

        self.root = (
            root
            .expanduser()
            .resolve()
        )

    def save(
        self,
        *,
        report: EvaluationSuiteReport,
        label: str,
    ) -> EvaluationReportArtifact:

        normalized_label = (
            label
            .strip()
        )

        if not normalized_label:
            raise ValueError(
                "Evaluation report label "
                "must not be empty."
            )

        report_id = (
            "eval-"
            f"{uuid.uuid4().hex}"
        )

        created_at = (
            _utc_now()
        )

        report_sha256 = (
            _report_hash(
                report
            )
        )

        artifact = (
            EvaluationReportArtifact(
                report_id=(
                    report_id
                ),

                created_at=(
                    created_at
                ),

                label=(
                    normalized_label
                ),

                suite=(
                    report.suite
                ),

                target=(
                    report.target
                ),

                model_key=(
                    report.model_key
                ),

                report_sha256=(
                    report_sha256
                ),

                report=(
                    report
                ),
            )
        )

        target_dir = (
            self.root
            / report.suite
            / report.target
        )

        target_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        target_path = (
            target_dir
            / f"{report_id}.json"
        )

        payload = (
            sanitize_value(
                artifact.model_dump(
                    mode="json",
                    by_alias=True,
                )
            )
        )

        # "x" mode guarantees accidental overwrite is impossible.
        with target_path.open(
            "x",
            encoding="utf-8",
        ) as handle:

            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )

            handle.write(
                "\n"
            )

        return artifact

    def load(
        self,
        *,
        suite: str,
        target: str,
        report_id: str,
    ) -> EvaluationReportArtifact:

        path = (
            self.root
            / suite
            / target
            / f"{report_id}.json"
        )

        if not path.is_file():
            raise ValueError(
                "Evaluation report does not "
                f"exist: {path}"
            )

        raw = (
            json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        )

        artifact = (
            EvaluationReportArtifact
            .model_validate(
                raw
            )
        )

        calculated_hash = (
            _report_hash(
                artifact.report
            )
        )

        if (
            calculated_hash
            != artifact.report_sha256
        ):
            raise ValueError(
                "Evaluation report SHA-256 "
                "verification failed."
            )

        return artifact

    def list_reports(
        self,
        *,
        suite: str,
        target: str,
    ) -> list[
        EvaluationReportArtifact
    ]:

        directory = (
            self.root
            / suite
            / target
        )

        if not directory.exists():
            return []

        artifacts: list[
            EvaluationReportArtifact
        ] = []

        for path in sorted(
            directory.glob(
                "eval-*.json"
            )
        ):
            raw = (
                json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )
            )

            artifacts.append(
                EvaluationReportArtifact
                .model_validate(
                    raw
                )
            )

        return artifacts


# ============================================================
# COMPARISON
# ============================================================


def compare_evaluation_reports(
    *,
    baseline: EvaluationReportArtifact,
    candidate: EvaluationReportArtifact,
) -> EvaluationComparison:
    """
    Compare like-for-like evaluation reports.

    Candidate promotion is allowed only when:

    - suite matches
    - target matches
    - benchmark coverage does not shrink
    - pass rate does not regress
    - mean score does not regress
    - no individual semantic metric regresses
    """

    if (
        baseline.suite
        != candidate.suite
    ):
        raise ValueError(
            "Cannot compare evaluation "
            "reports from different suites."
        )

    if (
        baseline.target
        != candidate.target
    ):
        raise ValueError(
            "Cannot compare evaluation "
            "reports from different targets."
        )

    baseline_report = (
        baseline.report
    )

    candidate_report = (
        candidate.report
    )

    comparisons = [
        _compare_metric(
            name="routing",

            baseline=(
                baseline_report
                .route_metric
            ),

            candidate=(
                candidate_report
                .route_metric
            ),
        ),

        _compare_metric(
            name="tool",

            baseline=(
                baseline_report
                .tool_metric
            ),

            candidate=(
                candidate_report
                .tool_metric
            ),
        ),

        _compare_metric(
            name="arguments",

            baseline=(
                baseline_report
                .arguments_metric
            ),

            candidate=(
                candidate_report
                .arguments_metric
            ),
        ),

        _compare_metric(
            name="outcome",

            baseline=(
                baseline_report
                .outcome_metric
            ),

            candidate=(
                candidate_report
                .outcome_metric
            ),
        ),
    ]

    regressions: list[
        str
    ] = []

    if (
        candidate_report.case_count
        < baseline_report.case_count
    ):
        regressions.append(
            "case_coverage"
        )

    if (
        candidate_report.pass_rate
        < baseline_report.pass_rate
    ):
        regressions.append(
            "pass_rate"
        )

    if (
        candidate_report.mean_score
        < baseline_report.mean_score
    ):
        regressions.append(
            "mean_score"
        )

    for comparison in comparisons:

        if comparison.regression:
            regressions.append(
                comparison.metric
            )

    # Preserve stable ordering while removing duplicates.
    regressions = list(
        dict.fromkeys(
            regressions
        )
    )

    return (
        EvaluationComparison(
            comparison_id=(
                "comparison-"
                f"{uuid.uuid4().hex}"
            ),

            created_at=(
                _utc_now()
            ),

            baseline_report_id=(
                baseline.report_id
            ),

            candidate_report_id=(
                candidate.report_id
            ),

            suite=(
                baseline.suite
            ),

            target=(
                baseline.target
            ),

            baseline_model_key=(
                baseline.model_key
            ),

            candidate_model_key=(
                candidate.model_key
            ),

            baseline_pass_rate=(
                baseline_report
                .pass_rate
            ),

            candidate_pass_rate=(
                candidate_report
                .pass_rate
            ),

            pass_rate_delta=(
                candidate_report
                .pass_rate
                - baseline_report
                .pass_rate
            ),

            baseline_mean_score=(
                baseline_report
                .mean_score
            ),

            candidate_mean_score=(
                candidate_report
                .mean_score
            ),

            mean_score_delta=(
                candidate_report
                .mean_score
                - baseline_report
                .mean_score
            ),

            metric_comparisons=(
                comparisons
            ),

            regressions=(
                regressions
            ),

            promotion_eligible=(
                len(
                    regressions
                )
                == 0
            ),
        )
    )