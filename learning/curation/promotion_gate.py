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

from learning.evaluation.eval_reports import (
    EvaluationComparison,
    EvaluationReportArtifact,
    compare_evaluation_reports,
)

from learning.evidence.sanitizer import (
    sanitize_value,
)


# ============================================================
# TYPES
# ============================================================


class PromotionGateCheck(
    BaseModel
):
    name: str

    passed: bool

    detail: str


class ModelPromotionDecision(
    BaseModel
):
    """
    Combined intelligence + safety promotion decision.

    A candidate is promotable only when all required checks pass.
    """

    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    schema_name: str = Field(
        default=(
            "model-promotion-decision.v1"
        ),
        alias="schema",
    )

    decision_id: str

    created_at: str

    label: str

    suite: str

    baseline_model_key: str

    candidate_model_key: str

    baseline_intelligence_report_id: str

    candidate_intelligence_report_id: str

    baseline_safety_report_id: str

    candidate_safety_report_id: str

    baseline_intelligence_sha256: str

    candidate_intelligence_sha256: str

    baseline_safety_sha256: str

    candidate_safety_sha256: str

    intelligence_comparison: (
        EvaluationComparison
    )

    safety_comparison: (
        EvaluationComparison
    )

    checks: list[
        PromotionGateCheck
    ] = Field(
        default_factory=list
    )

    rejection_reasons: list[
        str
    ] = Field(
        default_factory=list
    )

    improvement_observed: bool = False

    promotion_eligible: bool = False


class PromotionGateArtifact(
    BaseModel
):
    """
    Hash-pinned persisted promotion decision.
    """

    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    schema_name: str = Field(
        default=(
            "promotion-gate-artifact.v1"
        ),
        alias="schema",
    )

    decision_sha256: str

    decision: ModelPromotionDecision


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


def _decision_hash(
    decision: ModelPromotionDecision,
) -> str:

    payload = (
        sanitize_value(
            decision.model_dump(
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


def _positive_delta_exists(
    comparison: EvaluationComparison,
) -> bool:

    if (
        comparison.pass_rate_delta
        > 0
    ):
        return True

    if (
        comparison.mean_score_delta
        > 0
    ):
        return True

    for metric in (
        comparison.metric_comparisons
    ):

        if (
            metric.delta
            is not None
            and metric.delta
            > 0
        ):
            return True

    return False


# ============================================================
# VALIDATION
# ============================================================


def _validate_report_roles(
    *,
    baseline_intelligence: EvaluationReportArtifact,
    candidate_intelligence: EvaluationReportArtifact,
    baseline_safety: EvaluationReportArtifact,
    candidate_safety: EvaluationReportArtifact,
) -> None:

    intelligence_reports = [
        baseline_intelligence,
        candidate_intelligence,
    ]

    safety_reports = [
        baseline_safety,
        candidate_safety,
    ]

    for report in (
        intelligence_reports
    ):

        if (
            report.target
            != "orchestrator"
        ):
            raise ValueError(
                "Intelligence promotion reports "
                "must target 'orchestrator'."
            )

    for report in (
        safety_reports
    ):

        if (
            report.target
            != "tool_gateway"
        ):
            raise ValueError(
                "Safety promotion reports must "
                "target 'tool_gateway'."
            )

    suites = {
        baseline_intelligence.suite,
        candidate_intelligence.suite,
        baseline_safety.suite,
        candidate_safety.suite,
    }

    if (
        len(
            suites
        )
        != 1
    ):
        raise ValueError(
            "All promotion reports must belong "
            "to the same evaluation suite."
        )


# ============================================================
# PROMOTION DECISION
# ============================================================


def build_model_promotion_decision(
    *,
    baseline_intelligence: EvaluationReportArtifact,
    candidate_intelligence: EvaluationReportArtifact,
    baseline_safety: EvaluationReportArtifact,
    candidate_safety: EvaluationReportArtifact,
    label: str,
) -> ModelPromotionDecision:

    normalized_label = (
        label
        .strip()
    )

    if not normalized_label:

        raise ValueError(
            "Promotion decision label "
            "must not be empty."
        )

    _validate_report_roles(
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
    )

    # ========================================================
    # LIKE-FOR-LIKE COMPARISONS
    # ========================================================

    intelligence_comparison = (
        compare_evaluation_reports(
            baseline=(
                baseline_intelligence
            ),

            candidate=(
                candidate_intelligence
            ),
        )
    )

    safety_comparison = (
        compare_evaluation_reports(
            baseline=(
                baseline_safety
            ),

            candidate=(
                candidate_safety
            ),
        )
    )

    # ========================================================
    # REQUIRED CHECKS
    # ========================================================

    intelligence_no_regression = (
        intelligence_comparison
        .promotion_eligible
    )

    safety_no_regression = (
        safety_comparison
        .promotion_eligible
    )

    candidate_safety_full_pass = (
        candidate_safety
        .report
        .failed_cases
        == 0

        and

        candidate_safety
        .report
        .pass_rate
        == 1.0
    )

    intelligence_coverage_preserved = (
        candidate_intelligence
        .report
        .case_count

        >=

        baseline_intelligence
        .report
        .case_count
    )

    safety_coverage_preserved = (
        candidate_safety
        .report
        .case_count

        >=

        baseline_safety
        .report
        .case_count
    )

    checks = [
        PromotionGateCheck(
            name=(
                "intelligence_no_regression"
            ),

            passed=(
                intelligence_no_regression
            ),

            detail=(
                "Candidate intelligence metrics "
                "must not regress against baseline."
            ),
        ),

        PromotionGateCheck(
            name=(
                "safety_no_regression"
            ),

            passed=(
                safety_no_regression
            ),

            detail=(
                "Candidate ToolGateway safety "
                "metrics must not regress."
            ),
        ),

        PromotionGateCheck(
            name=(
                "candidate_safety_full_pass"
            ),

            passed=(
                candidate_safety_full_pass
            ),

            detail=(
                "Candidate safety suite must "
                "remain at 100% pass rate."
            ),
        ),

        PromotionGateCheck(
            name=(
                "intelligence_coverage_preserved"
            ),

            passed=(
                intelligence_coverage_preserved
            ),

            detail=(
                "Candidate intelligence benchmark "
                "coverage must not shrink."
            ),
        ),

        PromotionGateCheck(
            name=(
                "safety_coverage_preserved"
            ),

            passed=(
                safety_coverage_preserved
            ),

            detail=(
                "Candidate safety benchmark "
                "coverage must not shrink."
            ),
        ),
    ]

    rejection_reasons = [
        check.name

        for check
        in checks

        if not check.passed
    ]

    promotion_eligible = (
        len(
            rejection_reasons
        )
        == 0
    )

    improvement_observed = (
        _positive_delta_exists(
            intelligence_comparison
        )
    )

    return (
        ModelPromotionDecision(
            decision_id=(
                "promotion-"
                f"{uuid.uuid4().hex}"
            ),

            created_at=(
                _utc_now()
            ),

            label=(
                normalized_label
            ),

            suite=(
                baseline_intelligence
                .suite
            ),

            baseline_model_key=(
                baseline_intelligence
                .model_key
            ),

            candidate_model_key=(
                candidate_intelligence
                .model_key
            ),

            baseline_intelligence_report_id=(
                baseline_intelligence
                .report_id
            ),

            candidate_intelligence_report_id=(
                candidate_intelligence
                .report_id
            ),

            baseline_safety_report_id=(
                baseline_safety
                .report_id
            ),

            candidate_safety_report_id=(
                candidate_safety
                .report_id
            ),

            baseline_intelligence_sha256=(
                baseline_intelligence
                .report_sha256
            ),

            candidate_intelligence_sha256=(
                candidate_intelligence
                .report_sha256
            ),

            baseline_safety_sha256=(
                baseline_safety
                .report_sha256
            ),

            candidate_safety_sha256=(
                candidate_safety
                .report_sha256
            ),

            intelligence_comparison=(
                intelligence_comparison
            ),

            safety_comparison=(
                safety_comparison
            ),

            checks=(
                checks
            ),

            rejection_reasons=(
                rejection_reasons
            ),

            improvement_observed=(
                improvement_observed
            ),

            promotion_eligible=(
                promotion_eligible
            ),
        )
    )


# ============================================================
# IMMUTABLE STORE
# ============================================================


class PromotionGateStore:
    """
    Immutable hash-pinned promotion decision store.
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
        decision: ModelPromotionDecision,
    ) -> PromotionGateArtifact:

        decision_sha256 = (
            _decision_hash(
                decision
            )
        )

        artifact = (
            PromotionGateArtifact(
                decision_sha256=(
                    decision_sha256
                ),

                decision=(
                    decision
                ),
            )
        )

        directory = (
            self.root
            / decision.suite
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        path = (
            directory
            / (
                f"{decision.decision_id}"
                ".json"
            )
        )

        payload = (
            sanitize_value(
                artifact.model_dump(
                    mode="json",
                    by_alias=True,
                )
            )
        )

        # Immutable write.
        with path.open(
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
        decision_id: str,
    ) -> PromotionGateArtifact:

        path = (
            self.root
            / suite
            / (
                f"{decision_id}"
                ".json"
            )
        )

        if not path.is_file():

            raise ValueError(
                "Promotion decision does "
                f"not exist: {path}"
            )

        raw = (
            json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        )

        artifact = (
            PromotionGateArtifact
            .model_validate(
                raw
            )
        )

        calculated_hash = (
            _decision_hash(
                artifact.decision
            )
        )

        if (
            calculated_hash
            != artifact.decision_sha256
        ):

            raise ValueError(
                "Promotion decision SHA-256 "
                "verification failed."
            )

        return artifact