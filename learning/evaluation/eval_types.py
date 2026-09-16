from __future__ import annotations

from typing import (
    Any,
    Literal,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


EvaluationTarget = Literal[
    "orchestrator",
    "tool_gateway",
]


# ============================================================
# TOOL GATEWAY EVALUATION INPUT
# ============================================================


class GatewayEvaluationInput(
    BaseModel
):
    """
    Controlled proposal injected directly into ToolGateway.

    This is deliberately separate from expected behavior.

    Example:

        user_request:
            "Show frontend status"

        gateway proposal:
            repository = "ai"

        expected:
            grounding_failed

    The LLM is therefore never asked to intentionally make a
    dangerous or incorrect proposal just so security policy can
    be tested.
    """

    agent: str

    allowed_tools: list[
        str
    ] = Field(
        default_factory=list
    )

    tool: str

    arguments: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )

    # Security cases normally assert that execution must never
    # reach MCP.
    must_not_execute: bool = True


# ============================================================
# EXPECTATIONS
# ============================================================


class EvaluationExpectation(
    BaseModel
):
    """
    Deterministic expectations for one evaluation case.

    Fields may remain None when that dimension is intentionally
    outside the scope of the case.
    """

    # --------------------------------------------------------
    # Orchestrator expectations
    # --------------------------------------------------------

    hub_status: (
        str | None
    ) = None

    routes: (
        list[str]
        | None
    ) = None

    agent: (
        str | None
    ) = None

    tool: (
        str | None
    ) = None

    arguments: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None

    # --------------------------------------------------------
    # Shared runtime decision expectation
    # --------------------------------------------------------

    outcome_code: (
        str | None
    ) = None

    # --------------------------------------------------------
    # Direct ToolGateway expectation
    # --------------------------------------------------------

    gateway_status: (
        str | None
    ) = None


# ============================================================
# EVALUATION CASE
# ============================================================


class EvaluationCase(
    BaseModel
):
    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    schema_name: str = Field(
        default=(
            "evaluation-case.v1"
        ),
        alias="schema",
    )

    case_id: str

    suite: str

    target: EvaluationTarget = (
        "orchestrator"
    )

    description: str

    user_request: str

    # Present only for direct ToolGateway cases.
    gateway: (
        GatewayEvaluationInput
        | None
    ) = None

    expected: (
        EvaluationExpectation
    )


# ============================================================
# INDIVIDUAL CHECK
# ============================================================


class EvaluationCheck(
    BaseModel
):
    name: str

    passed: bool

    expected: Any = None

    observed: Any = None


# ============================================================
# SEMANTIC LABELS
# ============================================================


class EvaluationSemanticLabels(
    BaseModel
):
    """
    Semantic labels proven by this evaluation case.

    None means that property was not evaluated.
    """

    route_correct: (
        bool | None
    ) = None

    tool_correct: (
        bool | None
    ) = None

    arguments_correct: (
        bool | None
    ) = None

    outcome_correct: (
        bool | None
    ) = None


# ============================================================
# CASE RESULT
# ============================================================


class EvaluationResult(
    BaseModel
):
    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    schema_name: str = Field(
        default=(
            "evaluation-result.v1"
        ),
        alias="schema",
    )

    case_id: str

    suite: str

    target: EvaluationTarget = (
        "orchestrator"
    )

    passed: bool

    score: float

    passed_checks: int

    total_checks: int

    duration_seconds: (
        float | None
    ) = None

    checks: list[
        EvaluationCheck
    ] = Field(
        default_factory=list
    )

    semantic_labels: (
        EvaluationSemanticLabels
    )


# ============================================================
# AGGREGATED METRIC
# ============================================================


class EvaluationMetric(
    BaseModel
):
    checked: int = 0

    passed: int = 0

    accuracy: (
        float | None
    ) = None


# ============================================================
# SUITE REPORT
# ============================================================


class EvaluationSuiteReport(
    BaseModel
):
    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    schema_name: str = Field(
        default=(
            "evaluation-suite-report.v1"
        ),
        alias="schema",
    )

    suite: str

    target: EvaluationTarget

    model_key: str

    started_at: str

    completed_at: str

    case_count: int

    passed_cases: int

    failed_cases: int

    pass_rate: float

    mean_score: float

    total_duration_seconds: float

    route_metric: EvaluationMetric

    tool_metric: EvaluationMetric

    arguments_metric: EvaluationMetric

    outcome_metric: EvaluationMetric

    results: list[
        EvaluationResult
    ] = Field(
        default_factory=list
    )