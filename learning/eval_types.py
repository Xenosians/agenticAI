from __future__ import annotations

from typing import (
    Any,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class EvaluationExpectation(
    BaseModel
):
    """
    Deterministic expectations for one evaluation case.

    Fields may remain None when that aspect is intentionally
    outside the scope of the case.
    """

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

    outcome_code: (
        str | None
    ) = None


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

    description: str

    user_request: str

    expected: (
        EvaluationExpectation
    )


class EvaluationCheck(
    BaseModel
):
    name: str

    passed: bool

    expected: Any = None

    observed: Any = None


class EvaluationSemanticLabels(
    BaseModel
):
    """
    Semantic labels proven by this evaluation case.

    None means the case did not test that property.
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

    passed: bool

    score: float

    passed_checks: int

    total_checks: int

    checks: list[
        EvaluationCheck
    ] = Field(
        default_factory=list
    )

    semantic_labels: (
        EvaluationSemanticLabels
    )