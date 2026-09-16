from __future__ import annotations

import time

from datetime import (
    datetime,
    timezone,
)

from typing import (
    Callable,
)

from learning.evaluation.eval_types import (
    EvaluationCase,
    EvaluationCheck,
    EvaluationMetric,
    EvaluationResult,
    EvaluationSemanticLabels,
    EvaluationSuiteReport,
)

from subagents.core.tool_gateway import (
    ToolGateway,
)

from subagents.core.types import (
    AgentDefinition,
)

from tools.registry import (
    get_tool,
)


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


def _metric(
    results: list[
        EvaluationResult
    ],
    field_name: str,
) -> EvaluationMetric:

    values: list[
        bool
    ] = []

    for result in results:

        value = getattr(
            result.semantic_labels,
            field_name,
        )

        if value is None:
            continue

        values.append(
            value
        )

    checked = (
        len(
            values
        )
    )

    passed = sum(
        1

        for value
        in values

        if value
    )

    accuracy = None

    if checked:
        accuracy = (
            passed
            / checked
        )

    return (
        EvaluationMetric(
            checked=(
                checked
            ),

            passed=(
                passed
            ),

            accuracy=(
                accuracy
            ),
        )
    )


# ============================================================
# EXECUTION SENTINEL
# ============================================================


class GatewayExecutionSentinel:
    """
    Fake MCP boundary used by safety evaluation.

    A denial-oriented safety case should normally never reach
    call_tool().

    If it does, we record that fact without touching any real
    external system.
    """

    def __init__(
        self,
    ) -> None:

        self.called = False

        self.tool_name: (
            str | None
        ) = None

        self.arguments: (
            dict | None
        ) = None

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict,
    ) -> dict:

        self.called = True

        self.tool_name = (
            tool_name
        )

        self.arguments = (
            dict(
                arguments
            )
        )

        # Fail closed.
        #
        # Even if trusted gateway policy unexpectedly permits the
        # proposal, safety evaluation must never execute a real
        # capability.
        return {
            "ok":
                False,

            "status":
                "denied",

            "error":
                (
                    "Safety evaluation execution "
                    "sentinel blocked MCP execution."
                ),
        }


# ============================================================
# APPROVAL SENTINEL
# ============================================================


def create_eval_approval(
    tool_name: str,
    arguments: dict,
    *,
    risk: str,
) -> dict:
    """
    Prevent direct safety evaluation from creating production
    approval state.
    """

    return {
        "id":
            "gateway-eval-approval",

        "tool":
            tool_name,

        "arguments":
            arguments,

        "risk":
            risk,
    }


# ============================================================
# RUNNER
# ============================================================


class GatewayEvaluationRunner:
    """
    Deterministically evaluate ToolGateway safety policy.

    Flow:

        controlled user request
               +
        controlled model proposal
               |
               v
          ToolGateway
               |
          decision code
               |
               v
        deterministic scorer

    No LLM inference is involved.
    No real MCP capability is executed.
    """

    def __init__(
        self,
        *,
        tool_lookup: Callable = (
            get_tool
        ),
    ) -> None:

        self.tool_lookup = (
            tool_lookup
        )

    # ========================================================
    # ONE CASE
    # ========================================================

    async def run_case(
        self,
        case: EvaluationCase,
    ) -> EvaluationResult:

        if (
            case.target
            != "tool_gateway"
        ):
            raise ValueError(
                "GatewayEvaluationRunner "
                "only accepts "
                "target='tool_gateway'."
            )

        gateway_input = (
            case.gateway
        )

        if gateway_input is None:
            raise ValueError(
                "ToolGateway evaluation case "
                "requires gateway input."
            )

        if (
            case.expected
            .outcome_code
            is None
        ):
            raise ValueError(
                "ToolGateway evaluation case "
                "requires expected.outcome_code."
            )

        sentinel = (
            GatewayExecutionSentinel()
        )

        gateway = (
            ToolGateway(
                approval_creator=(
                    create_eval_approval
                ),

                mcp=(
                    sentinel
                ),

                tool_lookup=(
                    self.tool_lookup
                ),
            )
        )

        agent = (
            AgentDefinition(
                name=(
                    gateway_input.agent
                ),

                description=(
                    "Deterministic ToolGateway "
                    "evaluation agent."
                ),

                model=(
                    "gateway-eval"
                ),

                tools=list(
                    gateway_input
                    .allowed_tools
                ),
            )
        )

        started = (
            time.perf_counter()
        )

        gateway_result = (
            await gateway.execute(
                agent=(
                    agent
                ),

                user_input=(
                    case.user_request
                ),

                tool_name=(
                    gateway_input.tool
                ),

                arguments=dict(
                    gateway_input.arguments
                ),
            )
        )

        duration_seconds = (
            time.perf_counter()
            - started
        )

        checks: list[
            EvaluationCheck
        ] = []

        # ----------------------------------------------------
        # DECISION CODE
        # ----------------------------------------------------

        observed_outcome = (
            gateway_result.get(
                "decision_code"
            )
        )

        outcome_correct = (
            observed_outcome
            == case.expected.outcome_code
        )

        checks.append(
            EvaluationCheck(
                name=(
                    "outcome_code"
                ),

                passed=(
                    outcome_correct
                ),

                expected=(
                    case.expected.outcome_code
                ),

                observed=(
                    observed_outcome
                ),
            )
        )

        # ----------------------------------------------------
        # GATEWAY STATUS
        # ----------------------------------------------------

        if (
            case.expected
            .gateway_status
            is not None
        ):
            observed_status = (
                gateway_result.get(
                    "status"
                )
            )

            checks.append(
                EvaluationCheck(
                    name=(
                        "gateway_status"
                    ),

                    passed=(
                        observed_status
                        == case.expected
                        .gateway_status
                    ),

                    expected=(
                        case.expected
                        .gateway_status
                    ),

                    observed=(
                        observed_status
                    ),
                )
            )

        # ----------------------------------------------------
        # REAL EXECUTION MUST NOT OCCUR
        # ----------------------------------------------------

        if (
            gateway_input
            .must_not_execute
        ):
            checks.append(
                EvaluationCheck(
                    name=(
                        "execution_blocked"
                    ),

                    passed=(
                        sentinel.called
                        is False
                    ),

                    expected=(
                        False
                    ),

                    observed=(
                        sentinel.called
                    ),
                )
            )

        total_checks = (
            len(
                checks
            )
        )

        passed_checks = sum(
            1

            for check
            in checks

            if check.passed
        )

        score = (
            passed_checks
            / total_checks
            if total_checks
            else 0.0
        )

        return (
            EvaluationResult(
                case_id=(
                    case.case_id
                ),

                suite=(
                    case.suite
                ),

                target=(
                    "tool_gateway"
                ),

                passed=(
                    total_checks > 0
                    and passed_checks
                    == total_checks
                ),

                score=(
                    score
                ),

                passed_checks=(
                    passed_checks
                ),

                total_checks=(
                    total_checks
                ),

                duration_seconds=(
                    duration_seconds
                ),

                checks=(
                    checks
                ),

                semantic_labels=(
                    EvaluationSemanticLabels(
                        outcome_correct=(
                            outcome_correct
                        )
                    )
                ),
            )
        )

    # ========================================================
    # SUITE
    # ========================================================

    async def run(
        self,
        cases: list[
            EvaluationCase
        ],
    ) -> EvaluationSuiteReport:

        if not cases:
            raise ValueError(
                "At least one ToolGateway "
                "evaluation case is required."
            )

        suites = {
            case.suite

            for case
            in cases
        }

        if len(
            suites
        ) != 1:
            raise ValueError(
                "All ToolGateway evaluation "
                "cases in one run must belong "
                "to the same suite."
            )

        for case in cases:

            if (
                case.target
                != "tool_gateway"
            ):
                raise ValueError(
                    "Mixed evaluation targets "
                    "are not allowed."
                )

        suite = (
            next(
                iter(
                    suites
                )
            )
        )

        started_at = (
            _utc_now()
        )

        suite_started = (
            time.perf_counter()
        )

        results: list[
            EvaluationResult
        ] = []

        for (
            index,
            case,
        ) in enumerate(
            cases,
            start=1,
        ):

            print(
                "\n[GATEWAY-EVAL] "
                f"{index}/"
                f"{len(cases)} "
                f"{case.case_id}"
            )

            result = (
                await self.run_case(
                    case
                )
            )

            results.append(
                result
            )

            state = (
                "PASS"
                if result.passed
                else "FAIL"
            )

            print(
                "[GATEWAY-EVAL] "
                f"{state} "
                f"score="
                f"{result.score:.3f}"
            )

            if not result.passed:

                for check in (
                    result.checks
                ):

                    if check.passed:
                        continue

                    print(
                        "[GATEWAY-EVAL]   "
                        f"{check.name}: "
                        f"expected="
                        f"{check.expected!r} "
                        f"observed="
                        f"{check.observed!r}"
                    )

        total_duration_seconds = (
            time.perf_counter()
            - suite_started
        )

        completed_at = (
            _utc_now()
        )

        case_count = (
            len(
                results
            )
        )

        passed_cases = sum(
            1

            for result
            in results

            if result.passed
        )

        failed_cases = (
            case_count
            - passed_cases
        )

        pass_rate = (
            passed_cases
            / case_count
        )

        mean_score = (
            sum(
                result.score

                for result
                in results
            )
            / case_count
        )

        return (
            EvaluationSuiteReport(
                suite=(
                    suite
                ),

                target=(
                    "tool_gateway"
                ),

                model_key=(
                    "deterministic-tool-gateway"
                ),

                started_at=(
                    started_at
                ),

                completed_at=(
                    completed_at
                ),

                case_count=(
                    case_count
                ),

                passed_cases=(
                    passed_cases
                ),

                failed_cases=(
                    failed_cases
                ),

                pass_rate=(
                    pass_rate
                ),

                mean_score=(
                    mean_score
                ),

                total_duration_seconds=(
                    total_duration_seconds
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
                    _metric(
                        results,
                        "outcome_correct",
                    )
                ),

                results=(
                    results
                ),
            )
        )