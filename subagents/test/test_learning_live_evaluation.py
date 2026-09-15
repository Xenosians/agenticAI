import asyncio

import pytest

from learning.eval_types import (
    EvaluationCase,
    EvaluationExpectation,
)

from learning.live_evaluation import (
    LiveOrchestratorEvaluationRunner,
)

from subagents.core.types import (
    AgentResult,
    HubResult,
)


# ============================================================
# TEST HELPERS
# ============================================================


def run_async(
    coroutine,
):
    """
    Execute one async test operation without requiring
    pytest-asyncio or another pytest async plugin.

    The production evaluation runtime remains asynchronous.

    Only the test boundary is synchronous.
    """

    return (
        asyncio.run(
            coroutine
        )
    )


class FakeHub:
    async def run(
        self,
        user_request: str,
    ) -> HubResult:

        return (
            HubResult(
                status="success",

                user_request=(
                    user_request
                ),

                routes=[
                    "developer-specialist",
                ],

                results=[
                    AgentResult(
                        task_id=(
                            "task-live-eval"
                        ),

                        agent_name=(
                            "developer-specialist"
                        ),

                        status="success",

                        outcome_code=(
                            "success"
                        ),

                        proposed_tool=(
                            "workspace_git_status"
                        ),

                        proposed_arguments={
                            "repository":
                                "frontend",
                        },

                        tool_result={
                            "ok":
                                True,

                            "status":
                                "success",
                        },
                    )
                ],

                answer=(
                    "Frontend repository "
                    "is clean."
                ),
            )
        )


class WrongRepositoryHub:
    async def run(
        self,
        user_request: str,
    ) -> HubResult:

        return (
            HubResult(
                status=(
                    "partial_error"
                ),

                user_request=(
                    user_request
                ),

                routes=[
                    "developer-specialist",
                ],

                results=[
                    AgentResult(
                        task_id=(
                            "task-wrong"
                        ),

                        agent_name=(
                            "developer-specialist"
                        ),

                        status="error",

                        outcome_code=(
                            "grounding_failed"
                        ),

                        proposed_tool=(
                            "workspace_git_status"
                        ),

                        proposed_arguments={
                            "repository":
                                "ai",
                        },

                        error=(
                            "Grounding failed."
                        ),
                    )
                ],
            )
        )


def example_case(
) -> EvaluationCase:

    return (
        EvaluationCase(
            case_id=(
                "git.frontend.status"
            ),

            suite=(
                "core.v1"
            ),

            target=(
                "orchestrator"
            ),

            description=(
                "Frontend repository "
                "status."
            ),

            user_request=(
                "Show me the frontend "
                "git status."
            ),

            expected=(
                EvaluationExpectation(
                    hub_status=(
                        "success"
                    ),

                    routes=[
                        "developer-specialist",
                    ],

                    agent=(
                        "developer-specialist"
                    ),

                    tool=(
                        "workspace_git_status"
                    ),

                    arguments={
                        "repository":
                            "frontend",
                    },

                    outcome_code=(
                        "success"
                    ),
                )
            ),
        )
    )


# ============================================================
# LIVE CASE SCORING
# ============================================================


def test_live_runner_scores_real_hub_shape():

    runner = (
        LiveOrchestratorEvaluationRunner(
            hub=(
                FakeHub()
            ),

            hub_model=(
                "hub-main"
            ),
        )
    )

    result = (
        run_async(
            runner.run_case(
                example_case()
            )
        )
    )

    assert (
        result.passed
        is True
    )

    assert (
        result.score
        == 1.0
    )

    assert (
        result.duration_seconds
        is not None
    )

    assert (
        result.duration_seconds
        >= 0
    )

    assert (
        result.semantic_labels
        .route_correct
        is True
    )

    assert (
        result.semantic_labels
        .tool_correct
        is True
    )

    assert (
        result.semantic_labels
        .arguments_correct
        is True
    )

    assert (
        result.semantic_labels
        .outcome_correct
        is True
    )


# ============================================================
# SUITE METRICS
# ============================================================


def test_live_runner_builds_suite_metrics():

    runner = (
        LiveOrchestratorEvaluationRunner(
            hub=(
                FakeHub()
            ),

            hub_model=(
                "hub-main"
            ),
        )
    )

    report = (
        run_async(
            runner.run(
                [
                    example_case(),
                ]
            )
        )
    )

    assert (
        report.case_count
        == 1
    )

    assert (
        report.passed_cases
        == 1
    )

    assert (
        report.failed_cases
        == 0
    )

    assert (
        report.pass_rate
        == 1.0
    )

    assert (
        report.mean_score
        == 1.0
    )

    assert (
        report.route_metric
        .accuracy
        == 1.0
    )

    assert (
        report.tool_metric
        .accuracy
        == 1.0
    )

    assert (
        report.arguments_metric
        .accuracy
        == 1.0
    )

    assert (
        report.outcome_metric
        .accuracy
        == 1.0
    )

    assert (
        report.total_duration_seconds
        >= 0
    )


# ============================================================
# TARGET SEPARATION
# ============================================================


def test_live_runner_rejects_gateway_case():

    runner = (
        LiveOrchestratorEvaluationRunner(
            hub=(
                FakeHub()
            ),

            hub_model=(
                "hub-main"
            ),
        )
    )

    case = (
        example_case()
    )

    case.target = (
        "tool_gateway"
    )

    with pytest.raises(
        ValueError,
        match=(
            "target='orchestrator'"
        ),
    ):
        run_async(
            runner.run_case(
                case
            )
        )


# ============================================================
# FAILURE METRICS
# ============================================================


def test_failed_argument_changes_suite_metric():

    runner = (
        LiveOrchestratorEvaluationRunner(
            hub=(
                WrongRepositoryHub()
            ),

            hub_model=(
                "hub-main"
            ),
        )
    )

    report = (
        run_async(
            runner.run(
                [
                    example_case(),
                ]
            )
        )
    )

    assert (
        report.case_count
        == 1
    )

    assert (
        report.passed_cases
        == 0
    )

    assert (
        report.failed_cases
        == 1
    )

    assert (
        report.pass_rate
        == 0.0
    )

    # Routing was still correct.
    assert (
        report.route_metric
        .accuracy
        == 1.0
    )

    # Tool selection was also correct.
    assert (
        report.tool_metric
        .accuracy
        == 1.0
    )

    # The model invented the wrong repository.
    assert (
        report.arguments_metric
        .accuracy
        == 0.0
    )

    # Grounding correctly denied the bad proposal, but the eval
    # case expected successful execution.
    assert (
        report.outcome_metric
        .accuracy
        == 0.0
    )

    result = (
        report.results[
            0
        ]
    )

    assert (
        result.semantic_labels
        .route_correct
        is True
    )

    assert (
        result.semantic_labels
        .tool_correct
        is True
    )

    assert (
        result.semantic_labels
        .arguments_correct
        is False
    )

    assert (
        result.semantic_labels
        .outcome_correct
        is False
    )