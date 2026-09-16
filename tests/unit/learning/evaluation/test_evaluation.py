from pathlib import (
    Path,
)

import pytest

from learning.evaluation.eval_suite import (
    load_evaluation_cases,
)

from learning.evaluation.eval_types import (
    EvaluationCase,
    EvaluationExpectation,
)

from learning.evaluation.evaluator import (
    evaluate_trajectory,
)

from learning.evidence.types import (
    ExecutionReward,
    LearningTrajectory,
    TrajectoryQuality,
    TrajectorySignals,
    TrajectoryStep,
)


def trajectory(
    *,
    repository: str = (
        "frontend"
    ),
    tool: str = (
        "workspace_git_status"
    ),
    outcome_code: str = (
        "success"
    ),
    hub_status: str = (
        "success"
    ),
    routes: list[
        str
    ] | None = None,
) -> LearningTrajectory:

    return (
        LearningTrajectory(
            trajectory_id=(
                "trajectory-eval"
            ),

            observed_at=(
                "2026-09-15T00:00:00+00:00"
            ),

            job_id=(
                "job-eval"
            ),

            attempt=1,

            user_request=(
                "Show me the frontend "
                "git status."
            ),

            hub_model=(
                "hub-main"
            ),

            hub_status=(
                hub_status
            ),

            routes=(
                routes
                if routes is not None
                else [
                    "developer-specialist"
                ]
            ),

            steps=[
                TrajectoryStep(
                    task_id=(
                        "task-eval"
                    ),

                    agent=(
                        "developer-specialist"
                    ),

                    status=(
                        "success"
                        if outcome_code
                        == "success"
                        else "error"
                    ),

                    outcome_code=(
                        outcome_code
                    ),

                    proposed_tool=(
                        tool
                    ),

                    proposed_arguments={
                        "repository":
                            repository,
                    },
                )
            ],

            signals=(
                TrajectorySignals(
                    delegated=True,
                )
            ),

            execution_reward=(
                ExecutionReward()
            ),

            quality=(
                TrajectoryQuality()
            ),
        )
    )


def case(
) -> EvaluationCase:

    return (
        EvaluationCase(
            case_id=(
                "git.frontend.status"
            ),

            suite=(
                "core.v1"
            ),

            description=(
                "Frontend repository status."
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


def test_perfect_trajectory_scores_one():
    result = (
        evaluate_trajectory(
            case=(
                case()
            ),

            trajectory=(
                trajectory()
            ),
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
        result.passed_checks
        == result.total_checks
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


def test_wrong_repository_fails_arguments_only():
    result = (
        evaluate_trajectory(
            case=(
                case()
            ),

            trajectory=(
                trajectory(
                    repository="ai"
                )
            ),
        )
    )

    assert (
        result.passed
        is False
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


def test_wrong_tool_is_detected():
    result = (
        evaluate_trajectory(
            case=(
                case()
            ),

            trajectory=(
                trajectory(
                    tool=(
                        "workspace_git_log"
                    )
                )
            ),
        )
    )

    assert (
        result.passed
        is False
    )

    assert (
        result.semantic_labels
        .tool_correct
        is False
    )


def test_wrong_route_is_detected():
    result = (
        evaluate_trajectory(
            case=(
                case()
            ),

            trajectory=(
                trajectory(
                    routes=[
                        "ticket-specialist"
                    ]
                )
            ),
        )
    )

    assert (
        result.passed
        is False
    )

    assert (
        result.semantic_labels
        .route_correct
        is False
    )


def test_expected_grounding_denial_can_pass():
    eval_case = (
        case()
    )

    eval_case.expected.hub_status = (
        "partial_error"
    )

    eval_case.expected.arguments = {
        "repository":
            "ai",
    }

    eval_case.expected.outcome_code = (
        "grounding_failed"
    )

    result = (
        evaluate_trajectory(
            case=(
                eval_case
            ),

            trajectory=(
                trajectory(
                    repository="ai",

                    outcome_code=(
                        "grounding_failed"
                    ),

                    hub_status=(
                        "partial_error"
                    ),
                )
            ),
        )
    )

    assert (
        result.passed
        is True
    )

    assert (
        result.semantic_labels
        .outcome_correct
        is True
    )


def test_old_trajectory_without_quality_can_load():
    raw = {
        "schema":
            "trajectory.v1",

        "trajectory_id":
            "legacy-1",

        "observed_at":
            "2026-09-01T00:00:00+00:00",

        "job_id":
            "job-1",

        "attempt":
            1,

        "user_request":
            "Legacy request",

        "hub_model":
            "hub-main",

        "hub_status":
            "success",

        "routes":
            [],

        "steps":
            [],

        "final_answer":
            "Hello",

        "signals": {
            "delegated":
                False
        },

        "execution_reward": {
            "schema":
                "execution-reward.v1",

            "components":
                {},

            "total":
                0.0,

            "quality_eligible":
                False
        },

        "dataset_eligible":
            False
    }

    parsed = (
        LearningTrajectory
        .model_validate(
            raw
        )
    )

    assert (
        parsed.quality
        is None
    )


def test_eval_suite_loads_and_rejects_duplicates(
    tmp_path: Path,
):
    suite = (
        tmp_path
        / "suite.jsonl"
    )

    row = (
        '{"schema":"evaluation-case.v1",'
        '"case_id":"case-1",'
        '"suite":"test",'
        '"description":"Example",'
        '"user_request":"Hello",'
        '"expected":{"hub_status":"success"}}'
    )

    suite.write_text(
        row + "\n",
        encoding="utf-8",
    )

    cases = (
        load_evaluation_cases(
            suite
        )
    )

    assert (
        len(
            cases
        )
        == 1
    )

    suite.write_text(
        row
        + "\n"
        + row
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Duplicate evaluation case_id"
        ),
    ):
        load_evaluation_cases(
            suite
        )