from learning.continual.failure_diagnostics import diagnose_trajectory
from learning.evidence.types import (
    ExecutionReward,
    LearningTrajectory,
    TrajectorySignals,
    TrajectoryStep,
)


def _trajectory(*, request: str, intent: dict, arguments: dict, decision_code: str):
    return LearningTrajectory(
        trajectory_id="trajectory-1",
        observed_at="2026-09-25T00:00:00+00:00",
        job_id="job-1",
        attempt=1,
        user_request=request,
        hub_model="hub-main",
        hub_status="partial_error",
        routes=["jira-specialist"],
        steps=[
            TrajectoryStep(
                task_id="task-1",
                task_instructions="Create the requested ticket.",
                semantic_intent=intent,
                semantic_guard_decision={
                    "allowed": False,
                    "decision_code": decision_code,
                    "error": "denied",
                },
                agent="jira-specialist",
                status="error",
                outcome_code="semantic_guard_denied",
                proposed_tool="ticket_create",
                proposed_arguments=arguments,
            )
        ],
        signals=TrajectorySignals(
            delegated=True,
            route_count=1,
            specialist_count=1,
            specialist_error_count=1,
            overall_success=False,
            had_error=True,
        ),
        execution_reward=ExecutionReward(
            components={"hub_error": -1.0},
            total=-1.0,
            quality_eligible=False,
        ),
    )


def test_unbound_explicit_argument_becomes_review_only_candidate():
    trajectory = _trajectory(
        request='Create a Task in Jira project KAN with summary "Checkpoint demo ticket".',
        intent={
            "summary": "Create a Task",
            "effect": "mutation",
            "allowed_tools": ["ticket_create"],
            "forbidden_tools": [],
            "allowed_arguments": {
                "project_key": ["KAN"],
                "summary": ["Checkpoint demo ticket"],
            },
            "forbidden_arguments": {},
            "max_tool_calls": 1,
            "clarification_required": False,
        },
        arguments={
            "project_key": "KAN",
            "summary": "Checkpoint demo ticket",
            "ticket_type": "Task",
        },
        decision_code="semantic_argument_unbound",
    )

    diagnoses, candidates = diagnose_trajectory(trajectory)

    assert len(diagnoses) == 1
    assert diagnoses[0].failure_code == "semantic_argument_unbound"
    assert diagnoses[0].component == "hub_router"

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.review_required is True
    assert candidate.training_eligible is False
    assert candidate.missing_bindings == {"ticket_type": ["Task"]}
    assert candidate.rejected_semantic_intent["allowed_arguments"] == {
        "project_key": ["KAN"],
        "summary": ["Checkpoint demo ticket"],
    }
    assert candidate.proposed_semantic_intent["allowed_arguments"] == {
        "project_key": ["KAN"],
        "summary": ["Checkpoint demo ticket"],
        "ticket_type": ["Task"],
    }


def test_unbound_value_not_literal_in_user_request_is_not_auto_candidate():
    trajectory = _trajectory(
        request='Create something in Jira project KAN.',
        intent={
            "summary": "Create something",
            "effect": "mutation",
            "allowed_tools": ["ticket_create"],
            "forbidden_tools": [],
            "allowed_arguments": {"project_key": ["KAN"]},
            "forbidden_arguments": {},
            "max_tool_calls": 1,
            "clarification_required": False,
        },
        arguments={
            "project_key": "KAN",
            "ticket_type": "Task",
        },
        decision_code="semantic_argument_unbound",
    )

    diagnoses, candidates = diagnose_trajectory(trajectory)

    assert len(diagnoses) == 1
    assert candidates == []


def test_candidate_requires_tool_already_exactly_bound():
    trajectory = _trajectory(
        request='Create a Task in Jira project KAN.',
        intent={
            "summary": "Create a Task",
            "effect": "mutation",
            "allowed_tools": ["jira_project_get"],
            "forbidden_tools": [],
            "allowed_arguments": {"project_key": ["KAN"]},
            "forbidden_arguments": {},
            "max_tool_calls": 1,
            "clarification_required": False,
        },
        arguments={
            "project_key": "KAN",
            "ticket_type": "Task",
        },
        decision_code="semantic_argument_unbound",
    )

    _, candidates = diagnose_trajectory(trajectory)
    assert candidates == []
