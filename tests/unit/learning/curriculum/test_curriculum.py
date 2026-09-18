from learning.curriculum.difficulty import assess_difficulty
from learning.curriculum.domains import classify_trajectory_domains
from learning.evidence.types import ExecutionReward, LearningTrajectory, TrajectorySignals, TrajectoryStep


def _trajectory():
    return LearningTrajectory(
        trajectory_id="t1",
        observed_at="2026-09-18T00:00:00+00:00",
        job_id="j1",
        attempt=1,
        user_request="Check a ticket then inspect git",
        hub_model="hub-main",
        hub_status="partial_error",
        routes=["ticket-specialist", "developer-specialist"],
        steps=[
            TrajectoryStep(task_id="a", agent="ticket-specialist", status="success", outcome_code="success", proposed_tool="ticket_get", proposed_arguments={"ticket_key": "KAN-1"}),
            TrajectoryStep(task_id="b", agent="developer-specialist", status="error", outcome_code="grounding_failed", proposed_tool="workspace_git_status", proposed_arguments={"repository": "ai"}),
        ],
        signals=TrajectorySignals(delegated=True, route_count=2, specialist_count=2, specialist_error_count=1, had_error=True),
        execution_reward=ExecutionReward(total=-1.0),
    )


def test_cross_domain_grounding_failure_is_hard():
    trajectory = _trajectory()
    domain = classify_trajectory_domains(trajectory)
    difficulty = assess_difficulty(trajectory, domain=domain, correction_count=1, context_kind_count=3)
    assert domain.cross_domain is True
    assert "grounding" in difficulty.failure_clusters
    assert difficulty.band == "hard"
