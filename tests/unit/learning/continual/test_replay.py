from learning.continual.replay import ReplayPolicy, select_replay
from learning.continual.types import WeakLearningSignal
from learning.evidence.types import ExecutionReward, LearningTrajectory, TrajectorySignals, TrajectoryStep


def _trajectory(index: int, tool: str) -> LearningTrajectory:
    return LearningTrajectory(
        trajectory_id=f"trajectory-{index}",
        observed_at=f"2026-09-18T00:{index:02d}:00+00:00",
        job_id=f"job-{index}",
        attempt=1,
        user_request=f"request {index}",
        hub_model="hub-main",
        hub_status="success",
        routes=["specialist"],
        steps=[TrajectoryStep(task_id=f"task-{index}", agent="specialist", status="success", outcome_code="success", proposed_tool=tool, proposed_arguments={})],
        signals=TrajectorySignals(delegated=True, overall_success=True),
        execution_reward=ExecutionReward(total=1.0),
    )


def _signal(index: int, tier: str) -> WeakLearningSignal:
    return WeakLearningSignal(
        signal_id=f"signal-{index}",
        created_at="2026-09-18T01:00:00+00:00",
        trajectory_id=f"trajectory-{index}",
        tier=tier,
        reward_score=1.0,
        training_eligible=tier == "gold",
    )


def test_replay_is_bounded_and_keeps_multiple_domains():
    trajectories = [
        _trajectory(1, "git_status"),
        _trajectory(2, "ticket_get"),
        _trajectory(3, "process_exec"),
        _trajectory(4, "git_log"),
    ]
    signals = [_signal(1, "gold"), _signal(2, "silver"), _signal(3, "bronze"), _signal(4, "gold")]
    selection = select_replay(
        trajectories=trajectories,
        signals=signals,
        policy=ReplayPolicy(max_records=3, historical_fraction=0.0, max_domain_fraction=0.67),
    )
    assert len(selection.selected_trajectory_ids) == 3
    assert len(selection.source_counts) >= 2
