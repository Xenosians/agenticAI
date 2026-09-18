from learning.continual.weak_supervision import classify_trajectory
from learning.curation.reviews import ReviewDecision
from learning.evidence.types import (
    ExecutionReward,
    LearningTrajectory,
    TrajectorySignals,
    TrajectoryStep,
)


def _trajectory(*, eligible: bool = False, success: bool = True) -> LearningTrajectory:
    return LearningTrajectory(
        trajectory_id="trajectory-1",
        observed_at="2026-09-18T00:00:00+00:00",
        job_id="job-1",
        attempt=1,
        user_request="Show git status",
        hub_model="hub-main",
        hub_status="success" if success else "partial_error",
        routes=["developer-specialist"],
        steps=[
            TrajectoryStep(
                task_id="task-1",
                agent="developer-specialist",
                status="success" if success else "error",
                outcome_code="success" if success else "tool_execution_error",
                proposed_tool="git_status",
                proposed_arguments={"repository": "ai"},
            )
        ],
        signals=TrajectorySignals(
            delegated=True,
            route_count=1,
            specialist_count=1,
            specialist_success_count=1 if success else 0,
            specialist_error_count=0 if success else 1,
            tool_proposed_count=1,
            tool_success_count=1 if success else 0,
            overall_success=success,
            had_error=not success,
            waiting_approval=False,
        ),
        execution_reward=ExecutionReward(components={"hub_success": 1.0} if success else {"hub_error": -1.0}, total=1.0 if success else -1.0),
        dataset_eligible=eligible,
    )


def test_success_is_silver_not_automatic_training_data():
    signal = classify_trajectory(trajectory=_trajectory(), corrections=[], reviews=[])
    assert signal.tier == "silver"
    assert signal.training_eligible is False


def test_trusted_approved_eligible_trajectory_is_gold():
    review = ReviewDecision(
        review_id="review-1",
        observed_at="2026-09-18T00:01:00+00:00",
        subject_type="trajectory",
        subject_id="trajectory-1",
        decision="approve",
        source="trusted_review",
        reason="Human verified behavior.",
    )
    signal = classify_trajectory(trajectory=_trajectory(eligible=True), corrections=[], reviews=[review])
    assert signal.tier == "gold"
    assert signal.training_eligible is True


def test_trusted_rejection_wins():
    review = ReviewDecision(
        review_id="review-2",
        observed_at="2026-09-18T00:01:00+00:00",
        subject_type="trajectory",
        subject_id="trajectory-1",
        decision="reject",
        source="trusted_review",
        reason="Wrong semantic choice.",
    )
    signal = classify_trajectory(trajectory=_trajectory(eligible=True), corrections=[], reviews=[review])
    assert signal.tier == "reject"
    assert signal.training_eligible is False
