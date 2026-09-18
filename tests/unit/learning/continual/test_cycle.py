import json
from pathlib import Path

from learning.continual.cycle import ContinualLearningCycle, ContinualLearningPolicy
from learning.curation.reviews import ReviewDecision
from learning.evidence.types import ExecutionReward, LearningTrajectory, TrajectorySignals, TrajectoryStep


def _write_jsonl(path: Path, payloads):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(payload, sort_keys=True) + "\n" for payload in payloads), encoding="utf-8")


def test_cycle_never_executes_training(tmp_path: Path):
    trajectory = LearningTrajectory(
        trajectory_id="trajectory-1",
        observed_at="2026-09-18T00:00:00+00:00",
        job_id="job-1",
        attempt=1,
        user_request="Show git status",
        hub_model="hub-main",
        hub_status="success",
        routes=["developer-specialist"],
        steps=[TrajectoryStep(task_id="task-1", agent="developer-specialist", status="success", outcome_code="success", proposed_tool="git_status", proposed_arguments={"repository": "ai"})],
        signals=TrajectorySignals(delegated=True, overall_success=True),
        execution_reward=ExecutionReward(total=1.0),
        dataset_eligible=True,
    )
    review = ReviewDecision(
        review_id="review-1",
        observed_at="2026-09-18T00:01:00+00:00",
        subject_type="trajectory",
        subject_id="trajectory-1",
        decision="approve",
        source="trusted_review",
        reason="verified",
    )

    trajectories = tmp_path / "trajectories.jsonl"
    corrections = tmp_path / "corrections.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    context = tmp_path / "context.jsonl"
    _write_jsonl(trajectories, [trajectory.model_dump(mode="json", by_alias=True)])
    _write_jsonl(corrections, [])
    _write_jsonl(reviews, [review.model_dump(mode="json", by_alias=True)])
    _write_jsonl(context, [])

    manifest = ContinualLearningCycle(
        trajectories_path=trajectories,
        corrections_path=corrections,
        reviews_path=reviews,
        context_path=context,
        output_root=tmp_path / "cycles",
        policy=ContinualLearningPolicy(min_gold_records=1, min_recent_gold_records=1, min_gold_domains=1, min_novelty_fraction=0.0, min_failure_clusters=0, max_replay_records=8, max_domain_fraction=1.0),
    ).run()

    assert manifest.train_ready is True
    assert manifest.training_executed is False
    assert manifest.auto_promotion_performed is False
