from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime, timezone

from learning.curation.reviews import ReviewDecision
from learning.evidence.types import CorrectionEvent, LearningTrajectory
from learning.continual.types import WeakLearningSignal


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reward_score(trajectory: LearningTrajectory) -> float:
    """Operational score only; never treated as semantic correctness."""
    score = float(trajectory.execution_reward.total)
    quality = trajectory.quality
    if quality is not None:
        if quality.grounding_valid is True:
            score += 0.25
        elif quality.grounding_valid is False:
            score -= 1.0

        if quality.gateway_policy_passed is True:
            score += 0.25
        elif quality.gateway_policy_passed is False:
            score -= 1.0

        if quality.tool_execution_valid is True:
            score += 0.25
        elif quality.tool_execution_valid is False:
            score -= 1.0

    return max(-5.0, min(5.0, score))


def _review_map(reviews: list[ReviewDecision]) -> dict[tuple[str, str], ReviewDecision]:
    result: dict[tuple[str, str], ReviewDecision] = {}
    for review in reviews:
        result[(review.subject_type, review.subject_id)] = review
    return result


def classify_trajectory(
    *,
    trajectory: LearningTrajectory,
    corrections: list[CorrectionEvent],
    reviews: list[ReviewDecision],
) -> WeakLearningSignal:
    """
    Produce a conservative weak label.

    GOLD requires a trusted approval over evidence already marked eligible.
    SILVER is useful review/replay evidence only. Successful execution is not
    semantic truth and never becomes training data automatically.
    """

    review_index = _review_map(reviews)
    trajectory_review = review_index.get(("trajectory", trajectory.trajectory_id))
    related = [item for item in corrections if item.trajectory_id == trajectory.trajectory_id]
    correction_reviews = [
        review_index.get(("correction", item.correction_id))
        for item in related
    ]

    reasons: list[str] = []
    trusted_review_ids: list[str] = []

    if trajectory_review is not None:
        trusted_review_ids.append(trajectory_review.review_id)
        if trajectory_review.decision == "reject":
            return WeakLearningSignal(
                signal_id=f"signal-{uuid.uuid4().hex}",
                created_at=_utc_now(),
                trajectory_id=trajectory.trajectory_id,
                tier="reject",
                reward_score=_reward_score(trajectory),
                reasons=["trusted-review-rejected-trajectory"],
                correction_ids=[item.correction_id for item in related],
                trusted_review_ids=trusted_review_ids,
                training_eligible=False,
            )

    for review in correction_reviews:
        if review is not None:
            trusted_review_ids.append(review.review_id)

    approved_eligible_corrections = [
        correction
        for correction in related
        if correction.dataset_eligible
        and (review := review_index.get(("correction", correction.correction_id))) is not None
        and review.decision == "approve"
    ]

    trajectory_gold = (
        trajectory.dataset_eligible
        and trajectory_review is not None
        and trajectory_review.decision == "approve"
    )

    if trajectory_gold or approved_eligible_corrections:
        if trajectory_gold:
            reasons.append("trusted-review-approved-eligible-trajectory")
        if approved_eligible_corrections:
            reasons.append("trusted-review-approved-eligible-correction")
        return WeakLearningSignal(
            signal_id=f"signal-{uuid.uuid4().hex}",
            created_at=_utc_now(),
            trajectory_id=trajectory.trajectory_id,
            tier="gold",
            reward_score=_reward_score(trajectory),
            reasons=reasons,
            correction_ids=[item.correction_id for item in related],
            trusted_review_ids=sorted(set(trusted_review_ids)),
            training_eligible=True,
        )

    clean_success = (
        trajectory.signals.overall_success
        and not trajectory.signals.had_error
        and not trajectory.signals.waiting_approval
        and any(step.outcome_code == "success" for step in trajectory.steps)
    )

    if clean_success:
        reasons.extend([
            "deterministic-runtime-success",
            "runtime-success-is-weak-evidence-only",
        ])
        if related:
            reasons.append("has-unpromoted-correction-evidence")
        return WeakLearningSignal(
            signal_id=f"signal-{uuid.uuid4().hex}",
            created_at=_utc_now(),
            trajectory_id=trajectory.trajectory_id,
            tier="silver",
            reward_score=_reward_score(trajectory),
            reasons=reasons,
            correction_ids=[item.correction_id for item in related],
            trusted_review_ids=sorted(set(trusted_review_ids)),
            training_eligible=False,
        )

    reasons.append("insufficient-trusted-semantic-evidence")
    if trajectory.signals.had_error:
        reasons.append("runtime-error-observed")
    if trajectory.signals.waiting_approval:
        reasons.append("approval-pending-or-required")

    return WeakLearningSignal(
        signal_id=f"signal-{uuid.uuid4().hex}",
        created_at=_utc_now(),
        trajectory_id=trajectory.trajectory_id,
        tier="bronze",
        reward_score=_reward_score(trajectory),
        reasons=reasons,
        correction_ids=[item.correction_id for item in related],
        trusted_review_ids=sorted(set(trusted_review_ids)),
        training_eligible=False,
    )


def classify_trajectories(
    *,
    trajectories: list[LearningTrajectory],
    corrections: list[CorrectionEvent],
    reviews: list[ReviewDecision],
) -> list[WeakLearningSignal]:
    corrections_by_trajectory: dict[str, list[CorrectionEvent]] = defaultdict(list)
    for correction in corrections:
        corrections_by_trajectory[correction.trajectory_id].append(correction)

    return [
        classify_trajectory(
            trajectory=trajectory,
            corrections=corrections_by_trajectory.get(trajectory.trajectory_id, []),
            reviews=reviews,
        )
        for trajectory in trajectories
    ]
