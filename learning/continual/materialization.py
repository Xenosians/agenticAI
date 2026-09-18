from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel, ConfigDict, Field


class TrainingMaterializationPlan(BaseModel):
    """Planning artifact only. It does not create a trainer or update weights."""

    model_config = ConfigDict(extra="forbid")

    dpo_candidate_trajectory_ids: list[str] = Field(default_factory=list)
    sft_candidate_trajectory_ids: list[str] = Field(default_factory=list)
    review_only_trajectory_ids: list[str] = Field(default_factory=list)
    rejected_trajectory_ids: list[str] = Field(default_factory=list)


def build_materialization_plan(*, signals: list, corrections: list) -> TrainingMaterializationPlan:
    corrections_by_trajectory: dict[str, list] = defaultdict(list)
    for correction in corrections:
        corrections_by_trajectory[correction.trajectory_id].append(correction)

    dpo: list[str] = []
    sft: list[str] = []
    review: list[str] = []
    rejected: list[str] = []

    for signal in signals:
        if signal.tier == "reject":
            rejected.append(signal.trajectory_id)
            continue
        if signal.tier != "gold" or not signal.training_eligible:
            review.append(signal.trajectory_id)
            continue
        if corrections_by_trajectory.get(signal.trajectory_id):
            dpo.append(signal.trajectory_id)
        else:
            # Reviewed successful behavior can become an SFT candidate, but
            # downstream dataset promotion/provenance gates must still decide
            # whether a concrete SFT dataset may be produced.
            sft.append(signal.trajectory_id)

    return TrainingMaterializationPlan(
        dpo_candidate_trajectory_ids=dpo,
        sft_candidate_trajectory_ids=sft,
        review_only_trajectory_ids=review,
        rejected_trajectory_ids=rejected,
    )
