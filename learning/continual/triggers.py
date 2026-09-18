from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, ConfigDict, Field


class TrainingTriggerPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_gold_records: int = Field(default=16, ge=1)
    min_recent_gold_records: int = Field(default=4, ge=0)
    min_gold_domains: int = Field(default=2, ge=1)
    min_novelty_fraction: float = Field(default=0.25, ge=0.0, le=1.0)
    min_failure_clusters: int = Field(default=1, ge=0)
    max_domain_fraction: float = Field(default=0.60, gt=0.0, le=1.0)


class TrainingTriggerDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ready: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    gold_record_count: int
    recent_gold_record_count: int
    gold_domain_count: int
    novelty_fraction: float
    failure_cluster_count: int
    dominant_domain_fraction: float


def evaluate_training_trigger(
    *,
    signals: list,
    curriculum_plan,
    recent_trajectory_ids: set[str],
    policy: TrainingTriggerPolicy,
) -> TrainingTriggerDecision:
    candidate_index = {item.trajectory_id: item for item in curriculum_plan.candidates}
    gold_ids = [
        item.trajectory_id
        for item in signals
        if item.tier == "gold" and item.training_eligible
    ]
    recent_gold = [item_id for item_id in gold_ids if item_id in recent_trajectory_ids]
    gold_domains = {
        candidate_index[item_id].primary_domain
        for item_id in gold_ids
        if item_id in candidate_index
    }

    selected_ids = curriculum_plan.selection.selected_trajectory_ids
    selected = [candidate_index[item_id] for item_id in selected_ids if item_id in candidate_index]
    novelty_fraction = (
        len({item.novelty_key for item in selected}) / len(selected)
        if selected
        else 0.0
    )

    failure_clusters = {
        cluster
        for item in selected
        for cluster in item.failure_clusters
    }

    domain_counts = Counter(item.primary_domain for item in selected)
    dominant_domain_fraction = (
        max(domain_counts.values()) / len(selected)
        if selected
        else 1.0
    )

    checks = {
        "minimum_gold_records": len(gold_ids) >= policy.min_gold_records,
        "minimum_recent_gold_records": len(recent_gold) >= policy.min_recent_gold_records,
        "minimum_gold_domains": len(gold_domains) >= policy.min_gold_domains,
        "minimum_novelty": novelty_fraction >= policy.min_novelty_fraction,
        "minimum_failure_clusters": len(failure_clusters) >= policy.min_failure_clusters,
        "domain_balance": dominant_domain_fraction <= policy.max_domain_fraction,
    }
    reasons = [name for name, passed in checks.items() if not passed]
    return TrainingTriggerDecision(
        ready=all(checks.values()),
        checks=checks,
        reasons=reasons,
        gold_record_count=len(gold_ids),
        recent_gold_record_count=len(recent_gold),
        gold_domain_count=len(gold_domains),
        novelty_fraction=novelty_fraction,
        failure_cluster_count=len(failure_clusters),
        dominant_domain_fraction=dominant_domain_fraction,
    )
