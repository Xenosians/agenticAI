from __future__ import annotations

import hashlib
from collections import Counter

from pydantic import BaseModel, ConfigDict, Field


class CurriculumCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trajectory_id: str
    tier: str
    primary_domain: str
    domains: list[str] = Field(default_factory=list)
    difficulty: str
    difficulty_score: int
    novelty_key: str
    failure_clusters: list[str] = Field(default_factory=list)


class CurriculumSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_trajectory_ids: list[str] = Field(default_factory=list)
    domain_counts: dict[str, int] = Field(default_factory=dict)
    difficulty_counts: dict[str, int] = Field(default_factory=dict)
    failure_cluster_counts: dict[str, int] = Field(default_factory=dict)
    novelty_count: int = 0


def stable_novelty_key(*parts: str) -> str:
    material = "\x1f".join(parts)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def select_curriculum(
    candidates: list[CurriculumCandidate],
    *,
    max_records: int = 256,
    max_domain_fraction: float = 0.60,
    seed: str = "curriculum-v1",
) -> CurriculumSelection:
    if max_records < 1:
        raise ValueError("max_records must be positive")
    if not 0.0 < max_domain_fraction <= 1.0:
        raise ValueError("max_domain_fraction must be in (0, 1]")

    tier_weight = {"gold": 4, "silver": 3, "bronze": 2, "reject": 0}
    difficulty_weight = {"hard": 3, "medium": 2, "easy": 1}

    def rank(item: CurriculumCandidate) -> tuple:
        tie = hashlib.sha256(f"{seed}:{item.trajectory_id}".encode("utf-8")).hexdigest()
        return (
            -tier_weight.get(item.tier, 0),
            -difficulty_weight.get(item.difficulty, 0),
            tie,
        )

    eligible = [item for item in candidates if item.tier != "reject"]
    eligible.sort(key=rank)

    domain_limit = max(1, int(max_records * max_domain_fraction))
    selected: list[CurriculumCandidate] = []
    deferred: list[CurriculumCandidate] = []
    domain_counts: Counter[str] = Counter()

    for item in eligible:
        if len(selected) >= max_records:
            break
        if domain_counts[item.primary_domain] >= domain_limit:
            deferred.append(item)
            continue
        selected.append(item)
        domain_counts[item.primary_domain] += 1

    for item in deferred:
        if len(selected) >= max_records:
            break
        selected.append(item)
        domain_counts[item.primary_domain] += 1

    difficulty_counts = Counter(item.difficulty for item in selected)
    failure_counts: Counter[str] = Counter()
    for item in selected:
        failure_counts.update(item.failure_clusters)

    novelty_count = len({item.novelty_key for item in selected})
    return CurriculumSelection(
        selected_trajectory_ids=[item.trajectory_id for item in selected],
        domain_counts=dict(domain_counts),
        difficulty_counts=dict(difficulty_counts),
        failure_cluster_counts=dict(failure_counts),
        novelty_count=novelty_count,
    )
