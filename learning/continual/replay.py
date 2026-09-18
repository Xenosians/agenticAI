from __future__ import annotations

import hashlib
from collections import Counter
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from learning.continual.storage import canonical_json
from learning.continual.types import ReplaySelection, WeakLearningSignal
from learning.evidence.types import LearningTrajectory


class ReplayPolicy(BaseModel):
    max_records: int = Field(default=256, ge=1)
    historical_fraction: float = Field(default=0.35, ge=0.0, le=0.9)
    seed: str = "continual-replay-v1"
    max_domain_fraction: float = Field(default=0.60, gt=0.0, le=1.0)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _domain(trajectory: LearningTrajectory) -> str:
    values: list[str] = []
    values.extend(trajectory.routes)
    for step in trajectory.steps:
        values.append(step.agent)
        if step.proposed_tool:
            values.append(step.proposed_tool)

    joined = " ".join(values).casefold()
    if "ticket" in joined or "jira" in joined:
        return "jira"
    if "git" in joined or "repository" in joined:
        return "git"
    if "developer" in joined or "process_exec" in joined or "shell" in joined:
        return "shell"
    if "account" in joined or "access" in joined or "ldap" in joined:
        return "identity"
    return "other"


def _stable_rank(seed: str, trajectory_id: str) -> str:
    return hashlib.sha256(f"{seed}:{trajectory_id}".encode("utf-8")).hexdigest()


def select_replay(
    *,
    trajectories: list[LearningTrajectory],
    signals: list[WeakLearningSignal],
    policy: ReplayPolicy | None = None,
) -> ReplaySelection:
    policy = policy or ReplayPolicy()
    signal_index = {item.trajectory_id: item for item in signals}
    eligible = [item for item in trajectories if signal_index.get(item.trajectory_id) is not None and signal_index[item.trajectory_id].tier != "reject"]

    tier_weight = {"gold": 4, "silver": 3, "bronze": 1}
    eligible.sort(key=lambda item: item.observed_at)

    if not eligible:
        serialized = canonical_json([])
        return ReplaySelection(
            selected_at=_utc_now(),
            selected_trajectory_ids=[],
            source_counts={},
            tier_counts={},
            historical_count=0,
            recent_count=0,
            content_sha256=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        )

    recent_window_size = max(policy.max_records * 2, 32)
    recent_pool_ids = {item.trajectory_id for item in eligible[-recent_window_size:]}
    historical_pool = [item for item in eligible if item.trajectory_id not in recent_pool_ids]
    recent_pool = [item for item in eligible if item.trajectory_id in recent_pool_ids]

    def ranked(pool: list[LearningTrajectory]) -> list[LearningTrajectory]:
        return sorted(
            pool,
            key=lambda item: (
                -tier_weight.get(signal_index[item.trajectory_id].tier, 0),
                -signal_index[item.trajectory_id].reward_score,
                _stable_rank(policy.seed, item.trajectory_id),
            ),
        )

    historical_target = min(len(historical_pool), round(policy.max_records * policy.historical_fraction))
    recent_target = min(len(recent_pool), policy.max_records - historical_target)
    remaining = policy.max_records - historical_target - recent_target
    historical_target += min(remaining, max(0, len(historical_pool) - historical_target))
    remaining = policy.max_records - historical_target - recent_target
    recent_target += min(remaining, max(0, len(recent_pool) - recent_target))

    candidates = ranked(historical_pool)[:historical_target] + ranked(recent_pool)[:recent_target]
    candidates.sort(
        key=lambda item: (
            -tier_weight.get(signal_index[item.trajectory_id].tier, 0),
            _stable_rank(policy.seed, item.trajectory_id),
        )
    )

    selected: list[LearningTrajectory] = []
    domain_counts: Counter[str] = Counter()
    domain_limit = max(1, int(policy.max_records * policy.max_domain_fraction))

    deferred: list[LearningTrajectory] = []
    for item in candidates:
        domain = _domain(item)
        if domain_counts[domain] >= domain_limit:
            deferred.append(item)
            continue
        selected.append(item)
        domain_counts[domain] += 1
        if len(selected) >= policy.max_records:
            break

    for item in deferred:
        if len(selected) >= policy.max_records:
            break
        selected.append(item)
        domain_counts[_domain(item)] += 1

    selected_ids = [item.trajectory_id for item in selected]
    tier_counts = Counter(signal_index[item_id].tier for item_id in selected_ids)
    historical_count = sum(1 for item_id in selected_ids if item_id not in recent_pool_ids)
    recent_count = len(selected_ids) - historical_count
    serialized = canonical_json(selected_ids)

    return ReplaySelection(
        selected_at=_utc_now(),
        selected_trajectory_ids=selected_ids,
        source_counts=dict(domain_counts),
        tier_counts=dict(tier_counts),
        historical_count=historical_count,
        recent_count=recent_count,
        content_sha256=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    )
