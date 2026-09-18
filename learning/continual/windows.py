from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EvidenceWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recent_trajectory_ids: list[str] = Field(default_factory=list)
    historical_trajectory_ids: list[str] = Field(default_factory=list)
    recent_count: int = 0
    historical_count: int = 0


def build_evidence_window(
    trajectories: list,
    *,
    recent_size: int = 64,
    historical_size: int = 192,
) -> EvidenceWindow:
    if recent_size < 1 or historical_size < 0:
        raise ValueError("invalid evidence window sizes")

    ids = [item.trajectory_id for item in trajectories]
    recent = ids[-recent_size:]
    historical_pool = ids[: max(0, len(ids) - len(recent))]
    historical = historical_pool[-historical_size:] if historical_size else []

    return EvidenceWindow(
        recent_trajectory_ids=recent,
        historical_trajectory_ids=historical,
        recent_count=len(recent),
        historical_count=len(historical),
    )
