from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from learning.context.models import LearningContextBundle, StructuredContextRecord


ContextSource = Literal["git", "jira", "shell", "human", "task", "tool"]
LearningTier = Literal["gold", "silver", "bronze", "reject"]
ContextEvent = StructuredContextRecord
ContextBundle = LearningContextBundle


class WeakLearningSignal(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_name: str = Field(default="weak-learning-signal.v1", alias="schema")
    signal_id: str
    created_at: str
    trajectory_id: str
    tier: LearningTier
    reward_score: float
    reasons: list[str] = Field(default_factory=list)
    correction_ids: list[str] = Field(default_factory=list)
    trusted_review_ids: list[str] = Field(default_factory=list)
    training_eligible: bool = False


class ReplaySelection(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_name: str = Field(default="continual-replay-selection.v1", alias="schema")
    selected_at: str
    selected_trajectory_ids: list[str] = Field(default_factory=list)
    source_counts: dict[str, int] = Field(default_factory=dict)
    tier_counts: dict[str, int] = Field(default_factory=dict)
    historical_count: int = 0
    recent_count: int = 0
    content_sha256: str


class ContinualCycleManifest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_name: str = Field(default="continual-learning-cycle.v2", alias="schema")
    cycle_id: str
    created_at: str
    output_directory: str
    trajectory_sha256: str | None = None
    corrections_sha256: str | None = None
    reviews_sha256: str | None = None
    context_sha256: str | None = None
    trajectory_count: int
    correction_count: int
    review_count: int
    context_event_count: int = 0
    context_record_count: int = 0
    contextualized_trajectory_count: int = 0
    tier_counts: dict[str, int] = Field(default_factory=dict)
    selected_replay_count: int
    selected_domain_counts: dict[str, int] = Field(default_factory=dict)
    curriculum_difficulty_counts: dict[str, int] = Field(default_factory=dict)
    failure_cluster_counts: dict[str, int] = Field(default_factory=dict)
    recent_window_count: int = 0
    historical_window_count: int = 0
    min_gold_records: int
    min_gold_domains: int
    train_ready: bool
    trigger_checks: dict[str, bool] = Field(default_factory=dict)
    train_block_reasons: list[str] = Field(default_factory=list)
    review_queue_count: int
    dpo_candidate_count: int = 0
    sft_candidate_count: int = 0
    recommended_action: str
    training_executed: bool = False
    auto_promotion_performed: bool = False


class AdapterCheckpointManifest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_name: str = Field(default="adapter-checkpoint.v1", alias="schema")
    checkpoint_id: str
    created_at: str
    label: str
    adapter_directory: str
    adapter_sha256: str
    base_model_sha256: str
    source_cycle_id: str
    source_split_id: str
    target_agent: str
    target_model_key: str
    promotion_suite: str | None = None
    promotion_decision_id: str | None = None
    promotion_decision_sha256: str | None = None
    promotion_eligible: bool = False


class ActiveCheckpointPointer(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_name: str = Field(default="active-adapter-checkpoint.v1", alias="schema")
    updated_at: str
    checkpoint_id: str
    previous_checkpoint_id: str | None = None
    promotion_decision_id: str
    rollback: bool = False
