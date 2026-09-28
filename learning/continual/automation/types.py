from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


EventState = Literal[
    "pending",
    "claimed",
    "consumed",
    "failed",
]

CorpusKind = Literal[
    "semantic_code_jsonl",
    "text",
]

TargetRole = Literal[
    "hub",
    "developer",
    "shared",
]

CycleState = Literal[
    "planned",
    "training",
    "evaluating",
    "blocked",
    "candidate",
    "accepted",
    "rejected",
    "interrupted",
    "failed",
]


class LearningEvent(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default="continuous-learning-event.v1",
        alias="schema",
    )

    event_id: str
    created_at: str

    trajectory_id: str
    job_id: str | None = None
    attempt: int | None = None

    model_key: str
    hub_status: str | None = None

    outcome_codes: list[str] = Field(
        default_factory=list
    )
    failure_types: list[str] = Field(
        default_factory=list
    )

    execution_reward: float = 0.0
    overall_success: bool = False
    had_error: bool = False
    waiting_approval: bool = False

    payload: dict[str, Any] = Field(
        default_factory=dict
    )

    training_eligible: bool = False


class CorpusSource(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default="continual-corpus-source.v1",
        alias="schema",
    )

    source_id: str
    kind: CorpusKind
    path: str

    target_role: TargetRole = "developer"
    audience: list[str] = Field(
        default_factory=list
    )

    trusted: bool = False
    training_eligible: bool = False

    page_chars: int = Field(
        default=4096,
        ge=512,
    )

    records_per_page: int = Field(
        default=1,
        ge=1,
    )


class CorpusPage(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default="continual-corpus-page.v1",
        alias="schema",
    )

    page_id: str
    source_id: str
    source_sha256: str

    page_index: int = Field(ge=0)
    cursor_start: int = Field(ge=0)
    cursor_end: int = Field(ge=0)

    target_role: TargetRole
    title: str
    content: str

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    replay: bool = False
    training_eligible: bool = False


class AdaptiveRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    learning_rate: float = Field(
        default=5e-6,
        gt=0.0,
    )
    max_optimizer_steps: int = Field(
        default=4,
        ge=1,
        le=128,
    )
    gradient_accumulation_steps: int = Field(
        default=2,
        ge=1,
        le=64,
    )
    lora_r: int = Field(
        default=16,
        ge=1,
        le=256,
    )
    corpus_pages_per_cycle: int = Field(
        default=10,
        ge=0,
        le=256,
    )
    replay_pages_per_cycle: int = Field(
        default=2,
        ge=0,
        le=128,
    )
    ppo_updates: int = Field(
        default=1,
        ge=0,
        le=16,
    )


class ContinualAutomationCyclePlan(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default="continual-automation-cycle-plan.v1",
        alias="schema",
    )

    cycle_id: str
    created_at: str
    state: CycleState = "planned"

    event_ids: list[str] = Field(
        default_factory=list
    )
    corpus_page_ids: list[str] = Field(
        default_factory=list
    )

    seed_checkpoint_id: str | None = None
    behavior_materialization_id: str | None = None

    controls: dict[str, bool] = Field(
        default_factory=dict
    )
    corpus_target_roles: list[str] = Field(
        default_factory=list
    )
    target_coverage: list[dict[str, Any]] = Field(
        default_factory=list
    )

    stages: list[str] = Field(
        default_factory=list
    )

    recipe: AdaptiveRecipe


class ContinualAutomationCycleResult(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default="continual-automation-cycle-result.v1",
        alias="schema",
    )

    cycle_id: str
    completed_at: str
    state: CycleState

    training_started: bool = False
    training_executed: bool | None = False
    evaluation_executed: bool = False
    promotion_attempted: bool = False

    corpus_target_roles: list[str] = Field(
        default_factory=list
    )
    target_coverage: list[dict[str, Any]] = Field(
        default_factory=list
    )
    blocked_reasons: list[str] = Field(
        default_factory=list
    )

    candidate_checkpoint_id: str | None = None

    baseline_intelligence_report_id: str | None = None
    candidate_intelligence_report_id: str | None = None
    baseline_safety_report_id: str | None = None
    candidate_safety_report_id: str | None = None

    promotion_decision_id: str | None = None
    promotion_eligible: bool = False
    improvement_observed: bool = False

    activated: bool = False
    rolled_back: bool = False

    guard_signal: str = "none"
    notes: list[str] = Field(
        default_factory=list
    )
