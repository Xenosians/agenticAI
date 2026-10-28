"""Language-neutral, path-free requests and trusted resource metadata."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Protocol

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

Identifier = Annotated[str, Field(strict=True, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")]
Digest = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
Method = Literal["sft", "dpo", "ppo"]
Role = Literal["hub", "specialist"]
Tuning = Literal["full", "lora", "qlora"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class TrainingIdentity(Contract):
    model_sha256: Digest
    tokenizer_sha256: Digest
    prompt_sha256: Digest
    catalog_sha256: Digest


class SourceReview(Contract):
    source_id: Identifier
    kind: Literal["dataset", "web", "correction"]
    license: str = Field(max_length=200)
    reviewed: bool
    training_eligible: bool
    reference_only: bool = False
    review_id: Identifier | None = None


class LearningArtifact(Contract):
    """Metadata returned by a trusted owner-scoped registry, never a user body.

    Hashes and review flags are metadata attestations. The existing trainer must
    still verify actual files, review ledgers, membership and provenance.
    """

    resource_id: Identifier
    owner_id: str = Field(min_length=1, max_length=200)
    kind: Literal[
        "model", "adaptation_recipe", "dpo_settings", "ppo_settings",
        "preference_train", "preference_validation", "adapter_checkpoint",
        "simulator_episodes", "evaluation_suite", "trajectory_ledger",
        "correction_ledger", "review_ledger", "context_ledger",
    ]
    sha256: Digest
    identity: TrainingIdentity
    method: Method | None = None
    role: Role | None = None
    tuning: Tuning | None = None
    sources: tuple[SourceReview, ...] = Field(default=(), max_length=256)
    record_sha256s: tuple[Digest, ...] = Field(default=(), max_length=100_000)


class ArtifactResolver(Protocol):
    def resolve(self, owner_id: str, resource_id: str) -> LearningArtifact:
        """Resolve a currently authorized immutable artifact for this owner."""


class LearningPlanRequest(Contract):
    method: Method
    role: Role
    tuning: Tuning
    # Exact keys are method-specific. Values are IDs; paths and owner overrides
    # cannot enter the public request contract.
    inputs: dict[str, Identifier] = Field(min_length=1, max_length=8)


class ResourceObservation(Contract):
    """Measured/preflight metadata supplied separately by the trusted worker."""

    method: Method
    role: Role
    tuning: Tuning
    model_sha256: Digest
    configuration_sha256: Digest
    observed_at: AwareDatetime
    available_free_vram_gib: float = Field(ge=0, allow_inf_nan=False)
    required_free_vram_gib: float = Field(gt=0, allow_inf_nan=False)


class EvidenceCycleRequest(Contract):
    # The existing evidence-cycle CLI consumes these ledgers. Web collection,
    # dataset imports and corrections must enter their existing admission paths.
    inputs: dict[str, Identifier] = Field(min_length=4, max_length=4)


class ArtifactBinding(Contract):
    resource_id: Identifier
    sha256: Digest
    membership_sha256: Digest


class PreparedLearningPlan(Contract):
    schema_version: Literal["learning-candidate-plan.v1"] = "learning-candidate-plan.v1"
    status: Literal["prepared", "blocked"]
    plan_sha256: Digest
    method: Method
    role: Role
    tuning: Tuning
    entrypoint: str
    inputs: dict[str, ArtifactBinding]
    identity: TrainingIdentity
    resource_state: Literal["unknown", "stale", "insufficient", "observed_sufficient"]
    blockers: tuple[str, ...]
    execution_enabled: Literal[False] = False
    model_loaded: Literal[False] = False
    live_tool_execution: Literal[False] = False
    promotion_enabled: Literal[False] = False


class PreparedEvidenceCycle(Contract):
    schema_version: Literal["learning-evidence-plan.v1"] = "learning-evidence-plan.v1"
    status: Literal["prepared"] = "prepared"
    plan_sha256: Digest
    entrypoint: Literal["learning/cli/run_continual_cycle.py"] = "learning/cli/run_continual_cycle.py"
    inputs: dict[str, ArtifactBinding]
    blockers: tuple[str, ...] = ("host_dispatch_not_integrated", "curation_and_review_required")
    execution_enabled: Literal[False] = False
    training_enabled: Literal[False] = False
    automatic_promotion: Literal[False] = False
    web_collection_enabled: Literal[False] = False
