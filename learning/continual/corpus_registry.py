from __future__ import annotations

import json

from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


CorpusProvider = Literal[
    "local",
    "huggingface",
    "runtime",
    "generated",
]

CorpusObjective = Literal[
    "retrieval",
    "sft",
    "dpo",
    "ppo",
    "evaluation",
]

CorpusTrust = Literal[
    "untrusted",
    "curated",
    "verified",
]

MaterializationMode = Literal[
    "manual",
    "stream",
    "local",
    "runtime",
]


class CorpusSamplingPolicy(BaseModel):
    """
    Per-source admission budget.

    This intentionally describes how much MAY be admitted into one
    continual-learning cycle. It does not imply that the full upstream
    dataset is downloaded or trained.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    priority: int = Field(
        default=50,
        ge=0,
        le=100,
    )

    max_records_per_cycle: int = Field(
        default=32,
        ge=0,
    )

    max_records_per_snapshot: int = Field(
        default=5000,
        ge=0,
    )

    replay_weight: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
    )

    max_cycle_fraction: float = Field(
        default=0.25,
        gt=0.0,
        le=1.0,
    )


class CorpusFilterPolicy(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    languages: list[str] = Field(
        default_factory=list
    )

    include_tasks: list[str] = Field(
        default_factory=list
    )

    exclude_tasks: list[str] = Field(
        default_factory=list
    )

    include_paths: list[str] = Field(
        default_factory=list
    )

    exclude_paths: list[str] = Field(
        default_factory=list
    )

    require_verified_outcome: bool = False

    require_permissive_source_license: bool = False

    deduplicate: bool = True

    secret_scan: bool = True

    benchmark_decontamination: bool = True


class CorpusRegistrySource(BaseModel):
    """
    Declaration of an upstream knowledge/training source.

    Registration is NOT authorization to train.

    External sources remain inert until separately materialized into a
    governed local snapshot with provenance, filtering and a source hash.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default="continual-corpus-registry-source.v1",
        alias="schema",
    )

    source_id: str

    provider: CorpusProvider

    dataset_id: str | None = None
    subset: str | None = None
    revision: str | None = None

    local_path: str | None = None

    license: str | None = None
    upstream_url: str | None = None

    target_component: str

    audience: list[str] = Field(
        default_factory=list
    )

    objectives: list[CorpusObjective] = Field(
        default_factory=list
    )

    trust: CorpusTrust = "untrusted"

    enabled: bool = False

    training_eligible: bool = False

    evaluation_only: bool = False

    verified_reward: bool = False

    materialization_mode: MaterializationMode = "manual"

    contamination_group: str | None = None

    sampling: CorpusSamplingPolicy = Field(
        default_factory=CorpusSamplingPolicy
    )

    filters: CorpusFilterPolicy = Field(
        default_factory=CorpusFilterPolicy
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_source(
        self,
    ) -> "CorpusRegistrySource":
        if (
            self.provider == "huggingface"
            and not self.dataset_id
        ):
            raise ValueError(
                "Hugging Face corpus sources require dataset_id."
            )

        if (
            self.provider == "local"
            and not self.local_path
        ):
            raise ValueError(
                "Local corpus sources require local_path."
            )

        if (
            self.evaluation_only
            and self.training_eligible
        ):
            raise ValueError(
                "Evaluation-only corpus cannot be training eligible."
            )

        if (
            "evaluation" in self.objectives
            and self.training_eligible
            and self.evaluation_only
        ):
            raise ValueError(
                "Held-out evaluation sources cannot enter training."
            )

        # PPO in this project is environment-grounded RL.
        #
        # Static web datasets can provide SFT or DPO examples, reward-model
        # material, or evaluation examples. They are not direct PPO rollout
        # sources.
        if (
            "ppo" in self.objectives
            and self.provider != "runtime"
        ):
            raise ValueError(
                "Direct PPO sources must come from verified runtime "
                "environment outcomes."
            )

        if (
            "ppo" in self.objectives
            and not self.verified_reward
        ):
            raise ValueError(
                "PPO sources require verified_reward=true."
            )

        if (
            self.training_eligible
            and self.trust == "untrusted"
        ):
            raise ValueError(
                "Training-eligible corpus must be curated or verified."
            )

        return self


class ContinualCorpusMix(BaseModel):
    """
    Global cycle-level curriculum limits.

    External corpora are deliberately bounded so user/runtime evidence and
    replay cannot be drowned out by a giant public dataset.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    max_external_fraction: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
    )

    target_runtime_fraction: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
    )

    target_replay_fraction: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def validate_mix(
        self,
    ) -> "ContinualCorpusMix":
        total = (
            self.target_runtime_fraction
            + self.target_replay_fraction
        )

        if total > 1.0:
            raise ValueError(
                "Runtime + replay target fractions cannot exceed 1.0."
            )

        return self


class ContinualCorpusRegistry(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default="continual-corpus-registry.v1",
        alias="schema",
    )

    mix: ContinualCorpusMix = Field(
        default_factory=ContinualCorpusMix
    )

    sources: list[CorpusRegistrySource] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_registry(
        self,
    ) -> "ContinualCorpusRegistry":
        ids = [
            source.source_id
            for source in self.sources
        ]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "Corpus registry source_id values must be unique."
            )

        return self

    def enabled_sources(
        self,
    ) -> list[CorpusRegistrySource]:
        return [
            source
            for source in self.sources
            if source.enabled
        ]

    def training_sources(
        self,
        *,
        objective: CorpusObjective | None = None,
    ) -> list[CorpusRegistrySource]:
        result = []

        for source in self.sources:
            if not (
                source.enabled
                and source.training_eligible
                and not source.evaluation_only
            ):
                continue

            if (
                objective is not None
                and objective not in source.objectives
            ):
                continue

            result.append(source)

        return result

    def evaluation_sources(
        self,
    ) -> list[CorpusRegistrySource]:
        return [
            source
            for source in self.sources
            if (
                source.enabled
                and (
                    source.evaluation_only
                    or "evaluation"
                    in source.objectives
                )
            )
        ]


def load_corpus_registry(
    path: Path,
) -> ContinualCorpusRegistry:
    path = (
        path
        .expanduser()
        .resolve()
    )

    if not path.is_file():
        return ContinualCorpusRegistry()

    raw = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    return (
        ContinualCorpusRegistry
        .model_validate(
            raw
        )
    )