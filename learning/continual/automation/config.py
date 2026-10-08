from __future__ import annotations

import json
import os

from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from learning.continual.automation.types import (
    AdaptiveRecipe,
    CorpusSource,
)

from learning.continual.corpus_registry import (
    load_corpus_registry,
)

from learning.paths import (
    REPOSITORY_ROOT,
    RUNTIME_LEARNING_ROOT,
)


class ContinualAutomationSettings(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    enabled: bool = False

    # Master switch for unattended continual learning.
    #
    # This enables orchestration, not unrestricted authority.
    # Evaluation, promotion, shared-model coverage checks,
    # canary rollback, and GPU admission remain authoritative.
    autonomous_mode: bool = False

    poll_seconds: float = Field(
        default=5.0,
        ge=0.5,
    )

    idle_seconds_before_training: float = Field(
        default=90.0,
        ge=0.0,
    )

    min_events_per_cycle: int = Field(
        default=4,
        ge=1,
    )

    max_events_per_cycle: int = Field(
        default=64,
        ge=1,
    )

    failure_events_trigger: int = Field(
        default=2,
        ge=1,
    )

    max_cycle_age_seconds: float = Field(
        default=1800.0,
        ge=60.0,
    )

    # Evidence collection may run unattended.
    # Optimizer/evaluation/promotion stages are independent opt-ins.
    auto_train: bool = False
    auto_evaluate: bool = False
    auto_promote: bool = False

    require_improvement_for_auto_promote: bool = True

    # PPO is environment-grounded and separately authorized.
    ppo_enabled: bool = False

    ppo_every_n_cycles: int = Field(
        default=4,
        ge=1,
    )

    ppo_suite: str = "router-training-v1"
    eval_suite: str = "core.v1"

    single_gpu_idle_only: bool = True

    # Keep the previous physical runtime location for backward-compatible
    # recovery of any already-persisted state. The Python namespace itself
    # is now consistently "continual".
    state_db_path: Path = (
        RUNTIME_LEARNING_ROOT
        / "continuous"
        / "state.sqlite3"
    )

    corpus_registry_path: Path = (
        REPOSITORY_ROOT
        / "config"
        / "continual_corpus_registry.json"
    )

    # Backward compatibility only.
    #
    # Older tests/deployments may still explicitly provide a legacy
    # list-style corpus source file. New code should use
    # corpus_registry_path instead.
    corpus_sources_path: Path | None = None

    recipe: AdaptiveRecipe = Field(
        default_factory=AdaptiveRecipe
    )

    @model_validator(
        mode="before"
    )
    @classmethod
    def expand_autonomous_mode(
        cls,
        values,
    ):

        if not isinstance(
            values,
            dict,
        ):

            return values

        if not values.get(
            "autonomous_mode",
            False,
        ):

            return values

        expanded = dict(
            values
        )

        for key in (
            "enabled",
            "auto_train",
            "auto_evaluate",
            "auto_promote",
            "ppo_enabled",
        ):

            expanded[
                key
            ] = True

        return expanded

    @model_validator(
        mode="after"
    )
    def validate_stage_controls(
        self,
    ) -> "ContinualAutomationSettings":

        if (
            self.auto_promote
            and not self.auto_evaluate
        ):
            raise ValueError(
                "CONTINUAL_AUTO_PROMOTE requires "
                "CONTINUAL_AUTO_EVALUATE."
            )

        if (
            self.auto_promote
            and not self.auto_train
        ):
            raise ValueError(
                "CONTINUAL_AUTO_PROMOTE requires "
                "CONTINUAL_AUTO_TRAIN."
            )

        if (
            self.ppo_enabled
            and not self.auto_train
        ):
            raise ValueError(
                "CONTINUAL_PPO_ENABLED requires "
                "CONTINUAL_AUTO_TRAIN."
            )

        return self

    @staticmethod
    def _bool(
        name: str,
        default: bool,
    ) -> bool:

        value = os.getenv(
            name
        )

        if value is None:
            return default

        return (
            value.strip().lower()
            in {
                "1",
                "true",
                "yes",
                "on",
            }
        )

    @staticmethod
    def _path_env(
        primary: str,
        *,
        fallback: str | None,
        default: Path,
    ) -> Path:

        value = os.getenv(
            primary
        )

        if (
            value is None
            and fallback is not None
        ):
            value = os.getenv(
                fallback
            )

        if value:
            return Path(
                value
            )

        return default

    @classmethod
    def from_env(
        cls,
    ) -> "ContinualAutomationSettings":

        base = cls()

        autonomous_mode = cls._bool(
            "CONTINUAL_AUTONOMOUS_MODE",
            base.autonomous_mode,
        )

        recipe = AdaptiveRecipe(
            learning_rate=float(
                os.getenv(
                    "CONTINUAL_LEARNING_RATE",
                    str(
                        base.recipe.learning_rate
                    ),
                )
            ),

            max_optimizer_steps=int(
                os.getenv(
                    "CONTINUAL_MAX_OPTIMIZER_STEPS",
                    str(
                        base.recipe.max_optimizer_steps
                    ),
                )
            ),

            gradient_accumulation_steps=int(
                os.getenv(
                    "CONTINUAL_GRADIENT_ACCUMULATION",
                    str(
                        base.recipe.gradient_accumulation_steps
                    ),
                )
            ),

            lora_r=int(
                os.getenv(
                    "CONTINUAL_LORA_R",
                    str(
                        base.recipe.lora_r
                    ),
                )
            ),

            corpus_pages_per_cycle=int(
                os.getenv(
                    "CONTINUAL_CORPUS_PAGES_PER_CYCLE",
                    str(
                        base.recipe.corpus_pages_per_cycle
                    ),
                )
            ),

            replay_pages_per_cycle=int(
                os.getenv(
                    "CONTINUAL_CORPUS_REPLAY_PAGES",
                    str(
                        base.recipe.replay_pages_per_cycle
                    ),
                )
            ),

            ppo_updates=int(
                os.getenv(
                    "CONTINUAL_PPO_UPDATES",
                    str(
                        base.recipe.ppo_updates
                    ),
                )
            ),
        )

        return cls(
            autonomous_mode=(
                autonomous_mode
            ),

            enabled=cls._bool(
                "CONTINUAL_LEARNING_ENABLED",
                base.enabled,
            ),

            poll_seconds=float(
                os.getenv(
                    "CONTINUAL_POLL_SECONDS",
                    str(
                        base.poll_seconds
                    ),
                )
            ),

            idle_seconds_before_training=float(
                os.getenv(
                    "CONTINUAL_IDLE_SECONDS",
                    str(
                        base.idle_seconds_before_training
                    ),
                )
            ),

            min_events_per_cycle=int(
                os.getenv(
                    "CONTINUAL_MIN_EVENTS",
                    str(
                        base.min_events_per_cycle
                    ),
                )
            ),

            max_events_per_cycle=int(
                os.getenv(
                    "CONTINUAL_MAX_EVENTS",
                    str(
                        base.max_events_per_cycle
                    ),
                )
            ),

            failure_events_trigger=int(
                os.getenv(
                    "CONTINUAL_FAILURE_TRIGGER",
                    str(
                        base.failure_events_trigger
                    ),
                )
            ),

            max_cycle_age_seconds=float(
                os.getenv(
                    "CONTINUAL_MAX_CYCLE_AGE_SECONDS",
                    str(
                        base.max_cycle_age_seconds
                    ),
                )
            ),

            auto_train=cls._bool(
                "CONTINUAL_AUTO_TRAIN",
                base.auto_train,
            ),

            auto_evaluate=cls._bool(
                "CONTINUAL_AUTO_EVALUATE",
                base.auto_evaluate,
            ),

            auto_promote=cls._bool(
                "CONTINUAL_AUTO_PROMOTE",
                base.auto_promote,
            ),

            require_improvement_for_auto_promote=(
                cls._bool(
                    "CONTINUAL_REQUIRE_IMPROVEMENT",
                    base.require_improvement_for_auto_promote,
                )
            ),

            ppo_enabled=cls._bool(
                "CONTINUAL_PPO_ENABLED",
                base.ppo_enabled,
            ),

            ppo_every_n_cycles=int(
                os.getenv(
                    "CONTINUAL_PPO_EVERY_N_CYCLES",
                    str(
                        base.ppo_every_n_cycles
                    ),
                )
            ),

            ppo_suite=os.getenv(
                "CONTINUAL_PPO_SUITE",
                base.ppo_suite,
            ),

            eval_suite=os.getenv(
                "CONTINUAL_EVAL_SUITE",
                base.eval_suite,
            ),

            single_gpu_idle_only=cls._bool(
                "CONTINUAL_SINGLE_GPU_IDLE_ONLY",
                base.single_gpu_idle_only,
            ),

            state_db_path=cls._path_env(
                "CONTINUAL_STATE_DB",
                fallback=None,
                default=base.state_db_path,
            ),

            corpus_registry_path=cls._path_env(
                "CONTINUAL_CORPUS_REGISTRY",
                fallback=None,
                default=base.corpus_registry_path,
            ),

            corpus_sources_path=(
                Path(
                    os.environ[
                        "CONTINUAL_CORPUS_SOURCES"
                    ]
                )
                if os.getenv(
                    "CONTINUAL_CORPUS_SOURCES"
                )
                else None
            ),

            recipe=recipe,
        )


def _automation_target_role(
    target_component: str,
) -> str | None:

    if target_component == "hub":
        return "hub"

    if (
        target_component
        == "developer-specialist"
    ):
        return "developer"

    return None


def load_declared_corpus_sources(
    settings: ContinualAutomationSettings,
) -> list[CorpusSource]:
    """
    Compatibility bridge from the governed registry into the existing
    progressive page reader.

    The registry is inventory and policy, NOT optimizer authorization.

    Remote Hugging Face, runtime, and generated sources never enter the
    existing optimizer directly here. External datasets must first be
    converted into governed immutable local snapshots.

    Only explicitly enabled LOCAL sources in one of the legacy page
    formats can be exposed to automation/corpus.py.
    """

    # --------------------------------------------------------
    # Legacy compatibility
    # --------------------------------------------------------
    #
    # Old automation callers may explicitly provide the former
    # list-style corpus source file. If they do, honor that input
    # without confusing it with the new governed registry schema.
    #
    # This path should eventually disappear after every caller and
    # deployment config has migrated.
    # --------------------------------------------------------

    if (
        settings.corpus_sources_path
        is not None
    ):

        legacy_path = (
            settings
            .corpus_sources_path
            .expanduser()
        )

        if not legacy_path.is_absolute():
            legacy_path = (
                REPOSITORY_ROOT
                / legacy_path
            )

        legacy_path = (
            legacy_path
            .resolve()
        )

        if not legacy_path.is_file():
            return []

        raw = json.loads(
            legacy_path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(
            raw,
            list,
        ):
            raise ValueError(
                "Legacy CONTINUAL_CORPUS_SOURCES "
                "must contain a JSON list."
            )

        return [
            CorpusSource.model_validate(
                item
            )
            for item in raw
        ]

    # --------------------------------------------------------
    # Canonical governed registry
    # --------------------------------------------------------

    registry = load_corpus_registry(
        settings.corpus_registry_path
    )

    result: list[CorpusSource] = []

    for source in registry.sources:

        if not (
            source.enabled
            and source.training_eligible
            and not source.evaluation_only
            and source.provider == "local"
            and source.local_path
        ):
            continue

        kind = source.metadata.get(
            "automation_kind"
        )

        if kind not in {
            "semantic_code_jsonl",
            "text",
        }:
            continue

        target_role = (
            _automation_target_role(
                source.target_component
            )
        )

        if target_role is None:
            continue

        result.append(
            CorpusSource(
                source_id=source.source_id,
                kind=kind,
                path=source.local_path,
                target_role=target_role,
                audience=source.audience,
                trusted=(
                    source.trust
                    in {
                        "curated",
                        "verified",
                    }
                ),
                training_eligible=True,
                page_chars=int(
                    source.metadata.get(
                        "page_chars",
                        4096,
                    )
                ),
                records_per_page=int(
                    source.metadata.get(
                        "records_per_page",
                        1,
                    )
                ),
            )
        )

    return result
