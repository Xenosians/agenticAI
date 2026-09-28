from __future__ import annotations

import json
import os

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from learning.paths import (
    REPOSITORY_ROOT,
    RUNTIME_LEARNING_ROOT,
)

from learning.continuous.types import (
    AdaptiveRecipe,
    CorpusSource,
)


class ContinuousLearningSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False

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

    # Fail closed. Collection may run unattended, but optimizer/evaluation
    # stages require explicit deployment opt-in.
    auto_train: bool = False
    auto_evaluate: bool = False

    # Fail-safe default. Production can opt into automatic activation
    # only after the canary/monitor layer is proven.
    auto_promote: bool = False

    require_improvement_for_auto_promote: bool = True

    # PPO is a separately authorized experiment, never an implicit part
    # of enabling continuous collection.
    ppo_enabled: bool = False
    ppo_every_n_cycles: int = Field(
        default=4,
        ge=1,
    )
    ppo_suite: str = "core.v1"

    eval_suite: str = "core.v1"

    single_gpu_idle_only: bool = True

    state_db_path: Path = (
        RUNTIME_LEARNING_ROOT
        / "continuous"
        / "state.sqlite3"
    )

    corpus_sources_path: Path = (
        REPOSITORY_ROOT
        / "config"
        / "continuous_corpus_sources.json"
    )

    recipe: AdaptiveRecipe = Field(
        default_factory=AdaptiveRecipe
    )

    @model_validator(mode="after")
    def validate_stage_controls(
        self,
    ) -> "ContinuousLearningSettings":
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
        value = os.getenv(name)

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

    @classmethod
    def from_env(
        cls,
    ) -> "ContinuousLearningSettings":
        base = cls()

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
                        base.recipe
                        .gradient_accumulation_steps
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
                        base.recipe
                        .corpus_pages_per_cycle
                    ),
                )
            ),
            replay_pages_per_cycle=int(
                os.getenv(
                    "CONTINUAL_CORPUS_REPLAY_PAGES",
                    str(
                        base.recipe
                        .replay_pages_per_cycle
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
            state_db_path=Path(
                os.getenv(
                    "CONTINUAL_STATE_DB",
                    str(
                        base.state_db_path
                    ),
                )
            ),
            corpus_sources_path=Path(
                os.getenv(
                    "CONTINUAL_CORPUS_SOURCES",
                    str(
                        base.corpus_sources_path
                    ),
                )
            ),
            recipe=recipe,
        )


def load_declared_corpus_sources(
    settings: ContinuousLearningSettings,
) -> list[CorpusSource]:
    path = (
        settings
        .corpus_sources_path
        .expanduser()
        .resolve()
    )

    if not path.is_file():
        return []

    raw = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        raw,
        list,
    ):
        raise ValueError(
            "continuous_corpus_sources.json must contain a list."
        )

    return [
        CorpusSource.model_validate(
            item
        )
        for item in raw
    ]
