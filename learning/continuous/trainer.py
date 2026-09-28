from __future__ import annotations

from pathlib import Path

from config import Settings

from learning.continual.checkpoints import (
    AdapterCheckpointStore,
)
from learning.continuous.guard import (
    adapt_recipe_from_training_manifest,
)
from learning.continuous.types import (
    AdaptiveRecipe,
)
from learning.paths import (
    REPOSITORY_ROOT,
)
from learning.training.phase5_hybrid_qlora import (
    Phase5TrainingSettings,
    train_phase5_hub_adapter,
)


def seed_adapter_directory(
) -> tuple[
    str | None,
    Path | None,
]:
    store = (
        AdapterCheckpointStore()
    )

    pointer = store.active()

    if pointer is None:
        return (
            None,
            None,
        )

    checkpoint = (
        store
        .verify_adapter(
            pointer.checkpoint_id
        )
    )

    return (
        checkpoint.checkpoint_id,
        Path(
            checkpoint
            .adapter_directory
        )
        .expanduser()
        .resolve(),
    )


def train_continuous_candidate(
    *,
    materialization_directory: Path,
    recipe: AdaptiveRecipe,
    seed_adapter: Path | None,
):
    """
    One bounded micro-cycle.

    The patched Phase-5 trainer accepts a promoted seed adapter when
    available. With no active adapter it starts from the frozen base model.
    """
    settings = (
        Phase5TrainingSettings(
            learning_rate=(
                recipe.learning_rate
            ),
            lora_r=(
                recipe.lora_r
            ),
            max_optimizer_steps=(
                recipe
                .max_optimizer_steps
            ),
            checkpoint_every_steps=max(
                1,
                min(
                    4,
                    recipe
                    .max_optimizer_steps,
                ),
            ),
            early_stop_patience=3,
            gradient_accumulation_steps=(
                recipe
                .gradient_accumulation_steps
            ),
            epochs=3,
            max_length=1024,
        )
    )

    app_settings = Settings()

    profile = (
        app_settings
        .require_model_profile(
            app_settings
            .hub_model_key
        )
    )

    return (
        train_phase5_hub_adapter(
            materialization_directory=(
                materialization_directory
            ),
            settings=settings,
            allow_training=True,
            backend=(
                profile.backend
            ),
            agent_directory=(
                REPOSITORY_ROOT
                / "subagents"
                / "agents"
            ),
            seed_adapter_directory=(
                seed_adapter
            ),
        )
    )


def update_recipe(
    *,
    manifest,
    recipe: AdaptiveRecipe,
) -> tuple[
    AdaptiveRecipe,
    str,
]:
    return (
        adapt_recipe_from_training_manifest(
            manifest=manifest,
            current=recipe,
        )
    )
