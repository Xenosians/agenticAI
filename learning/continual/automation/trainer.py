from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from config import Settings

from learning.continual.checkpoints import (
    AdapterCheckpointStore,
)
from learning.continual.automation.guard import (
    adapt_recipe_from_training_manifest,
)
from learning.continual.automation.types import (
    AdaptiveRecipe,
)
from learning.paths import (
    REPOSITORY_ROOT,
)
from learning.training.hub_hybrid_qlora import (
    HubTrainingSettings,
    train_hub_adapter,
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


@contextmanager
def _after_each_optimizer_step(check: Callable[[], None] | None):
    """Run ``check`` after every optimizer step taken in this block.

    The check may raise (for example TrainingYielded) to end training at a
    step boundary. The hook is removed when the block ends, so other
    optimizers in the process are unaffected.
    """
    if check is None:
        yield
        return

    from torch.optim.optimizer import register_optimizer_step_post_hook

    handle = register_optimizer_step_post_hook(lambda *_: check())
    try:
        yield
    finally:
        handle.remove()


def train_continual_candidate(
    *,
    materialization_directory: Path,
    recipe: AdaptiveRecipe,
    seed_adapter: Path | None,
    between_steps: Callable[[], None] | None = None,
):
    """
    One bounded micro-cycle.

    The patched Phase-5 trainer accepts a promoted seed adapter when
    available. With no active adapter it starts from the frozen base model.
    ``between_steps`` runs after every optimizer step (see limiter.py).
    """
    settings = (
        HubTrainingSettings(
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

    with _after_each_optimizer_step(between_steps):
        return (
            train_hub_adapter(
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
