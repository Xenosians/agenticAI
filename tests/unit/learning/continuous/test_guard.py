from types import SimpleNamespace

from learning.continuous.guard import (
    adapt_recipe_from_training_manifest,
)
from learning.continuous.types import (
    AdaptiveRecipe,
)


def test_overfit_guard_reduces_next_cycle_pressure():
    recipe = AdaptiveRecipe(
        learning_rate=5e-6,
        max_optimizer_steps=12,
        replay_pages_per_cycle=2,
    )

    manifest = SimpleNamespace(
        fit_diagnosis=(
            "overfit_guard_triggered"
        ),
        checkpoint_observations=[],
    )

    updated, signal = (
        adapt_recipe_from_training_manifest(
            manifest=manifest,
            current=recipe,
        )
    )

    assert signal == "overfit"
    assert (
        updated.learning_rate
        < recipe.learning_rate
    )
    assert (
        updated.max_optimizer_steps
        < recipe.max_optimizer_steps
    )
    assert (
        updated.replay_pages_per_cycle
        > recipe.replay_pages_per_cycle
    )
