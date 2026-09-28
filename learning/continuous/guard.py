from __future__ import annotations

from learning.continuous.types import (
    AdaptiveRecipe,
)


def adapt_recipe_from_training_manifest(
    *,
    manifest,
    current: AdaptiveRecipe,
) -> tuple[
    AdaptiveRecipe,
    str,
]:
    """
    Deterministic bounded controller.

    It may adjust a small trusted recipe envelope. It never accepts
    arbitrary optimizer commands from the model.
    """
    observations = list(
        getattr(
            manifest,
            "checkpoint_observations",
            [],
        )
        or []
    )

    fit_diagnosis = str(
        getattr(
            manifest,
            "fit_diagnosis",
            "none",
        )
    )

    if fit_diagnosis == "overfit_guard_triggered":
        updated = current.model_copy(
            update={
                "learning_rate":
                    max(
                        current.learning_rate
                        * 0.5,
                        5e-7,
                    ),

                "max_optimizer_steps":
                    max(
                        2,
                        int(
                            current.max_optimizer_steps
                            * 0.75
                        ),
                    ),

                "replay_pages_per_cycle":
                    min(
                        32,
                        current.replay_pages_per_cycle
                        + 2,
                    ),
            }
        )

        return (
            updated,
            "overfit",
        )

    contract_rates = [
        float(
            item.contract_pass_rate
        )
        for item in observations
    ]

    if (
        fit_diagnosis
        == "underfit_suspected"
        or (
            contract_rates
            and max(
                contract_rates
            )
            < 0.75
        )
    ):
        updated = current.model_copy(
            update={
                "max_optimizer_steps":
                    min(
                        64,
                        current.max_optimizer_steps
                        + 4,
                    ),

                "corpus_pages_per_cycle":
                    min(
                        64,
                        current.corpus_pages_per_cycle
                        + 2,
                    ),
            }
        )

        return (
            updated,
            "underfit",
        )

    return (
        current,
        "stable",
    )
