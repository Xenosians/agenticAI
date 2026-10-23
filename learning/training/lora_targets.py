"""Keep text adaptation inside the language tower of multimodal checkpoints."""

from __future__ import annotations

import re


TEXT_PROJECTION_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def text_lora_targets(
    model_type: str, target_modules: list[str] | str,
) -> list[str] | str:
    """Return PEFT targets without matching a Mistral3 vision tower by suffix.

    PEFT lists match module suffixes across *all* towers. Its string form is
    instead a full regular expression, so the architecture's language prefix
    must be anchored. Other checkpoint architectures keep their given targets.
    """
    if model_type != "mistral3":
        return target_modules

    if target_modules == "all-linear":
        suffixes = TEXT_PROJECTION_MODULES
    elif isinstance(target_modules, list) and target_modules and all(
        isinstance(name, str) and name.strip() == name and name
        for name in target_modules
    ):
        suffixes = target_modules
    else:
        raise ValueError("Mistral3 text LoRA requires nonempty module suffixes.")

    return (
        r"^model\.language_model\..*\."
        + "(" + "|".join(re.escape(name) for name in suffixes) + ")$"
    )


def validate_text_adapter_scope(model, model_type: str) -> None:
    """Check actual trainable weights, including adapters resumed from disk.

    Rewriting an existing adapter's config would lose its learned weights.
    Inspect the loaded PEFT model instead, before any optimizer is created.
    """
    if model_type != "mistral3":
        return
    trainable = [name for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    if not trainable:
        raise ValueError("Text-only Mistral3 adapter has no trainable parameters.")
    if any(not name.startswith("base_model.model.model.language_model.")
           for name in trainable):
        raise ValueError(
            "Text-only Mistral3 adapter has trainable weights outside the language model."
        )
