import re

import pytest

from learning.training.hub_hybrid_qlora import HubTrainingSettings
from learning.training.lora_targets import (
    TEXT_PROJECTION_MODULES, text_lora_targets, validate_text_adapter_scope,
)


def test_multimodal_suffixes_match_only_language_projections():
    pattern = text_lora_targets("mistral3", HubTrainingSettings().lora_target_modules)
    for suffix in TEXT_PROJECTION_MODULES:
        assert re.fullmatch(pattern, f"model.language_model.layers.0.self_attn.{suffix}")
        assert not re.fullmatch(pattern, f"model.vision_tower.transformer.layers.0.attention.{suffix}")
        assert not re.fullmatch(pattern, f"model.multi_modal_projector.{suffix}")
        assert not re.fullmatch(pattern, f"other.model.language_model.layers.0.{suffix}")
    assert not re.fullmatch(pattern, "lm_head")


def test_requested_subset_is_preserved_without_regex_injection():
    pattern = text_lora_targets("mistral3", ["q_proj", "custom.*"])
    assert re.fullmatch(pattern, "model.language_model.layers.0.q_proj")
    assert re.fullmatch(pattern, "model.language_model.layers.0.custom.*")
    assert not re.fullmatch(pattern, "model.language_model.layers.0.k_proj")
    assert not re.fullmatch(pattern, "model.language_model.layers.0.custom_layer")


def test_existing_specialist_all_linear_policy_keeps_language_projection_set():
    assert text_lora_targets("mistral3", "all-linear") == text_lora_targets(
        "mistral3", list(TEXT_PROJECTION_MODULES)
    )


@pytest.mark.parametrize("targets", [[], [""], [" q_proj"], [None], "q_proj"])
def test_invalid_multimodal_configuration_is_rejected(targets):
    with pytest.raises(ValueError, match="nonempty module suffixes"):
        text_lora_targets("mistral3", targets)


@pytest.mark.parametrize("model_type", ["qwen2", "qwen3", "llama", "mistral"])
@pytest.mark.parametrize("targets", [["q_proj"], "all-linear"])
def test_text_checkpoint_configuration_is_preserved(model_type, targets):
    assert text_lora_targets(model_type, targets) == targets


def test_peft_attaches_only_inside_language_tower():
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    for tower_name in ("language_model", "vision_tower", "multi_modal_projector"):
        tower = torch.nn.Module()
        tower.layers = torch.nn.ModuleList([torch.nn.Module()])
        tower.layers[0].q_proj = torch.nn.Linear(4, 4)
        setattr(model.model, tower_name, tower)

    adapted = peft.get_peft_model(model, peft.LoraConfig(
        r=2, lora_alpha=4,
        target_modules=text_lora_targets("mistral3", ["q_proj"]),
        bias="none",
    ))
    trainable = [name for name, value in adapted.named_parameters() if value.requires_grad]
    assert trainable
    assert all(".language_model." in name and ".lora_" in name for name in trainable)
    assert all(not value.requires_grad for name, value in adapted.named_parameters()
               if ".vision_tower." in name or ".multi_modal_projector." in name)
    validate_text_adapter_scope(adapted, "mistral3")

    # A real CPU optimizer step updates adapter weights, not the frozen towers.
    model.forward = lambda values: model.model.language_model.layers[0].q_proj(values)
    before = {name: value.detach().clone() for name, value in adapted.named_parameters()}
    optimizer = torch.optim.SGD([value for value in adapted.parameters() if value.requires_grad], lr=0.1)
    adapted(torch.ones(1, 4)).sum().backward()
    optimizer.step()
    changed = [name for name, value in adapted.named_parameters()
               if not torch.equal(value.detach(), before[name])]
    assert changed
    assert all(".language_model." in name and ".lora_" in name for name in changed)


def test_resumed_adapter_scope_checks_actual_weights_without_changing_config():
    from types import SimpleNamespace
    safe = "base_model.model.model.language_model.layers.0.q_proj.lora_A.default.weight"
    unsafe = "base_model.model.model.vision_tower.layers.0.q_proj.lora_A.default.weight"
    weights = {safe: SimpleNamespace(requires_grad=True),
               unsafe: SimpleNamespace(requires_grad=False)}
    model = SimpleNamespace(named_parameters=lambda: weights.items())
    validate_text_adapter_scope(model, "mistral3")
    weights[unsafe].requires_grad = True
    with pytest.raises(ValueError, match="outside the language model"):
        validate_text_adapter_scope(model, "mistral3")


def test_adapter_scope_rejects_no_trainable_parameters():
    from types import SimpleNamespace
    with pytest.raises(ValueError, match="no trainable parameters"):
        validate_text_adapter_scope(SimpleNamespace(named_parameters=lambda: []), "mistral3")


def test_other_architectures_are_not_given_a_mistral_parameter_layout():
    from types import SimpleNamespace
    validate_text_adapter_scope(SimpleNamespace(), "qwen2")
