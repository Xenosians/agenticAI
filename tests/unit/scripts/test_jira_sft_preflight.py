"""Preflight admits exactly the trainer's sequences and recipe, without weights."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from config import ModelProfileSettings
from learning.training import jira_sft
from scripts import jira_sft_preflight as preflight


class CharacterTokenizer:
    """Each character is one token; assets, model weights and GPU are unnecessary."""
    chat_template = None
    eos_token = "\u00b6"
    eos_token_id = ord(eos_token)
    pad_token_id = 0

    def apply_chat_template(self, *args, **kwargs):
        raise ValueError("fixture has no native template")

    def __call__(self, text, **kwargs):
        import torch
        return {"input_ids": torch.tensor([[ord(char) for char in text]])}


def record(identity, split, *, content="List Jira projects"):
    return jira_sft.JiraSftRecord(
        record_id=identity, split=split, tool_name="jira_project_list",
        user_request="Legacy request should not replace saved messages " * 20,
        semantic_context={}, target_response='[{"name":"jira_project_list","arguments":{}}]',
        prompt_messages=[{"role": "system", "content": "Trusted task policy"},
                         {"role": "user", "content": content}],
    )


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    import transformers

    base = tmp_path / "base"
    base.mkdir()
    fixture = SimpleNamespace(
        train=[record("train-scope", "train")],
        validation=[record("validation-scope", "validation")],
        tokenizer=CharacterTokenizer(), corpus_calls=[], tokenizer_calls=[], argument_calls=[],
        profile=ModelProfileSettings(backend="hf-causal", model_path=base, worker_prompt_profile="compact"),
    )
    manifest = SimpleNamespace(base_model_key="jira-func", worker_prompt_profile="compact_grounded",
                               prompt_layout="runtime_task")
    def verify(**kwargs):
        fixture.corpus_calls.append(kwargs)
        return manifest, fixture.train, fixture.validation, "Legacy system policy " * 30
    def tokenizer(path, **kwargs):
        fixture.tokenizer_calls.append((path, kwargs))
        return fixture.tokenizer
    def arguments(**kwargs):
        fixture.argument_calls.append(kwargs)
        return SimpleNamespace(**kwargs)
    monkeypatch.setattr(preflight, "assert_training_versions", lambda: {"fixture": "isolated"})
    monkeypatch.setattr(preflight, "verify_jira_sft_corpus", verify)
    monkeypatch.setattr(preflight, "Settings", lambda: SimpleNamespace(require_model_profile=lambda _: fixture.profile))
    monkeypatch.setattr(preflight, "hardware_summary", lambda: {"cuda_device": "isolated metadata fixture"})
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer)
    # Capture the exact real builder's constructor inputs, without pretending a
    # CPU test established the deployment's BF16/CUDA hardware readiness.
    monkeypatch.setattr(transformers, "TrainingArguments", arguments)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                        lambda *args, **kwargs: pytest.fail("Preflight loaded model weights"))
    return fixture


def test_explicit_candidate_uses_saved_runtime_messages_and_actual_guarded_recipe(isolated, capsys):
    original = isolated.profile.model_dump()
    assert preflight.main([
        "--worker-prompt-profile", "compact_grounded", "--max-length", "256",
        "--max-steps", "17", "--learning-rate", "0.0003",
        "--gradient-accumulation-steps", "4", "--lora-r", "8", "--lora-alpha", "16",
        "--checkpoint-interval", "6", "--early-stopping-patience", "2",
    ]) == 0
    output = capsys.readouterr().out
    assert "prompt profile: compact_grounded" in output
    assert "prompt layout: runtime_task" in output
    assert "train records: 1" in output and "validation records: 1" in output
    assert "PREFLIGHT: PASS (prerequisites only" in output
    assert isolated.corpus_calls[0]["worker_prompt_profile"] == "compact_grounded"
    assert isolated.profile.model_dump() == original
    assert isolated.tokenizer_calls == [(str(isolated.profile.model_path), {"local_files_only": True})]
    assert len(isolated.argument_calls) == 1
    args = isolated.argument_calls[0]
    assert args["max_steps"] == 17 and args["learning_rate"] == 0.0003
    assert args["gradient_accumulation_steps"] == 4 and args["warmup_steps"] == 1
    assert args["eval_strategy"] == args["save_strategy"] == "steps"
    assert args["eval_steps"] == args["save_steps"] == 6
    assert args["eval_on_start"] is True and args["load_best_model_at_end"] is True
    assert args["metric_for_best_model"] == "eval_loss" and args["greater_is_better"] is False
    assert args["restore_callback_states_from_checkpoint"] is True
    assert args["bf16"] is True and args["per_device_train_batch_size"] == 1


def test_default_recipe_budgets_are_unchanged(isolated, capsys):
    assert preflight.main([]) == 0
    assert isolated.corpus_calls[0]["worker_prompt_profile"] is None
    args = isolated.argument_calls[0]
    assert args["max_steps"] == 160 and args["gradient_accumulation_steps"] == 8
    assert args["eval_steps"] == args["save_steps"] == 10
    assert "configured train max length: 1280" in capsys.readouterr().out


@pytest.mark.parametrize("split", ["train", "validation"])
def test_oversized_record_in_either_partition_rejects_before_hardware(isolated, monkeypatch, capsys, split):
    getattr(isolated, split)[:] = [record(f"oversized-{split}", split, content="x" * 300)]
    monkeypatch.setattr(preflight, "hardware_summary", lambda: pytest.fail("Hardware reached before sequence admission"))
    with pytest.raises(ValueError, match=f"oversized-{split}.*exceeds max_length"):
        preflight.main(["--max-length", "256"])
    assert isolated.argument_calls == []
    assert "PREFLIGHT: PASS" not in capsys.readouterr().out


@pytest.mark.parametrize("split", ["train", "validation"])
def test_empty_partition_rejects_before_tokenizer_or_hardware(isolated, capsys, split):
    getattr(isolated, split).clear()
    with pytest.raises(ValueError, match="non-empty train and validation"):
        preflight.main([])
    assert isolated.tokenizer_calls == [] and isolated.argument_calls == []
    assert "PREFLIGHT: PASS" not in capsys.readouterr().out


@pytest.mark.parametrize("flag,value", [
    ("--max-steps", "0"), ("--max-length", "255"), ("--learning-rate", "nan"),
    ("--gradient-accumulation-steps", "0"), ("--lora-r", "0"), ("--lora-alpha", "0"),
    ("--checkpoint-interval", "0"), ("--early-stopping-patience", "0"),
])
def test_invalid_recipe_rejects_before_dependency_and_corpus_io(isolated, monkeypatch, capsys, flag, value):
    monkeypatch.setattr(preflight, "assert_training_versions", lambda: pytest.fail("Invalid recipe reached dependency I/O"))
    with pytest.raises(ValueError):
        preflight.main([flag, value])
    assert isolated.corpus_calls == [] and isolated.tokenizer_calls == []
    assert "PREFLIGHT: PASS" not in capsys.readouterr().out


@pytest.mark.parametrize("field,value,match", [
    ("backend", "http", "hf-causal"), ("model_path", None, "no model_path"),
])
def test_incompatible_base_rejects_before_tokenizer(isolated, capsys, field, value, match):
    isolated.profile = isolated.profile.model_copy(update={field: value})
    with pytest.raises(ValueError, match=match):
        preflight.main([])
    assert isolated.tokenizer_calls == [] and isolated.argument_calls == []
    assert "PREFLIGHT: PASS" not in capsys.readouterr().out


def test_tokenizer_without_pad_or_eos_is_not_admitted(isolated, capsys):
    isolated.tokenizer.pad_token_id = isolated.tokenizer.eos_token_id = None
    with pytest.raises(ValueError, match="neither pad nor EOS"):
        preflight.main([])
    assert isolated.argument_calls == [] and "PREFLIGHT: PASS" not in capsys.readouterr().out


@pytest.mark.parametrize("available,bf16,match", [(False, False, "CUDA"), (True, False, "BF16")])
def test_hardware_requires_actual_training_prerequisites(monkeypatch, available, bf16, match):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: bf16)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: pytest.fail("Unavailable hardware was queried"))
    with pytest.raises(ValueError, match=match):
        preflight.hardware_summary()


def test_constructor_failure_is_not_reported_as_success(isolated, monkeypatch, capsys):
    def broken(**kwargs):
        raise TypeError("isolated constructor drift")
    monkeypatch.setattr(preflight, "build_jira_training_arguments", broken)
    with pytest.raises(TypeError, match="constructor drift"):
        preflight.main([])
    assert "PREFLIGHT: PASS" not in capsys.readouterr().out
