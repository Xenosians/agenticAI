"""Validation-guarded Jira training: best checkpoint, early stopping and resume.

These run the real Trainer with a tiny CPU BLOOM+LoRA model; no project weights,
GPU or Jira provider is involved.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from learning.training import jira_sft


PROJECT_ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


@pytest.fixture
def transformers():
    pytest.importorskip("peft")
    return pytest.importorskip("transformers")


def _item(prompt, target):
    return {"input_ids": prompt + target, "attention_mask": [1] * (len(prompt) + len(target)),
            "labels": [-100] * len(prompt) + target}


TRAIN = [_item([5, 6, 7], [9, 10]), _item([11, 12], [14, 15, 16]), _item([19, 20, 21], [22, 23])]


def _model(torch, transformers):
    import peft

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = transformers.BloomForCausalLM(transformers.BloomConfig(
            vocab_size=64, hidden_size=16, n_layer=1, n_head=2,
            hidden_dropout=0.0, attention_dropout=0.0,
        ))
        return peft.get_peft_model(model, peft.LoraConfig(
            r=2, lora_alpha=4, target_modules=["query_key_value"], task_type="CAUSAL_LM",
        ))


def _trainer(transformers, model, directory, *, max_steps, learning_rate, validation,
             patience, baseline=True, callbacks=()):
    arguments = transformers.TrainingArguments(
        output_dir=str(directory), report_to="none", use_cpu=True, seed=0,
        per_device_train_batch_size=1, gradient_accumulation_steps=1,
        max_steps=max_steps, learning_rate=learning_rate, weight_decay=0.0,
        remove_unused_columns=False, logging_strategy="no", disable_tqdm=True,
        **jira_sft.jira_validation_guard_arguments(2, baseline_evaluation=baseline),
    )
    trainer = jira_sft.build_jira_trainer(
        model=model, training_args=arguments, train_items=TRAIN, validation_items=validation,
        pad_token_id=0, early_stopping_patience=patience,
    )
    for callback in callbacks:
        trainer.add_callback(callback)
    return trainer


def _adapter(model):
    return {name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad}


def test_non_improving_validation_stops_early_and_keeps_the_best_checkpoint(torch, transformers, tmp_path):
    # Without updates the validation loss never improves after the first checkpoint.
    trainer = _trainer(transformers, _model(torch, transformers), tmp_path, max_steps=12,
                       learning_rate=0.0, validation=TRAIN[:2], patience=2)
    trainer.train()
    summary = jira_sft.summarize_validation(trainer.state, max_steps=12)
    assert summary.early_stopped is True
    assert summary.optimizer_steps == 6
    assert summary.best_checkpoint_step == 2
    assert summary.evaluated_steps == [0, 2, 4, 6]
    assert summary.baseline_eval_loss == pytest.approx(summary.best_eval_loss)


def test_improving_validation_trains_to_budget_and_ends_on_the_best_adapter(torch, transformers, tmp_path):
    model = _model(torch, transformers)
    trainer = _trainer(transformers, model, tmp_path, max_steps=6, learning_rate=0.05,
                       validation=TRAIN, patience=1)
    trainer.train()
    summary = jira_sft.summarize_validation(trainer.state, max_steps=6)
    assert summary.early_stopped is False
    assert summary.best_checkpoint_step == 6
    assert summary.best_eval_loss < summary.baseline_eval_loss
    best = jira_sft.resumable_checkpoint(tmp_path)
    assert best is not None and best.endswith("checkpoint-6")


def test_validation_split_is_required(torch, transformers, tmp_path):
    with pytest.raises(ValueError, match="validation"):
        _trainer(transformers, _model(torch, transformers), tmp_path, max_steps=2,
                 learning_rate=0.0, validation=[], patience=1)


def test_interrupted_run_resumes_from_its_last_checkpoint_with_the_same_result(torch, transformers, tmp_path):
    reference_model = _model(torch, transformers)
    _trainer(transformers, reference_model, tmp_path / "reference", max_steps=4,
             learning_rate=0.05, validation=TRAIN, patience=1).train()

    class Interrupt(transformers.TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 3:
                raise KeyboardInterrupt("simulated interruption after checkpoint-2")

    directory = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        _trainer(transformers, _model(torch, transformers), directory, max_steps=4,
                 learning_rate=0.05, validation=TRAIN, patience=1, callbacks=[Interrupt()]).train()
    checkpoint = jira_sft.resumable_checkpoint(directory)
    assert checkpoint is not None and checkpoint.endswith("checkpoint-2")

    resumed_model = _model(torch, transformers)
    # The baseline was recorded before the interruption; re-evaluating the
    # restored checkpoint would count as a non-improving evaluation.
    resumed = _trainer(transformers, resumed_model, directory, max_steps=4, learning_rate=0.05,
                       validation=TRAIN, patience=1, baseline=False)
    resumed.train(resume_from_checkpoint=checkpoint)
    summary = jira_sft.summarize_validation(resumed.state, max_steps=4)
    assert summary.optimizer_steps == 4 and summary.early_stopped is False
    assert summary.evaluated_steps == [0, 2, 4]
    expected = _adapter(reference_model)
    observed = _adapter(resumed_model)
    assert observed.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(observed[name], expected[name])


def test_checkpoint_interval_must_be_positive():
    with pytest.raises(ValueError, match="checkpoint_interval"):
        jira_sft.jira_validation_guard_arguments(0)


@pytest.fixture
def configured_profiles(monkeypatch):
    from config import ModelProfileSettings, Settings

    settings = Settings(_env_file=None, model_profiles={
        "jira-func": ModelProfileSettings(backend="hf-causal", worker_prompt_profile="compact"),
    })
    monkeypatch.setattr(jira_sft, "Settings", lambda: settings)
    return settings


def _resume_fixture(tmp_path, configured_profiles, monkeypatch):
    directory = tmp_path / "corpus"
    jira_sft.build_jira_sft_corpus(
        project_root=PROJECT_ROOT, output_directory=directory,
        worker_prompt_profile="compact_grounded", prompt_layout="runtime_task",
    )

    class HardwareGateReached(Exception):
        pass

    def stop_at_hardware_gate():
        raise HardwareGateReached

    monkeypatch.setattr(jira_sft, "assert_training_versions", stop_at_hardware_gate)
    options = dict(
        project_root=PROJECT_ROOT, corpus_directory=directory, output_root=tmp_path / "output",
        allow_training=True, max_steps=40, max_length=1408, worker_prompt_profile="compact_grounded",
        checkpoint_interval=10, early_stopping_patience=3,
    )
    return options, HardwareGateReached


def test_resume_requires_the_same_run_configuration_before_any_hardware_use(
        configured_profiles, tmp_path, monkeypatch):
    options, hardware_gate = _resume_fixture(tmp_path, configured_profiles, monkeypatch)
    corpus, _, _, _ = jira_sft.verify_jira_sft_corpus(
        project_root=PROJECT_ROOT, corpus_directory=options["corpus_directory"],
        worker_prompt_profile="compact_grounded",
    )
    run = tmp_path / "output" / "runs" / "interrupted"
    run.mkdir(parents=True)
    identity = jira_sft.jira_run_identity(corpus_manifest=corpus, **{
        key: value for key, value in options.items()
        if key not in {"project_root", "corpus_directory", "output_root", "allow_training",
                       "worker_prompt_profile"}
    })
    (run / "run-config.json").write_text(json.dumps(identity), encoding="utf-8")

    with pytest.raises(hardware_gate):
        jira_sft.train_jira_sft(**options, resume_run_directory=run)
    for changed in ({"max_steps": 41}, {"checkpoint_interval": 5}, {"max_length": 1536}):
        with pytest.raises(ValueError, match="does not match"):
            jira_sft.train_jira_sft(**{**options, **changed}, resume_run_directory=run)
    with pytest.raises(ValueError, match="run-config"):
        jira_sft.train_jira_sft(**options, resume_run_directory=tmp_path / "missing-run")
    assert sorted(path.name for path in run.iterdir()) == ["run-config.json"]


def test_training_cli_forwards_checkpoint_and_resume_options(monkeypatch, tmp_path):
    import sys
    from scripts import jira_sft_train

    class StopBeforeTraining(Exception):
        pass

    observed = {}

    def capture(**kwargs):
        observed.update(kwargs)
        raise StopBeforeTraining

    monkeypatch.setattr(jira_sft_train, "train_jira_sft", capture)
    monkeypatch.setattr(sys, "argv", [
        "jira_sft_train.py", "--corpus", str(tmp_path / "corpus"),
        "--checkpoint-interval", "10", "--early-stopping-patience", "4",
        "--resume-run", str(tmp_path / "run"),
    ])
    with pytest.raises(StopBeforeTraining):
        jira_sft_train.main()
    assert observed["checkpoint_interval"] == 10
    assert observed["early_stopping_patience"] == 4
    assert observed["resume_run_directory"] == tmp_path / "run"
