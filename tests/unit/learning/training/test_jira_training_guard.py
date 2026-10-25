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
             patience, baseline=True, callbacks=(), checkpoint_interval=2):
    arguments = transformers.TrainingArguments(
        output_dir=str(directory), report_to="none", use_cpu=True, seed=0,
        per_device_train_batch_size=1, gradient_accumulation_steps=1,
        max_steps=max_steps, learning_rate=learning_rate, weight_decay=0.0,
        remove_unused_columns=False, logging_strategy="no", disable_tqdm=True,
        **jira_sft.jira_validation_guard_arguments(checkpoint_interval, baseline_evaluation=baseline),
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


def test_short_budget_still_evaluates_and_saves_its_final_step(torch, transformers, tmp_path):
    # The locked Trainer's end-of-budget callback evaluates/saves a nonmultiple
    # final step. Keep the user's interval and optimizer budget exactly as given.
    trainer = _trainer(
        transformers, _model(torch, transformers), tmp_path, max_steps=1,
        learning_rate=0.0, validation=TRAIN, patience=3, checkpoint_interval=10,
    )
    trainer.train()
    summary = jira_sft.summarize_validation(trainer.state, max_steps=1)
    assert trainer.args.eval_steps == trainer.args.save_steps == 10
    assert summary.optimizer_steps == 1
    assert summary.evaluated_steps == [0, 1]
    assert summary.best_checkpoint_step == 1
    assert summary.early_stopped is False
    assert jira_sft.resumable_checkpoint(tmp_path).endswith("checkpoint-1")


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
def configured_profiles(monkeypatch, tmp_path):
    from config import ModelProfileSettings, Settings

    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    (base / "tokenizer.json").write_text("{}", encoding="utf-8")
    (base / "model.safetensors").write_bytes(b"isolated model identity fixture")
    settings = Settings(_env_file=None, model_profiles={
        "jira-func": ModelProfileSettings(backend="hf-causal", worker_prompt_profile="compact", model_path=base),
    })
    monkeypatch.setattr(jira_sft, "Settings", lambda: settings)
    return settings


def _resume_fixture(tmp_path, configured_profiles, monkeypatch):
    directory = tmp_path / "corpus"
    # Resume metadata does not depend on hundreds of language paraphrases.
    # Keep a real reviewed train/validation record for every registered Jira
    # capability, while retaining the real corpus builder and verifier.
    records = jira_sft._build_records(project_root=PROJECT_ROOT)
    representatives = {}
    for record in records:
        representatives.setdefault((record.split, record.tool_name), record)
    monkeypatch.setattr(jira_sft, "_build_records", lambda **kwargs: list(representatives.values()))
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
    }, base_model_identity=jira_sft.jira_base_model_identity(
        configured_profiles.require_model_profile("jira-func").model_path,
    ))
    (run / "run-config.json").write_text(json.dumps(identity), encoding="utf-8")
    _checkpoint_fixture(run, configured_profiles.require_model_profile("jira-func").model_path)

    before = {path.relative_to(run): path.read_bytes() for path in run.rglob("*") if path.is_file()}
    with pytest.raises(hardware_gate):
        jira_sft.train_jira_sft(**options, resume_run_directory=run)
    for changed in ({"max_steps": 41}, {"checkpoint_interval": 5}, {"max_length": 1536}):
        with pytest.raises(ValueError, match="does not match"):
            jira_sft.train_jira_sft(**{**options, **changed}, resume_run_directory=run)
    with pytest.raises(ValueError, match="run-config"):
        jira_sft.train_jira_sft(**options, resume_run_directory=tmp_path / "missing-run")
    assert {path.relative_to(run): path.read_bytes() for path in run.rglob("*") if path.is_file()} == before


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


def _checkpoint_fixture(run, base, *, step=10, max_steps=40, best=None):
    """Plain preflight metadata; binary fixtures are never passed to Trainer."""
    checkpoint = run / "trainer-work" / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    for name in ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"):
        (checkpoint / name).write_bytes(b"opaque preflight fixture; not a usable model")
    (checkpoint / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": str(base.resolve()),
    }), encoding="utf-8")
    (checkpoint / "trainer_state.json").write_text(json.dumps({
        "global_step": step, "max_steps": max_steps,
        "best_global_step": step if best is None else int(best.name.rsplit("-", 1)[1]),
        "best_model_checkpoint": str((checkpoint if best is None else best).resolve()),
        "best_metric": 0.5,
        "stateful_callbacks": {"EarlyStoppingCallback": {
            "args": {"early_stopping_patience": 3, "early_stopping_threshold": 0.0},
            "attributes": {"early_stopping_patience_counter": 0},
        }},
    }), encoding="utf-8")
    return checkpoint


def _bound_resume_fixture(tmp_path, configured_profiles, monkeypatch):
    options, hardware_gate = _resume_fixture(tmp_path, configured_profiles, monkeypatch)
    corpus, _, _, _ = jira_sft.verify_jira_sft_corpus(
        project_root=PROJECT_ROOT, corpus_directory=options["corpus_directory"],
        worker_prompt_profile="compact_grounded",
    )
    base = configured_profiles.require_model_profile("jira-func").model_path
    identity = jira_sft.jira_run_identity(
        corpus_manifest=corpus, base_model_identity=jira_sft.jira_base_model_identity(base),
        max_steps=40, max_length=1408, checkpoint_interval=10, early_stopping_patience=3,
    )
    run = tmp_path / "output" / "runs" / "interrupted"
    run.mkdir(parents=True)
    (run / "run-config.json").write_text(json.dumps(identity), encoding="utf-8")
    checkpoint = _checkpoint_fixture(run, base)
    options["resume_run_directory"] = run
    return options, run, checkpoint, hardware_gate


@pytest.mark.parametrize("filename", ["model.safetensors", "tokenizer.json", "config.json"])
def test_resume_rejects_same_path_content_drift_without_rewriting_run(
        filename, configured_profiles, tmp_path, monkeypatch):
    from learning.evidence.execution_provenance import fingerprint_runtime_model_artifact

    options, run, _, _ = _bound_resume_fixture(tmp_path, configured_profiles, monkeypatch)
    base = configured_profiles.require_model_profile("jira-func").model_path
    # Pre-populate the runtime snapshot. Training must read fresh bytes, not it.
    fingerprint_runtime_model_artifact(base)
    before = (run / "run-config.json").read_bytes()
    (base / filename).write_bytes((base / filename).read_bytes() + b" ")
    with pytest.raises(ValueError, match="does not match.*base_model_identity"):
        jira_sft.train_jira_sft(**options)
    assert (run / "run-config.json").read_bytes() == before
    assert not (run / "training-manifest.json").exists()


def test_resume_binds_resolved_base_path_as_well_as_content(configured_profiles, tmp_path, monkeypatch):
    import shutil

    options, _, _, _ = _bound_resume_fixture(tmp_path, configured_profiles, monkeypatch)
    profile = configured_profiles.model_profiles["jira-func"]
    relocated = tmp_path / "relocated-base"
    shutil.copytree(profile.model_path, relocated)
    profile.model_path = relocated
    with pytest.raises(ValueError, match="does not match.*base_model_identity"):
        jira_sft.train_jira_sft(**options)


def test_legacy_resume_cannot_claim_the_current_base_as_its_original_identity(
        configured_profiles, tmp_path, monkeypatch):
    options, run, _, _ = _bound_resume_fixture(tmp_path, configured_profiles, monkeypatch)
    path = run / "run-config.json"
    legacy = json.loads(path.read_text(encoding="utf-8"))
    legacy["schema"] = "jira-specialist-sft-run.v1"
    legacy.pop("base_model_identity")
    path.write_text(json.dumps(legacy), encoding="utf-8")
    before = {item.relative_to(run): item.read_bytes() for item in run.rglob("*") if item.is_file()}
    with pytest.raises(ValueError, match="no recorded base-model/tokenizer content identity.*legacy run"):
        jira_sft.train_jira_sft(**options)
    assert {item.relative_to(run): item.read_bytes() for item in run.rglob("*") if item.is_file()} == before


@pytest.mark.parametrize("filename", [
    "adapter_config.json", "adapter_model.safetensors", "trainer_state.json",
    "optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin",
])
@pytest.mark.parametrize("missing", [True, False], ids=["missing", "empty"])
def test_resume_incomplete_checkpoint_is_rejected_before_hardware(
        filename, missing, configured_profiles, tmp_path, monkeypatch):
    options, _, checkpoint, _ = _bound_resume_fixture(tmp_path, configured_profiles, monkeypatch)
    path = checkpoint / filename
    if missing:
        path.unlink()
    else:
        path.write_bytes(b"")
    with pytest.raises(ValueError, match="checkpoint is incomplete"):
        jira_sft.train_jira_sft(**options)


def test_resume_does_not_fall_back_to_older_complete_checkpoint(
        configured_profiles, tmp_path, monkeypatch):
    options, run, _, _ = _bound_resume_fixture(tmp_path, configured_profiles, monkeypatch)
    (run / "trainer-work" / "checkpoint-20").mkdir()
    with pytest.raises(ValueError, match="checkpoint is incomplete: checkpoint-20"):
        jira_sft.train_jira_sft(**options)


def test_resume_without_a_saved_checkpoint_fails_before_hardware(
        configured_profiles, tmp_path, monkeypatch):
    options, _, checkpoint, _ = _bound_resume_fixture(tmp_path, configured_profiles, monkeypatch)
    for path in checkpoint.iterdir():
        path.unlink()
    checkpoint.rmdir()
    with pytest.raises(ValueError, match="no saved checkpoint"):
        jira_sft.train_jira_sft(**options)


@pytest.mark.parametrize("filename", ["run-config.json", "trainer_state.json", "adapter_config.json"])
def test_resume_unreadable_json_fails_before_hardware(
        filename, configured_profiles, tmp_path, monkeypatch):
    options, run, checkpoint, _ = _bound_resume_fixture(tmp_path, configured_profiles, monkeypatch)
    path = (run if filename == "run-config.json" else checkpoint) / filename
    path.write_text("{interrupted JSON", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        jira_sft.train_jira_sft(**options)


@pytest.mark.parametrize("mutation, error", [
    ({"global_step": 9}, "global_step"),
    ({"max_steps": 99}, "max_steps"),
    ({"best_metric": float("nan")}, "finite"),
    ({"best_model_checkpoint": None}, "no recorded validation-best"),
    ({"best_model_checkpoint": "/unrelated/checkpoint-10"}, "does not belong"),
    ({"stateful_callbacks": {}}, "early-stopping state"),
])
def test_resume_rejects_incompatible_checkpoint_metadata_before_hardware(
        mutation, error, configured_profiles, tmp_path, monkeypatch):
    options, _, checkpoint, _ = _bound_resume_fixture(tmp_path, configured_profiles, monkeypatch)
    path = checkpoint / "trainer_state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    state.update(mutation)
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        jira_sft.train_jira_sft(**options)


def test_resume_requires_the_retained_best_adapter(configured_profiles, tmp_path, monkeypatch):
    options, run, checkpoint, _ = _bound_resume_fixture(tmp_path, configured_profiles, monkeypatch)
    base = configured_profiles.require_model_profile("jira-func").model_path
    newer = _checkpoint_fixture(run, base, step=20, best=checkpoint)
    (checkpoint / "adapter_model.safetensors").unlink()
    with pytest.raises(ValueError, match="checkpoint is incomplete: checkpoint-10/adapter_model"):
        jira_sft.train_jira_sft(**options)
    assert newer.is_dir()


def test_resume_adapter_must_name_its_bound_base_model(configured_profiles, tmp_path, monkeypatch):
    options, _, checkpoint, _ = _bound_resume_fixture(tmp_path, configured_profiles, monkeypatch)
    (checkpoint / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": str(tmp_path / "unrelated-base"),
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="adapter base model does not match"):
        jira_sft.train_jira_sft(**options)


def test_resume_already_exported_run_is_rejected_without_rewriting_manifest(
        configured_profiles, tmp_path, monkeypatch):
    options, run, _, _ = _bound_resume_fixture(tmp_path, configured_profiles, monkeypatch)
    manifest = run / "training-manifest.json"
    manifest.write_text('{"existing": "completion evidence"}', encoding="utf-8")
    before = manifest.read_bytes()
    with pytest.raises(ValueError, match="completed training manifest"):
        jira_sft.train_jira_sft(**options)
    assert manifest.read_bytes() == before


@pytest.mark.parametrize("finished", ["budget", "early-stop"])
def test_resume_finished_optimization_requires_export_reconciliation(
        finished, configured_profiles, tmp_path, monkeypatch):
    options, _, checkpoint, _ = _bound_resume_fixture(tmp_path, configured_profiles, monkeypatch)
    if finished == "budget":
        options["max_steps"] = 10
        config_path = options["resume_run_directory"] / "run-config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["max_steps"] = 10
        config_path.write_text(json.dumps(config), encoding="utf-8")
    path = checkpoint / "trainer_state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    if finished == "budget":
        state["max_steps"] = 10
    else:
        state["stateful_callbacks"]["EarlyStoppingCallback"]["attributes"]["early_stopping_patience_counter"] = 3
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="already finished.*Reconcile"):
        jira_sft.train_jira_sft(**options)


@pytest.mark.parametrize("name, value", [
    ("max_steps", True), ("max_steps", 2001), ("max_length", 255), ("max_length", 1280.5),
    ("gradient_accumulation_steps", 0), ("gradient_accumulation_steps", True),
    ("checkpoint_interval", 0), ("checkpoint_interval", True),
    ("early_stopping_patience", 0), ("early_stopping_patience", 1.5),
    ("lora_r", 0), ("lora_alpha", -1), ("lora_dropout", 1.0), ("lora_dropout", float("nan")),
    ("learning_rate", -0.01), ("learning_rate", float("inf")), ("learning_rate", True),
    ("seed", -1), ("seed", 2**32), ("seed", False),
])
def test_invalid_training_recipe_fails_before_corpus_or_hardware(name, value, tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid numeric options reached corpus or hardware preflight")

    monkeypatch.setattr(jira_sft, "verify_jira_sft_corpus", unexpected)
    monkeypatch.setattr(jira_sft, "assert_training_versions", unexpected)
    options = dict(
        project_root=PROJECT_ROOT, corpus_directory=tmp_path / "corpus",
        output_root=tmp_path / "output", allow_training=True, max_steps=40,
    )
    options[name] = value
    with pytest.raises(ValueError, match=name if name != "max_steps" else "max_steps"):
        jira_sft.train_jira_sft(**options)
    assert not options["output_root"].exists()
