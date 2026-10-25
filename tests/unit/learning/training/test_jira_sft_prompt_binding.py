from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import ModelProfileSettings, Settings
from learning.training import jira_sft


PROJECT_ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def configured_profiles(monkeypatch):
    settings = Settings(
        _env_file=None,
        model_profiles={
            "jira-func": ModelProfileSettings(
                backend="hf-causal", worker_prompt_profile="compact",
            ),
            "networking-base": ModelProfileSettings(
                backend="hf-causal", worker_prompt_profile="compact",
            ),
        },
    )
    monkeypatch.setattr(jira_sft, "Settings", lambda: settings)
    return settings


def _build_candidate(tmp_path, profile="compact_grounded"):
    directory = tmp_path / "candidate-corpus"
    manifest = jira_sft.build_jira_sft_corpus(
        project_root=PROJECT_ROOT, output_directory=directory,
        worker_prompt_profile=profile,
    )
    return directory, manifest


def _training_fixture(tmp_path, directory, corpus, *, legacy=False):
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text("{}\n", encoding="utf-8")
    (merged / "model.safetensors").write_bytes(b"isolated model identity fixture")
    values = dict(
        created_at="2026-10-23T00:00:00+00:00",
        base_model_key=corpus.base_model_key,
        base_model_path=str(tmp_path / "base"),
        output_directory=str(tmp_path),
        adapter_directory=str(tmp_path / "adapter"),
        merged_model_directory=str(merged),
        corpus_directory=str(directory),
        corpus_train_sha256=corpus.train_sha256,
        corpus_validation_sha256=corpus.validation_sha256,
        max_steps=1, learning_rate=0.0001, max_length=512,
        lora_r=4, lora_alpha=8, trainable_parameters=4,
        total_parameters=40, trainable_ratio=0.1,
        adapter_sha256="a" * 64,
        merged_model_sha256=jira_sft.fingerprint_directory(merged),
        training_versions={},
    )
    if not legacy:
        values.update(
            worker_prompt_profile=corpus.worker_prompt_profile,
            system_prompt_sha256=corpus.system_prompt_sha256,
            prompt_layout=corpus.prompt_layout,
        )
    manifest = jira_sft.JiraSftTrainingManifest(**values)
    path = tmp_path / "training-manifest.json"
    path.write_text(manifest.model_dump_json(by_alias=True), encoding="utf-8")
    env = tmp_path / ".env"
    env.write_text(
        "MODEL_PROFILES='" + json.dumps({
            "networking-base": {"backend": "hf-causal", "worker_prompt_profile": "compact"},
        }) + "'\n",
        encoding="utf-8",
    )
    return path, env


def test_candidate_prompt_is_explicit_without_changing_serving_or_networking(configured_profiles, tmp_path):
    original = configured_profiles.model_dump(mode="json")
    directory, manifest = _build_candidate(tmp_path)
    assert manifest.worker_prompt_profile == "compact_grounded"
    with pytest.raises(ValueError, match="prompt profile"):
        jira_sft.verify_jira_sft_corpus(project_root=PROJECT_ROOT, corpus_directory=directory)
    verified, train, validation, prompt = jira_sft.verify_jira_sft_corpus(
        project_root=PROJECT_ROOT, corpus_directory=directory,
        worker_prompt_profile="compact_grounded",
    )
    assert verified == manifest
    assert train and validation
    assert "COMPACT GROUNDED TOOL PROTOCOL" in prompt
    assert configured_profiles.model_dump(mode="json") == original


def test_default_corpus_keeps_configured_legacy_prompt(configured_profiles, tmp_path):
    directory = tmp_path / "legacy-corpus"
    manifest = jira_sft.build_jira_sft_corpus(
        project_root=PROJECT_ROOT, output_directory=directory,
    )
    verified, _, _, prompt = jira_sft.verify_jira_sft_corpus(
        project_root=PROJECT_ROOT, corpus_directory=directory,
    )
    assert verified == manifest
    assert manifest.worker_prompt_profile == "compact"
    assert "COMPACT TOOL PROTOCOL" in prompt
    assert "COMPACT GROUNDED TOOL PROTOCOL" not in prompt


def test_candidate_prompt_hash_drift_is_rejected(configured_profiles, tmp_path, monkeypatch):
    directory, _ = _build_candidate(tmp_path)
    original = jira_sft.build_worker_system_prompt
    monkeypatch.setattr(jira_sft, "build_worker_system_prompt", lambda *a, **kw: original(*a, **kw) + "\nchanged")
    with pytest.raises(ValueError, match="system prompt changed"):
        jira_sft.verify_jira_sft_corpus(
            project_root=PROJECT_ROOT, corpus_directory=directory,
            worker_prompt_profile="compact_grounded",
        )


@pytest.mark.parametrize("legacy", [False, True])
def test_registration_inherits_verified_corpus_prompt(configured_profiles, tmp_path, legacy):
    directory, corpus = _build_candidate(tmp_path, "compact" if legacy else "compact_grounded")
    manifest, env = _training_fixture(tmp_path, directory, corpus, legacy=legacy)
    profile = jira_sft.register_trained_candidate_profile(
        env_path=env, training_manifest_path=manifest,
    )
    assert profile["worker_prompt_profile"] == corpus.worker_prompt_profile
    profiles = json.loads(env.read_text(encoding="utf-8").split("=", 1)[1].strip().strip("'"))
    assert profiles["networking-base"] == {"backend": "hf-causal", "worker_prompt_profile": "compact"}


@pytest.mark.parametrize("field", [
    "corpus_train_sha256", "corpus_validation_sha256", "system_prompt_sha256", "prompt_layout",
])
def test_registration_rejects_unbound_training_artifacts_without_env_writes(configured_profiles, tmp_path, field):
    directory, corpus = _build_candidate(tmp_path)
    manifest, env = _training_fixture(tmp_path, directory, corpus)
    values = json.loads(manifest.read_text(encoding="utf-8"))
    values[field] = "runtime_task" if field == "prompt_layout" else "b" * 64
    manifest.write_text(json.dumps(values), encoding="utf-8")
    before = env.read_bytes()
    with pytest.raises(ValueError, match="does not match"):
        jira_sft.register_trained_candidate_profile(env_path=env, training_manifest_path=manifest)
    assert env.read_bytes() == before
    assert not (tmp_path / ".env.before-jira-sft-registration").exists()


def test_training_checks_explicit_candidate_binding_before_hardware_or_optimizer(configured_profiles, tmp_path, monkeypatch):
    directory, _ = _build_candidate(tmp_path)
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    (base / "tokenizer.json").write_text("{}", encoding="utf-8")
    (base / "model.safetensors").write_bytes(b"isolated model identity fixture")
    configured_profiles.model_profiles["jira-func"].model_path = base
    original_settings = configured_profiles.model_dump(mode="json")

    class HardwareGateReached(Exception):
        pass

    reached = []

    def stop_at_hardware_gate():
        reached.append(True)
        raise HardwareGateReached

    monkeypatch.setattr(jira_sft, "assert_training_versions", stop_at_hardware_gate)
    kwargs = dict(
        project_root=PROJECT_ROOT, corpus_directory=directory,
        output_root=tmp_path / "uncreated-training-output", allow_training=True, max_steps=1,
    )
    with pytest.raises(ValueError, match="prompt profile"):
        jira_sft.train_jira_sft(**kwargs)
    assert not reached
    with pytest.raises(HardwareGateReached):
        jira_sft.train_jira_sft(**kwargs, worker_prompt_profile="compact_grounded")
    assert reached == [True]
    assert not kwargs["output_root"].exists()
    assert configured_profiles.model_dump(mode="json") == original_settings


def test_training_cli_forwards_opt_in_without_enabling_optimizer(monkeypatch, tmp_path):
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
        "--worker-prompt-profile", "compact_grounded",
    ])
    with pytest.raises(StopBeforeTraining):
        jira_sft_train.main()
    assert observed["worker_prompt_profile"] == "compact_grounded"
    assert observed["allow_training"] is False
