from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from learning.training.jira_sft import fingerprint_directory
from learning.training.specialist_sft import (
    SpecialistSftTrainingManifest,
    _resolve_training_profile_paths,
    register_candidate_profile,
)


def _make_adapter(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"adapter")


def test_training_profile_resolves_existing_adapter(tmp_path: Path) -> None:
    model = tmp_path / "base"
    adapter = tmp_path / "adapter"
    model.mkdir()
    _make_adapter(adapter)
    profile = SimpleNamespace(model_path=model, adapter_path=adapter)

    model_path, adapter_path = _resolve_training_profile_paths(
        profile, label="developer-func-trained"
    )

    assert model_path == model.resolve()
    assert adapter_path == adapter.resolve()


def test_training_profile_rejects_missing_adapter(tmp_path: Path) -> None:
    model = tmp_path / "base"
    model.mkdir()
    profile = SimpleNamespace(
        model_path=model,
        adapter_path=tmp_path / "missing-adapter",
    )

    try:
        _resolve_training_profile_paths(
            profile, label="developer-func-trained"
        )
    except ValueError as exc:
        assert "adapter" in str(exc).lower()
    else:
        raise AssertionError("missing adapter should fail closed")


def test_registered_adapter_candidate_replaces_base_adapter_overlay(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}\n", encoding="utf-8")

    predecessor = tmp_path / "old-adapter"
    predecessor.mkdir()
    (predecessor / "adapter_model.safetensors").write_bytes(b"old")
    (predecessor / "adapter_config.json").write_text("{}\n", encoding="utf-8")

    successor = tmp_path / "adapter"
    successor.mkdir()
    (successor / "adapter_model.safetensors").write_bytes(b"new")
    (successor / "adapter_config.json").write_text("{}\n", encoding="utf-8")

    env_path = tmp_path / ".env"
    env_path.write_text(
        "MODEL_PROFILES='"
        + json.dumps(
            {
                "developer-func-trained": {
                    "backend": "transformers",
                    "model_path": str(base),
                    "adapter_path": str(predecessor),
                    "enabled": True,
                    "worker_prompt_profile": "compact",
                }
            },
            separators=(",", ":"),
        )
        + "'\n",
        encoding="utf-8",
    )

    manifest = SpecialistSftTrainingManifest(
        created_at="2026-10-05T00:00:00+00:00",
        specialist="developer-specialist",
        agent_name="developer-specialist",
        base_model_key="developer-func-trained",
        candidate_model_key="developer-func-trained-shell-v3",
        worker_prompt_profile="compact",
        base_model_path=str(base),
        base_adapter_path=str(predecessor),
        base_adapter_sha256=fingerprint_directory(predecessor),
        continued_from_adapter=True,
        output_directory=str(tmp_path),
        adapter_directory=str(successor),
        merged_model_directory=None,
        corpus_directory=str(tmp_path / "corpus"),
        corpus_train_sha256="train",
        corpus_validation_sha256="validation",
        external_sft_sources=[],
        max_steps=1,
        learning_rate=1e-4,
        max_length=512,
        lora_r=16,
        lora_alpha=32,
        trainable_parameters=1,
        total_parameters=2,
        trainable_ratio=0.5,
        adapter_sha256=fingerprint_directory(successor),
        merged_model_sha256=None,
        training_versions={},
    )
    manifest_path = tmp_path / "training-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json", by_alias=True)),
        encoding="utf-8",
    )

    registered = register_candidate_profile(
        env_path=env_path,
        training_manifest_path=manifest_path,
    )
    assert registered["model_path"] == str(base.resolve())
    assert registered["adapter_path"] == str(successor.resolve())
    assert registered["adapter_path"] != str(predecessor.resolve())
    assert registered["enabled"] is True

