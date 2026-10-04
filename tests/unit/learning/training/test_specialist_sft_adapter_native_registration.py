from __future__ import annotations

import json

from pathlib import Path

from learning.training.jira_sft import (
    fingerprint_directory,
)

from learning.training.specialist_sft import (
    SpecialistSftTrainingManifest,
    register_candidate_profile,
)


def _artifact(
    directory: Path,
    payload: bytes,
) -> str:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        directory
        / "adapter_model.safetensors"
    ).write_bytes(payload)

    (
        directory
        / "adapter_config.json"
    ).write_text(
        "{}\n",
        encoding="utf-8",
    )

    return fingerprint_directory(
        directory
    )


def test_register_candidate_profile_is_adapter_native(
    tmp_path: Path,
) -> None:
    base_model = (
        tmp_path
        / "base-model"
    )
    base_model.mkdir()

    predecessor_adapter = (
        tmp_path
        / "developer-v1"
    )

    predecessor_sha = _artifact(
        predecessor_adapter,
        b"old-developer-adapter",
    )

    successor_adapter = (
        tmp_path
        / "developer-shell-v3"
    )

    successor_sha = _artifact(
        successor_adapter,
        b"new-developer-shell-v3-adapter",
    )

    env_path = (
        tmp_path
        / ".env"
    )

    profiles = {
        "developer-func-trained": {
            "enabled": True,
            "backend": "ministral",
            "model_path": str(
                base_model
            ),
            "adapter_path": str(
                predecessor_adapter
            ),
            "quantization": "bnb4",
            "worker_prompt_profile": "compact",
        },
    }

    env_path.write_text(
        "MODEL_PROFILES='"
        + json.dumps(
            profiles,
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
        candidate_model_key=(
            "developer-func-trained-shell-v3"
        ),
        worker_prompt_profile="compact",
        base_model_path=str(
            base_model
        ),
        base_adapter_path=str(
            predecessor_adapter
        ),
        base_adapter_sha256=(
            predecessor_sha
        ),
        continued_from_adapter=True,
        output_directory=str(
            tmp_path
            / "training-run"
        ),
        adapter_directory=str(
            successor_adapter
        ),
        merged_model_directory=None,
        corpus_directory=str(
            tmp_path
            / "corpus"
        ),
        corpus_train_sha256="a" * 64,
        corpus_validation_sha256="b" * 64,
        external_sft_sources=[],
        max_steps=1,
        learning_rate=2e-4,
        max_length=1536,
        lora_r=16,
        lora_alpha=32,
        trainable_parameters=1,
        total_parameters=2,
        trainable_ratio=0.5,
        adapter_sha256=(
            successor_sha
        ),
        merged_model_sha256=None,
        training_versions={},
    )

    manifest_path = (
        tmp_path
        / "training-manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest.model_dump(
                mode="json",
                by_alias=True,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    candidate = (
        register_candidate_profile(
            env_path=env_path,
            training_manifest_path=(
                manifest_path
            ),
        )
    )

    assert candidate["model_path"] == str(
        base_model.resolve()
    )

    assert candidate["adapter_path"] == str(
        successor_adapter.resolve()
    )

    assert (
        candidate["adapter_path"]
        != str(
            predecessor_adapter.resolve()
        )
    )

    assert candidate["enabled"] is True

    assert (
        candidate["worker_prompt_profile"]
        == "compact"
    )

    # No merged model needs to exist.
    assert not (
        tmp_path
        / "merged"
    ).exists()

    raw = (
        env_path
        .read_text(
            encoding="utf-8"
        )
        .strip()
        .split(
            "=",
            1,
        )[1]
    )

    assert (
        raw.startswith("'")
        and raw.endswith("'")
    )

    stored = json.loads(
        raw[1:-1]
    )

    registered = stored[
        "developer-func-trained-shell-v3"
    ]

    assert registered["model_path"] == str(
        base_model.resolve()
    )

    assert registered["adapter_path"] == str(
        successor_adapter.resolve()
    )


def test_registration_rejects_modified_adapter(
    tmp_path: Path,
) -> None:
    base_model = (
        tmp_path
        / "base-model"
    )
    base_model.mkdir()

    adapter = (
        tmp_path
        / "adapter"
    )

    original_sha = _artifact(
        adapter,
        b"original",
    )

    env_path = (
        tmp_path
        / ".env"
    )

    env_path.write_text(
        "MODEL_PROFILES='"
        + json.dumps(
            {
                "developer-func-trained": {
                    "enabled": True,
                    "backend": "ministral",
                    "model_path": str(
                        base_model
                    ),
                    "adapter_path": None,
                    "quantization": "bnb4",
                    "worker_prompt_profile": (
                        "compact"
                    ),
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
        candidate_model_key="candidate",
        worker_prompt_profile="compact",
        base_model_path=str(
            base_model
        ),
        output_directory=str(
            tmp_path
            / "run"
        ),
        adapter_directory=str(
            adapter
        ),
        merged_model_directory=None,
        corpus_directory=str(
            tmp_path
            / "corpus"
        ),
        corpus_train_sha256="a" * 64,
        corpus_validation_sha256="b" * 64,
        max_steps=1,
        learning_rate=2e-4,
        max_length=1536,
        lora_r=16,
        lora_alpha=32,
        trainable_parameters=1,
        total_parameters=2,
        trainable_ratio=0.5,
        adapter_sha256=original_sha,
        merged_model_sha256=None,
        training_versions={},
    )

    manifest_path = (
        tmp_path
        / "training-manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest.model_dump(
                mode="json",
                by_alias=True,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # Artifact changed after its training manifest was created.
    (
        adapter
        / "adapter_model.safetensors"
    ).write_bytes(
        b"tampered"
    )

    try:
        register_candidate_profile(
            env_path=env_path,
            training_manifest_path=(
                manifest_path
            ),
        )
    except ValueError as exc:
        assert (
            "adapter fingerprint"
            in str(exc).lower()
        )
    else:
        raise AssertionError(
            "Modified adapter was accepted."
        )
