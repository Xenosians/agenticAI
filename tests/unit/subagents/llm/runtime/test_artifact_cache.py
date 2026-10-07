from pathlib import Path

from config import ModelProfileSettings

from subagents.llm.runtime.artifact_cache import (
    ModelArtifactCache,
    physical_model_signature,
)


def _profile(
    model_path: Path,
    *,
    adapter_path: Path | None = None,
    quantization: str = "bnb4",
):
    return ModelProfileSettings(
        backend="ministral",
        model_path=model_path,
        adapter_path=adapter_path,
        quantization=quantization,
        compute_dtype="bfloat16",
        device_map="auto",
    )


def test_artifact_key_changes_when_source_metadata_changes(
    tmp_path,
):
    model = tmp_path / "model"
    model.mkdir()

    config = model / "config.json"
    config.write_text(
        '{"model_type":"mistral3"}',
        encoding="utf-8",
    )

    shard = model / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"first")

    cache = ModelArtifactCache(
        tmp_path / "cache"
    )

    first = cache.key_for(
        _profile(model)
    )

    shard.write_bytes(
        b"changed-weight-artifact"
    )

    second = cache.key_for(
        _profile(model)
    )

    assert first != second


def test_artifact_key_changes_with_quantization_profile(
    tmp_path,
):
    model = tmp_path / "model"
    model.mkdir()

    (model / "config.json").write_text(
        "{}",
        encoding="utf-8",
    )

    cache = ModelArtifactCache(
        tmp_path / "cache"
    )

    bnb4 = cache.key_for(
        _profile(
            model,
            quantization="bnb4",
        )
    )

    none = cache.key_for(
        _profile(
            model,
            quantization="none",
        )
    )

    assert bnb4 != none


def test_physical_signature_ignores_adapter_identity(
    tmp_path,
):
    model = tmp_path / "model"
    model.mkdir()

    first_adapter = tmp_path / "adapter-a"
    first_adapter.mkdir()

    second_adapter = tmp_path / "adapter-b"
    second_adapter.mkdir()

    first = physical_model_signature(
        _profile(
            model,
            adapter_path=first_adapter,
        )
    )

    second = physical_model_signature(
        _profile(
            model,
            adapter_path=second_adapter,
        )
    )

    assert first == second


def test_physical_signature_changes_for_different_base(
    tmp_path,
):
    first_model = tmp_path / "model-a"
    first_model.mkdir()

    second_model = tmp_path / "model-b"
    second_model.mkdir()

    assert physical_model_signature(
        _profile(first_model)
    ) != physical_model_signature(
        _profile(second_model)
    )
