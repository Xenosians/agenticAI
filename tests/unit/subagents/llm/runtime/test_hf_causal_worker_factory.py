import subagents.llm.runtime.factory as factory

from config import (
    ModelProfileSettings,
)


def test_factory_builds_generic_hf_causal_worker(
    monkeypatch,
    tmp_path,
):

    captured = {}

    class FakeBackend:

        def __init__(
            self,
            *,
            model_path,
            model_load_kwargs,
        ):
            captured[
                "model_path"
            ] = model_path

            captured[
                "model_load_kwargs"
            ] = model_load_kwargs

    monkeypatch.setattr(
        factory,
        "HFCausalWorkerBackend",
        FakeBackend,
    )

    profile = (
        ModelProfileSettings(
            backend="hf-causal",
            model_path=tmp_path,
            quantization="none",
            compute_dtype="float32",
            model_dtype="float32",
            device_map="cpu",
        )
    )

    backend = (
        factory.build_model_backend(
            profile
        )
    )

    assert isinstance(
        backend,
        FakeBackend,
    )

    assert (
        captured[
            "model_path"
        ]
        == tmp_path
    )

    kwargs = (
        captured[
            "model_load_kwargs"
        ]
    )

    assert (
        kwargs[
            "local_files_only"
        ]
        is True
    )

    assert (
        kwargs[
            "device_map"
        ]
        == "cpu"
    )
