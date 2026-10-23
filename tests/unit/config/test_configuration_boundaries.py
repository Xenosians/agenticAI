from pathlib import (
    Path,
)

import pytest
import torch

from pydantic import (
    ValidationError,
)

from config import (
    ModelProfileSettings,
    Settings,
)

from services.assets import (
    MockAssetService,
    build_asset_service,
)

from services.knowledge import (
    MockKnowledgeService,
    build_knowledge_service,
)

from subagents.hub import (
    build_hub,
)

from subagents.llm.runtime import (
    factory as model_factory,
)


def build_settings(
    **overrides,
) -> Settings:

    values = {
        "asset_backend":
            "mock",

        "knowledge_backend":
            "mock",
    }

    values.update(
        overrides
    )

    return (
        Settings(
            _env_file=None,
            **values,
        )
    )


def test_router_prompt_profile_defaults_to_standard_independently_of_worker_profile():
    profile = ModelProfileSettings(backend="hf-causal", worker_prompt_profile="compact")
    assert profile.router_prompt_profile == "standard"


def test_grounded_compact_worker_profile_is_an_explicit_normalized_opt_in():
    profile = ModelProfileSettings(
        backend="hf-causal", worker_prompt_profile=" COMPACT_GROUNDED ",
    )
    assert profile.worker_prompt_profile == "compact_grounded"
    assert profile.router_prompt_profile == "standard"


def test_grounded_compact_worker_contract_is_not_a_router_profile():
    with pytest.raises(ValidationError):
        ModelProfileSettings(backend="hf-causal", router_prompt_profile="compact_grounded")


@pytest.mark.parametrize("value", ["standard", " compact ", "COMPACT"])
def test_router_prompt_profile_is_explicit_and_normalized(value):
    assert ModelProfileSettings(backend="hf-causal", router_prompt_profile=value).router_prompt_profile == value.strip().lower()


@pytest.mark.parametrize("value", [None, "", "unknown", 1])
def test_invalid_router_prompt_profile_is_rejected(value):
    with pytest.raises(ValidationError):
        ModelProfileSettings(backend="hf-causal", router_prompt_profile=value)


def test_asset_and_knowledge_provider_selection_is_configured():

    settings = (
        build_settings()
    )

    assert isinstance(
        build_asset_service(
            settings
        ),
        MockAssetService,
    )

    assert isinstance(
        build_knowledge_service(
            settings
        ),
        MockKnowledgeService,
    )


@pytest.mark.parametrize(
    "field_name",
    [
        "asset_backend",
        "knowledge_backend",
    ],
)
def test_unknown_provider_fails_validation(
    field_name,
):

    with pytest.raises(
        ValidationError
    ):

        build_settings(
            **{
                field_name:
                    "not-configured",
            }
        )


def test_generation_budgets_are_validated_settings():

    settings = (
        build_settings(
            hub_router_max_new_tokens=111,

            specialist_max_new_tokens=222,

            primary_response_max_new_tokens=333,

            primary_synthesis_max_new_tokens=444,
        )
    )

    assert (
        settings
        .hub_router_max_new_tokens
        == 111
    )

    assert (
        settings
        .specialist_max_new_tokens
        == 222
    )

    assert (
        settings
        .primary_response_max_new_tokens
        == 333
    )

    assert (
        settings
        .primary_synthesis_max_new_tokens
        == 444
    )


@pytest.mark.parametrize(
    (
        "backend_name",
        "backend_attribute",
    ),
    [
        (
            "qwen-funccall",
            "QwenFuncCallBackend",
        ),

        (
            "qwen3",
            "Qwen3WorkerBackend",
        ),

        (
            "qwen-coder",
            "QwenCoderWorkerBackend",
        ),
    ],
)
def test_qwen_factory_uses_model_profile_runtime_settings(
    monkeypatch,
    tmp_path: Path,
    backend_name: str,
    backend_attribute: str,
):

    captured = {}

    class FakeBackend:

        def __init__(
            self,
            *,
            model_path,
            model_load_kwargs,
        ) -> None:

            captured[
                "model_path"
            ] = (
                model_path
            )

            captured[
                "model_load_kwargs"
            ] = (
                model_load_kwargs
            )

    monkeypatch.setattr(
        model_factory,
        backend_attribute,
        FakeBackend,
    )

    monkeypatch.setattr(
        model_factory,
        "release_unused_accelerator_memory",
        lambda: None,
    )

    profile = (
        ModelProfileSettings(
            backend=(
                backend_name
            ),

            model_path=(
                tmp_path
            ),

            model_dtype=(
                "float16"
            ),

            compute_dtype=(
                "bfloat16"
            ),

            quantization=(
                "none"
            ),

            device_map=(
                "cpu"
            ),
        )
    )

    model_factory.build_model_backend(
        profile
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
            "torch_dtype"
        ]
        is torch.float16
    )

    assert (
        kwargs[
            "device_map"
        ]
        == "cpu"
    )

    assert (
        "quantization_config"
        not in kwargs
    )


@pytest.mark.parametrize("router_prompt_profile", ["standard", "compact"])
def test_build_hub_injects_generation_configuration(
    monkeypatch,
    tmp_path: Path,
    router_prompt_profile,
):

    import subagents.hub as hub_module

    monkeypatch.setattr(
        hub_module,
        "load_agent_directory",
        lambda _directory: [],
    )

    settings = (
        build_settings(
            agents_dir=(
                tmp_path
            ),

            hub_router_max_new_tokens=123,

            specialist_max_new_tokens=234,

            primary_response_max_new_tokens=345,

            primary_synthesis_max_new_tokens=456,

            model_profiles={
                "hub-main":
                    ModelProfileSettings(
                        router_prompt_profile=router_prompt_profile,
                        backend=(
                            "ministral"
                        ),

                        model_path=(
                            tmp_path
                        ),
                    ),
            },
        )
    )

    class FakeModelManager:

        def exists(
            self,
            _model_key,
        ) -> bool:

            return True

        def model_profile(
            self,
            model_key,
        ):

            return (
                settings
                .require_model_profile(
                    model_key
                )
            )

    class FakeInference:
        pass

    hub = (
        build_hub(
            settings=(
                settings
            ),

            model_manager=(
                FakeModelManager()
            ),

            inference=(
                FakeInference()
            ),

            tool_gateway=(
                object()
            ),
        )
    )

    assert (
        hub.router
        .max_new_tokens
        == 123
    )
    assert hub.router.router_prompt_profile == router_prompt_profile

    assert (
        hub.runtime
        .max_new_tokens
        == 234
    )

    assert (
        hub.primary_assistant
        .response_max_new_tokens
        == 345
    )

    assert (
        hub.primary_assistant
        .synthesis_max_new_tokens
        == 456
    )
