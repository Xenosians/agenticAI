from pathlib import Path

from config import (
    ModelProfileSettings,
)

from subagents.llm.runtime.base import (
    LLMBackend,
)

from subagents.llm.runtime.memory import (
    release_unused_accelerator_memory,
)

# The backends and the Transformers load profile import PyTorch and
# Transformers, which take tens of seconds. They load on first use, when a
# local model is built, so the API, the MCP server and tests that never load
# a model do not pay for them. Each stays a module attribute (tests replace
# them by name).
_LAZY_ATTRIBUTES = {
    "build_transformers_model_load_kwargs": "subagents.llm.runtime.load_profile",
    "MinistralHubBackend": "subagents.llm.backends.ministral_hub",
    "QwenFuncCallBackend": "subagents.llm.backends.qwen_funcall",
    "Qwen3WorkerBackend": "subagents.llm.backends.qwen3_worker",
    "QwenCoderWorkerBackend": "subagents.llm.backends.qwen_coder_worker",
    "HFCausalWorkerBackend": "subagents.llm.backends.hf_causal_worker",
}


def __getattr__(name: str):
    module_name = _LAZY_ATTRIBUTES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def _attribute(name: str):
    """A backend or helper, as replaced by a test or loaded on first use."""
    return globals()[name] if name in globals() else __getattr__(name)


def require_local_model_path(
    profile: ModelProfileSettings,
) -> Path:
    model_path = (
        profile.model_path
    )

    if model_path is None:
        raise RuntimeError(
            "Local model backend "
            f"'{profile.backend}' requires "
            "a model_path."
        )

    return model_path


def build_model_backend(
    profile: ModelProfileSettings,
    *,
    model_path_override: (
        Path | None
    ) = None,
) -> LLMBackend:
    """
    Construct one backend from a logical model profile.

    Model brand, quantization strategy, device placement, and
    checkpoint path are deployment concerns.

    Router/PrimaryAssistant/AgentRuntime continue to depend only
    on logical model keys.

    Before Transformers performs automatic device placement,
    reclaim accelerator allocations that are no longer owned by
    live model objects.

    This does NOT evict live models.

    It only prevents stale PyTorch allocator cache from making
    `device_map="auto"` reason from artificially reduced free
    VRAM.
    """

    backend_type = (
        profile.backend
        .strip()
        .lower()
    )

    model_path = (
        (
            Path(
                model_path_override
            )
            .expanduser()
            .resolve()
        )
        if (
            model_path_override
            is not None
        )
        else require_local_model_path(
            profile
        )
    )

    from services.lua_adapters.host_boundaries import inspect_data
    inspect_data("model", "admission", profile.model_dump(mode="json"),
                 {"backend": backend_type, "model_path": str(model_path)})

    # ========================================================
    # ACCELERATOR LIFECYCLE BOUNDARY
    # ========================================================

    release_unused_accelerator_memory()

    # ========================================================
    # MINISTRAL HUB
    # ========================================================

    if (
        backend_type
        == "ministral"
    ):
        return (
            _attribute("MinistralHubBackend")(
                model_path=(
                    model_path
                ),

                quantization=(
                    profile
                    .quantization
                ),

                compute_dtype=(
                    profile
                    .compute_dtype
                ),

                device_map=(
                    profile
                    .device_map
                ),

                dequantize_fp8=(
                    profile
                    .dequantize_fp8
                ),

                bnb_4bit_quant_type=(
                    profile
                    .bnb_4bit_quant_type
                ),

                bnb_4bit_use_double_quant=(
                    profile
                    .bnb_4bit_use_double_quant
                ),

                offload_folder=(
                    profile
                    .offload_folder
                ),
            )
        )

    # ========================================================
    # QWEN FUNCTION CALLING
    # ========================================================

    if (
        backend_type
        == "qwen-funccall"
    ):
        return (
            _attribute("QwenFuncCallBackend")(
                model_path=(
                    model_path
                ),

                model_load_kwargs=(
                    _attribute("build_transformers_model_load_kwargs")(
                        profile
                    )
                ),
            )
        )

    # ========================================================
    # QWEN3 WORKER
    # ========================================================

    if (
        backend_type
        == "qwen3"
    ):
        return (
            _attribute("Qwen3WorkerBackend")(
                model_path=(
                    model_path
                ),

                model_load_kwargs=(
                    _attribute("build_transformers_model_load_kwargs")(
                        profile
                    )
                ),
            )
        )

    # ========================================================
    # QWEN CODER WORKER
    # ========================================================

    if (
        backend_type
        == "qwen-coder"
    ):
        return (
            _attribute("QwenCoderWorkerBackend")(
                model_path=(
                    model_path
                ),

                model_load_kwargs=(
                    _attribute("build_transformers_model_load_kwargs")(
                        profile
                    )
                ),
            )
        )

    # ========================================================
    # GENERIC HF CAUSAL SPECIALIST WORKER
    # ========================================================

    if (
        backend_type
        == "hf-causal"
    ):
        return (
            _attribute("HFCausalWorkerBackend")(
                model_path=(
                    model_path
                ),

                model_load_kwargs=(
                    _attribute("build_transformers_model_load_kwargs")(
                        profile
                    )
                ),
            )
        )

    raise ValueError(
        "Unsupported model backend: "
        f"{backend_type}"
    )
