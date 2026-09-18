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

from subagents.llm.runtime.load_profile import (
    build_transformers_model_load_kwargs,
)

from subagents.llm.backends.ministral_hub import (
    MinistralHubBackend,
)

from subagents.llm.backends.qwen_funcall import (
    QwenFuncCallBackend,
)

from subagents.llm.backends.qwen3_worker import (
    Qwen3WorkerBackend,
)

from subagents.llm.backends.qwen_coder_worker import (
    QwenCoderWorkerBackend,
)


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
        require_local_model_path(
            profile
        )
    )

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
            MinistralHubBackend(
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
            QwenFuncCallBackend(
                model_path=(
                    model_path
                ),

                model_load_kwargs=(
                    build_transformers_model_load_kwargs(
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
            Qwen3WorkerBackend(
                model_path=(
                    model_path
                ),

                model_load_kwargs=(
                    build_transformers_model_load_kwargs(
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
            QwenCoderWorkerBackend(
                model_path=(
                    model_path
                ),

                model_load_kwargs=(
                    build_transformers_model_load_kwargs(
                        profile
                    )
                ),
            )
        )

    raise ValueError(
        "Unsupported model backend: "
        f"{backend_type}"
    )