from pathlib import Path

from config import (
    ModelProfileSettings,
)

from subagents.llm.base import (
    LLMBackend,
)

from subagents.llm.ministral_hub import (
    MinistralHubBackend,
)

from subagents.llm.qwen_funcall import (
    QwenFuncCallBackend,
)

from subagents.llm.qwen3_worker import (
    Qwen3WorkerBackend,
)

from subagents.llm.qwen_coder_worker import (
    QwenCoderWorkerBackend,
)


def require_local_model_path(
    profile: ModelProfileSettings,
) -> Path:
    """
    Return the configured local model path.

    Current local model backends require an on-disk checkpoint.
    Future remote/provider backends may not.
    """

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
    Build one configured model backend.

    The caller deals only with a model profile.

    It does not need to know whether that profile belongs to
    the Hub, Account specialist, Access specialist, Developer
    specialist, or a future role.
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

    if backend_type == "ministral":
        return MinistralHubBackend(
            model_path=(
                model_path
            ),
            dequantize_fp8=(
                profile
                .dequantize_fp8
            ),
            offload_folder=(
                profile
                .offload_folder
            ),
        )

    if (
        backend_type
        == "qwen-funccall"
    ):
        return QwenFuncCallBackend(
            model_path=(
                model_path
            ),
        )

    if backend_type == "qwen3":
        return Qwen3WorkerBackend(
            model_path=(
                model_path
            ),
        )

    if (
        backend_type
        == "qwen-coder"
    ):
        return QwenCoderWorkerBackend(
            model_path=(
                model_path
            ),
        )

    raise ValueError(
        "Unsupported model backend: "
        f"{backend_type}"
    )


# ============================================================
# TRANSITIONAL COMPATIBILITY HELPERS
#
# Some current integration tests still import these names.
#
# They now delegate to the generic model-profile factory.
#
# We will remove these compatibility wrappers before the AI
# cleanup phase is frozen.
# ============================================================


def build_hub_backend(
    backend_type: str,
    model_path: Path,
    *,
    dequantize_fp8: bool = True,
    offload_folder: Path | None = None,
) -> LLMBackend:
    return build_model_backend(
        ModelProfileSettings(
            backend=(
                backend_type
            ),
            model_path=(
                model_path
            ),
            dequantize_fp8=(
                dequantize_fp8
            ),
            offload_folder=(
                offload_folder
            ),
        )
    )


def build_worker_backend(
    backend_type: str,
    model_path: Path,
) -> LLMBackend:
    return build_model_backend(
        ModelProfileSettings(
            backend=(
                backend_type
            ),
            model_path=(
                model_path
            ),
        )
    )