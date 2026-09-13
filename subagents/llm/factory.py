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
    Return the configured local checkpoint path.

    Current local backends require an on-disk model.

    Remote/API-backed providers may eventually use a different
    provider implementation and therefore need a different
    validation path.
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
    Construct one backend from a logical model profile.

    Callers do not distinguish Hub models from specialist
    models here.

    Role -> logical model key mapping belongs to configuration.
    Backend implementation selection belongs here.
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

    if (
        backend_type
        == "ministral"
    ):
        return (
            MinistralHubBackend(
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
        )

    if (
        backend_type
        == "qwen-funccall"
    ):
        return (
            QwenFuncCallBackend(
                model_path=(
                    model_path
                ),
            )
        )

    if (
        backend_type
        == "qwen3"
    ):
        return (
            Qwen3WorkerBackend(
                model_path=(
                    model_path
                ),
            )
        )

    if (
        backend_type
        == "qwen-coder"
    ):
        return (
            QwenCoderWorkerBackend(
                model_path=(
                    model_path
                ),
            )
        )

    raise ValueError(
        "Unsupported model backend: "
        f"{backend_type}"
    )