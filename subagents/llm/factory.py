from pathlib import Path

from subagents.llm.base import LLMBackend
from subagents.llm.ministral_hub import (
    MinistralHubBackend,
)
from subagents.llm.qwen_funcall import (
    QwenFuncCallBackend,
)
from subagents.llm.qwen3_worker import (
    Qwen3WorkerBackend,
)


def build_hub_backend(
    backend_type: str,
    model_path: Path,
    *,
    dequantize_fp8: bool = True,
    offload_folder: Path | None = None,
) -> LLMBackend:
    """
    Build the Main Hub backend.

    Current Hub:
        Ministral 3B
    """

    backend_type = (
        backend_type
        .strip()
        .lower()
    )

    if backend_type == "ministral":
        return MinistralHubBackend(
            model_path=model_path,
            dequantize_fp8=(
                dequantize_fp8
            ),
            offload_folder=(
                offload_folder
            ),
        )

    raise ValueError(
        f"Unsupported Hub backend: {backend_type}"
    )


def build_worker_backend(
    backend_type: str,
    model_path: Path,
) -> LLMBackend:
    """
    Build specialist-worker backends.

    Account:
        Qwen2.5-0.5B FuncCall

    Access:
        Qwen3-0.6B
    """

    backend_type = (
        backend_type
        .strip()
        .lower()
    )

    if backend_type == "qwen-funccall":
        return QwenFuncCallBackend(
            model_path=model_path,
        )

    if backend_type == "qwen3":
        return Qwen3WorkerBackend(
            model_path=model_path,
        )

    raise ValueError(
        f"Unsupported worker backend: {backend_type}"
    )