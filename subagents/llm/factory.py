from pathlib import Path

from subagents.llm.base import LLMBackend
from subagents.llm.ministral_hub import (
    MinistralHubBackend,
)
from subagents.llm.qwen_funcall import (
    QwenFuncCallBackend,
)


def build_hub_backend(
    backend_type: str,
    model_path: Path,
    *,
    dequantize_fp8: bool = True,
) -> LLMBackend:
    backend_type = backend_type.strip().lower()

    if backend_type == "ministral":
        return MinistralHubBackend(
            model_path=model_path,
            dequantize_fp8=dequantize_fp8,
        )

    raise ValueError(
        f"Unsupported Hub backend: "
        f"{backend_type}"
    )


def build_worker_backend(
    backend_type: str,
    model_path: Path,
) -> LLMBackend:
    backend_type = backend_type.strip().lower()

    if backend_type == "qwen-funccall":
        return QwenFuncCallBackend(
            model_path
        )

    raise ValueError(
        f"Unsupported worker backend: "
        f"{backend_type}"
    )