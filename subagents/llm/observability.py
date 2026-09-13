from __future__ import annotations

from dataclasses import (
    dataclass,
)

from typing import (
    Callable,
)


@dataclass(
    frozen=True
)
class CudaMemorySnapshot:
    available: bool

    device_index: (
        int | None
    )

    device_name: (
        str | None
    )

    free_bytes: (
        int | None
    )

    total_bytes: (
        int | None
    )

    allocated_bytes: (
        int | None
    )

    reserved_bytes: (
        int | None
    )

    error: (
        str | None
    ) = None


@dataclass(
    frozen=True
)
class InferenceMetric:
    operation: str

    model_key: str

    priority: int

    loaded_before: bool

    loaded_after: bool

    queue_wait_seconds: float

    execution_seconds: float

    total_seconds: float

    generated_tokens: (
        int | None
    )

    tokens_per_second: (
        float | None
    )

    first_token_seconds: (
        float | None
    )

    memory_before: (
        CudaMemorySnapshot
    )

    memory_after: (
        CudaMemorySnapshot
    )


MetricSink = Callable[
    [
        InferenceMetric
    ],
    None,
]


MemorySampler = Callable[
    [],
    CudaMemorySnapshot,
]


def capture_cuda_memory(
) -> CudaMemorySnapshot:
    try:
        import torch

        if not (
            torch.cuda
            .is_available()
        ):
            return (
                CudaMemorySnapshot(
                    available=False,
                    device_index=None,
                    device_name=None,
                    free_bytes=None,
                    total_bytes=None,
                    allocated_bytes=None,
                    reserved_bytes=None,
                )
            )

        device_index = (
            torch.cuda
            .current_device()
        )

        device_name = (
            torch.cuda
            .get_device_name(
                device_index
            )
        )

        (
            free_bytes,
            total_bytes,
        ) = (
            torch.cuda
            .mem_get_info(
                device_index
            )
        )

        allocated_bytes = (
            torch.cuda
            .memory_allocated(
                device_index
            )
        )

        reserved_bytes = (
            torch.cuda
            .memory_reserved(
                device_index
            )
        )

        return (
            CudaMemorySnapshot(
                available=True,

                device_index=(
                    device_index
                ),

                device_name=(
                    device_name
                ),

                free_bytes=int(
                    free_bytes
                ),

                total_bytes=int(
                    total_bytes
                ),

                allocated_bytes=int(
                    allocated_bytes
                ),

                reserved_bytes=int(
                    reserved_bytes
                ),
            )
        )

    except Exception as exc:
        return (
            CudaMemorySnapshot(
                available=False,
                device_index=None,
                device_name=None,
                free_bytes=None,
                total_bytes=None,
                allocated_bytes=None,
                reserved_bytes=None,
                error=repr(
                    exc
                ),
            )
        )


def _mib(
    value: (
        int | None
    ),
) -> str:
    if value is None:
        return "n/a"

    return (
        f"{value / (1024 * 1024):.1f}"
    )


def _seconds_or_na(
    value: (
        float | None
    ),
) -> str:
    if value is None:
        return "n/a"

    return (
        f"{value:.4f}"
    )


def _number_or_na(
    value: (
        float | None
    ),
) -> str:
    if value is None:
        return "n/a"

    return (
        f"{value:.2f}"
    )


def _memory_text(
    snapshot: CudaMemorySnapshot,
) -> str:
    if not (
        snapshot.available
    ):
        if snapshot.error:
            return (
                "cuda=unavailable "
                f"error={snapshot.error}"
            )

        return (
            "cuda=unavailable"
        )

    return (
        f"gpu={snapshot.device_index} "
        f"name={snapshot.device_name!r} "
        f"free_mib="
        f"{_mib(snapshot.free_bytes)} "
        f"total_mib="
        f"{_mib(snapshot.total_bytes)} "
        f"allocated_mib="
        f"{_mib(snapshot.allocated_bytes)} "
        f"reserved_mib="
        f"{_mib(snapshot.reserved_bytes)}"
    )


def emit_inference_metric(
    metric: InferenceMetric,
) -> None:
    print(
        "[GPU_METRIC] "
        f"operation={metric.operation} "
        f"model={metric.model_key!r} "
        f"priority={metric.priority} "
        f"loaded_before="
        f"{metric.loaded_before} "
        f"loaded_after="
        f"{metric.loaded_after} "
        f"queue_wait_ms="
        f"{metric.queue_wait_seconds * 1000:.2f} "
        f"execution_ms="
        f"{metric.execution_seconds * 1000:.2f} "
        f"total_ms="
        f"{metric.total_seconds * 1000:.2f} "
        f"generated_tokens="
        f"{metric.generated_tokens} "
        f"tokens_per_second="
        f"{_number_or_na(metric.tokens_per_second)} "
        f"ttft_seconds="
        f"{_seconds_or_na(metric.first_token_seconds)}"
    )

    print(
        "[GPU_METRIC] "
        f"operation={metric.operation} "
        f"model={metric.model_key!r} "
        "memory_before "
        f"{_memory_text(metric.memory_before)}"
    )

    print(
        "[GPU_METRIC] "
        f"operation={metric.operation} "
        f"model={metric.model_key!r} "
        "memory_after "
        f"{_memory_text(metric.memory_after)}"
    )