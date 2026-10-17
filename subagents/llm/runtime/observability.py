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

    loaded_before: bool | None

    loaded_after: bool | None

    queue_wait_seconds: float | None

    execution_seconds: float | None

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
        CudaMemorySnapshot | None
    )

    memory_after: (
        CudaMemorySnapshot | None
    )

    stage: str
    repair_of: str | None
    adapter: str
    outcome: str
    error_type: str | None
    input_message_count: int
    estimated_input_tokens: int | None
    request_token_budget: int | None
    output_token_cap: int | None
    dropped_messages: int
    # Registered specialist name whose task ran, or None for hub stages.
    role: str | None = None


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
    # Labels come from trusted enums/adapter dispatch; prompt, response,
    # credential, provider URL and exception text never enter this event.
    print(
        "[INFERENCE_METRIC] "
        f"operation={metric.operation} stage={metric.stage} "
        f"repair_of={metric.repair_of} adapter={metric.adapter} "
        f"outcome={metric.outcome} error_type={metric.error_type} "
        f"total_ms={metric.total_seconds * 1000:.2f} "
        f"messages={metric.input_message_count} "
        f"estimated_input_tokens={metric.estimated_input_tokens} "
        f"request_budget={metric.request_token_budget} "
        f"output_cap={metric.output_token_cap} "
        f"dropped_messages={metric.dropped_messages} "
        f"role={metric.role}"
    )

    if metric.memory_before is None or metric.memory_after is None:
        return

    print(
        "[GPU_METRIC] "
        f"operation={metric.operation} "
        f"model={metric.model_key!r} "
        f"stage={metric.stage} "
        f"priority={metric.priority} "
        f"loaded_before="
        f"{metric.loaded_before} "
        f"loaded_after="
        f"{metric.loaded_after} "
        f"queue_wait_ms="
        f"{_number_or_na(metric.queue_wait_seconds * 1000 if metric.queue_wait_seconds is not None else None)} "
        f"execution_ms="
        f"{_number_or_na(metric.execution_seconds * 1000 if metric.execution_seconds is not None else None)} "
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
