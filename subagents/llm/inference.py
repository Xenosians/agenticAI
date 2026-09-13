from typing import (
    Protocol,
)

from subagents.llm.base import (
    GenerationOutput,
)

from subagents.llm.model_manager import (
    ModelManager,
)

from subagents.llm.observability import (
    InferenceMetric,
    MemorySampler,
    MetricSink,
    capture_cuda_memory,
    emit_inference_metric,
)

from subagents.llm.scheduler import (
    GpuScheduler,
    InferencePriority,
)


class InferenceEngine(
    Protocol
):
    async def generate(
        self,
        *,
        model_key: str,
        messages: list[
            dict[str, str]
        ],
        max_new_tokens: int,
        priority: int,
    ) -> str:
        ...


class InferenceCoordinator:
    """
    Central inference entry point.

    Scheduling policy remains unchanged.

    This layer now measures:

    - queue wait
    - execution latency
    - total scheduler latency
    - cold/hot model state
    - VRAM state
    - exact generated-token count when supported
    - tokens/sec when token count is available
    - true TTFT when a backend eventually supports it
    """

    def __init__(
        self,
        model_manager: ModelManager,
        scheduler: GpuScheduler,
        memory_sampler: (
            MemorySampler
            | None
        ) = None,
        metric_sink: (
            MetricSink
            | None
        ) = None,
    ) -> None:
        self.model_manager = (
            model_manager
        )

        self.scheduler = (
            scheduler
        )

        self.memory_sampler = (
            memory_sampler
            if memory_sampler
            is not None
            else capture_cuda_memory
        )

        self.metric_sink = (
            metric_sink
            if metric_sink
            is not None
            else emit_inference_metric
        )

    async def generate(
        self,
        *,
        model_key: str,
        messages: list[
            dict[str, str]
        ],
        max_new_tokens: int,
        priority: int,
    ) -> str:
        if not (
            self.model_manager
            .exists(
                model_key
            )
        ):
            raise KeyError(
                f"Model '{model_key}' "
                "is not registered."
            )

        loaded_before = (
            self.model_manager
            .is_loaded(
                model_key
            )
        )

        memory_before = (
            self.memory_sampler()
        )

        def work(
        ) -> GenerationOutput:
            return (
                self.model_manager
                .generate_observed(
                    model_key=(
                        model_key
                    ),

                    messages=(
                        messages
                    ),

                    max_new_tokens=(
                        max_new_tokens
                    ),
                )
            )

        scheduled = (
            await self.scheduler
            .run_observed(
                priority=(
                    priority
                ),

                work=(
                    work
                ),
            )
        )

        generation = (
            scheduled.value
        )

        memory_after = (
            self.memory_sampler()
        )

        loaded_after = (
            self.model_manager
            .is_loaded(
                model_key
            )
        )

        tokens_per_second: (
            float | None
        ) = None

        if (
            generation.generated_tokens
            is not None
            and generation.generated_tokens
            >= 0
            and scheduled.metrics
            .execution_seconds
            > 0
        ):
            tokens_per_second = (
                generation.generated_tokens
                / scheduled.metrics
                .execution_seconds
            )

        self.metric_sink(
            InferenceMetric(
                operation=(
                    "generate"
                ),

                model_key=(
                    model_key
                ),

                priority=int(
                    priority
                ),

                loaded_before=(
                    loaded_before
                ),

                loaded_after=(
                    loaded_after
                ),

                queue_wait_seconds=(
                    scheduled
                    .metrics
                    .queue_wait_seconds
                ),

                execution_seconds=(
                    scheduled
                    .metrics
                    .execution_seconds
                ),

                total_seconds=(
                    scheduled
                    .metrics
                    .total_seconds
                ),

                generated_tokens=(
                    generation
                    .generated_tokens
                ),

                tokens_per_second=(
                    tokens_per_second
                ),

                first_token_seconds=(
                    generation
                    .first_token_seconds
                ),

                memory_before=(
                    memory_before
                ),

                memory_after=(
                    memory_after
                ),
            )
        )

        return (
            generation.text
        )

    async def warm(
        self,
        model_key: str,
    ) -> None:
        if not (
            self.model_manager
            .exists(
                model_key
            )
        ):
            raise KeyError(
                f"Model '{model_key}' "
                "is not registered."
            )

        loaded_before = (
            self.model_manager
            .is_loaded(
                model_key
            )
        )

        memory_before = (
            self.memory_sampler()
        )

        scheduled = (
            await self.scheduler
            .run_observed(
                priority=(
                    InferencePriority
                    .STARTUP
                ),

                work=lambda: (
                    self.model_manager
                    .load(
                        model_key
                    )
                ),
            )
        )

        memory_after = (
            self.memory_sampler()
        )

        loaded_after = (
            self.model_manager
            .is_loaded(
                model_key
            )
        )

        self.metric_sink(
            InferenceMetric(
                operation=(
                    "warm"
                ),

                model_key=(
                    model_key
                ),

                priority=int(
                    InferencePriority
                    .STARTUP
                ),

                loaded_before=(
                    loaded_before
                ),

                loaded_after=(
                    loaded_after
                ),

                queue_wait_seconds=(
                    scheduled
                    .metrics
                    .queue_wait_seconds
                ),

                execution_seconds=(
                    scheduled
                    .metrics
                    .execution_seconds
                ),

                total_seconds=(
                    scheduled
                    .metrics
                    .total_seconds
                ),

                generated_tokens=None,

                tokens_per_second=None,

                first_token_seconds=None,

                memory_before=(
                    memory_before
                ),

                memory_after=(
                    memory_after
                ),
            )
        )