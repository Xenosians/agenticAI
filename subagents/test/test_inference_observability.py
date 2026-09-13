import asyncio

from subagents.llm.base import (
    GenerationOutput,
)

from subagents.llm.inference import (
    InferenceCoordinator,
)

from subagents.llm.observability import (
    CudaMemorySnapshot,
)

from subagents.llm.scheduler import (
    GpuScheduler,
    InferencePriority,
)


class FakeModelManager:
    def __init__(
        self,
    ):
        self.loaded = (
            False
        )

    def exists(
        self,
        model_key,
    ):
        return (
            model_key
            == "test-model"
        )

    def is_loaded(
        self,
        model_key,
    ):
        return (
            self.loaded
        )

    def load(
        self,
        model_key,
    ):
        self.loaded = (
            True
        )

        return object()

    def generate_observed(
        self,
        *,
        model_key,
        messages,
        max_new_tokens,
    ):
        self.loaded = (
            True
        )

        return GenerationOutput(
            text=(
                "generated response"
            ),

            generated_tokens=32,

            first_token_seconds=0.25,
        )


def fake_memory(
):
    return (
        CudaMemorySnapshot(
            available=True,

            device_index=0,

            device_name=(
                "Fake GPU"
            ),

            free_bytes=8_000,

            total_bytes=16_000,

            allocated_bytes=4_000,

            reserved_bytes=5_000,
        )
    )


def test_generate_emits_observability():
    async def scenario():
        manager = (
            FakeModelManager()
        )

        metrics = []

        inference = (
            InferenceCoordinator(
                model_manager=(
                    manager
                ),

                scheduler=(
                    GpuScheduler()
                ),

                memory_sampler=(
                    fake_memory
                ),

                metric_sink=(
                    metrics.append
                ),
            )
        )

        response = (
            await inference.generate(
                model_key=(
                    "test-model"
                ),

                messages=[
                    {
                        "role":
                            "user",

                        "content":
                            "hello",
                    }
                ],

                max_new_tokens=16,

                priority=(
                    InferencePriority
                    .HUB_ROUTING
                ),
            )
        )

        assert (
            response
            == "generated response"
        )

        assert (
            len(
                metrics
            )
            == 1
        )

        metric = (
            metrics[
                0
            ]
        )

        assert (
            metric.operation
            == "generate"
        )

        assert (
            metric.model_key
            == "test-model"
        )

        assert (
            metric.loaded_before
            is False
        )

        assert (
            metric.loaded_after
            is True
        )

        assert (
            metric.generated_tokens
            == 32
        )

        assert (
            metric.tokens_per_second
            is not None
        )

        assert (
            metric.tokens_per_second
            > 0
        )

        assert (
            metric.first_token_seconds
            == 0.25
        )

        assert (
            metric.queue_wait_seconds
            >= 0
        )

        assert (
            metric.execution_seconds
            >= 0
        )

        assert (
            metric.total_seconds
            >= metric.execution_seconds
        )

        assert (
            metric.memory_before
            .device_name
            == "Fake GPU"
        )

    asyncio.run(
        scenario()
    )


def test_warm_emits_observability():
    async def scenario():
        manager = (
            FakeModelManager()
        )

        metrics = []

        inference = (
            InferenceCoordinator(
                model_manager=(
                    manager
                ),

                scheduler=(
                    GpuScheduler()
                ),

                memory_sampler=(
                    fake_memory
                ),

                metric_sink=(
                    metrics.append
                ),
            )
        )

        await inference.warm(
            "test-model"
        )

        assert (
            len(
                metrics
            )
            == 1
        )

        metric = (
            metrics[
                0
            ]
        )

        assert (
            metric.operation
            == "warm"
        )

        assert (
            metric.loaded_before
            is False
        )

        assert (
            metric.loaded_after
            is True
        )

        assert (
            metric.generated_tokens
            is None
        )

        assert (
            metric.tokens_per_second
            is None
        )

        assert (
            metric.first_token_seconds
            is None
        )

    asyncio.run(
        scenario()
    )