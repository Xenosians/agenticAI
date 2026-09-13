from typing import (
    Protocol,
)

from subagents.llm.model_manager import (
    ModelManager,
)

from subagents.llm.scheduler import (
    GpuScheduler,
    InferencePriority,
)


class InferenceEngine(
    Protocol
):
    """
    Interface consumed by reasoning/orchestration components.

    Callers request inference by logical model key.

    They never receive or own model backend objects.
    """

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

    This boundary hides:

    - model backend construction
    - cached model instances
    - GPU admission
    - inference priority

    Future GPU residency, eviction, model-version promotion,
    and scheduling policy can evolve behind this interface
    without changing Router, PrimaryAssistant, or AgentRuntime.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        scheduler: GpuScheduler,
    ) -> None:
        self.model_manager = (
            model_manager
        )

        self.scheduler = (
            scheduler
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
        """
        Execute one model generation through the shared
        scheduler.

        The caller never receives the underlying backend.
        """

        if not self.model_manager.exists(
            model_key
        ):
            raise KeyError(
                f"Model '{model_key}' "
                "is not registered."
            )

        def work() -> str:
            return (
                self.model_manager
                .generate(
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

        return await self.scheduler.run(
            priority=(
                priority
            ),
            work=work,
        )

    async def warm(
        self,
        model_key: str,
    ) -> None:
        """
        Load a model through the same resource-admission
        boundary used for normal inference.

        This will later allow startup warming to obey the same
        GPU ownership rules as runtime inference.
        """

        if not self.model_manager.exists(
            model_key
        ):
            raise KeyError(
                f"Model '{model_key}' "
                "is not registered."
            )

        await self.scheduler.run(
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