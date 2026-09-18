from __future__ import annotations

import asyncio

from dataclasses import (
    dataclass,
    field,
)

import httpx

from agent.approvals import (
    ApprovalManager,
)

from agent.completion_outbox import (
    CompletionOutbox,
)

from agent.mcp_client import (
    MCPRuntime,
)

from config import (
    Settings,
)

from learning.evidence.recorder import (
    TrajectoryRecorder,
)

from learning.integrations.runtime_hooks import (
    ContinualLearningRuntimeHooks,
)

from subagents.core.orchestration.orchestrator import (
    Orchestrator,
)

from subagents.core.tooling.gateway import (
    ToolGateway,
)

from subagents.llm.runtime.inference import (
    InferenceCoordinator,
)

from subagents.llm.runtime.model_manager import (
    ModelManager,
)

from subagents.llm.runtime.scheduler import (
    GpuScheduler,
)


@dataclass
class ApplicationRuntime:
    """
    Explicit process-owned runtime graph for the AI service.

    FastAPI lifespan creates exactly one instance.

    Tests may construct isolated runtime graphs independently.

    Learning capture is observability only.

    The learning subsystem is downstream from authoritative runtime
    execution and cannot authorize capabilities, execute tools, or
    mutate model weights.
    """

    settings: Settings

    completion_outbox: (
        CompletionOutbox
    )

    outbox_wakeup: (
        asyncio.Event
    )

    phoenix_client: (
        httpx.AsyncClient
    )

    mcp: MCPRuntime

    approvals: (
        ApprovalManager
    )

    model_manager: (
        ModelManager
    )

    gpu_scheduler: (
        GpuScheduler
    )

    inference: (
        InferenceCoordinator
    )

    tool_gateway: (
        ToolGateway
    )

    hub: Orchestrator

    active_jobs: dict[
        str,
        asyncio.Task,
    ] = field(
        default_factory=dict
    )

    outbox_task: (
        asyncio.Task
        | None
    ) = None

    ready: bool = False

    learning_hooks: (
        ContinualLearningRuntimeHooks
    ) = field(
        init=False
    )

    trajectory_recorder: (
        TrajectoryRecorder
    ) = field(
        init=False
    )

    def __post_init__(
        self,
    ) -> None:

        trajectory_path = (
            self.settings
            .resolve_runtime_path(
                self.settings
                .learning_trajectory_path
            )
        )

        # ====================================================
        # PHASE-5 CONTEXT HOOKS
        #
        # Context capture follows the same enable/disable switch
        # as trajectory capture.
        #
        # Context is:
        #     evidence only
        #     non-authoritative
        #     non-training-eligible by default
        #
        # Actual dataset promotion continues through the governed
        # curation/review/evaluation pipeline.
        # ====================================================

        self.learning_hooks = (
            ContinualLearningRuntimeHooks(
                enabled=(
                    self.settings
                    .learning_capture_enabled
                )
            )
        )

        self.trajectory_recorder = (
            TrajectoryRecorder(
                path=(
                    trajectory_path
                ),

                enabled=(
                    self.settings
                    .learning_capture_enabled
                ),

                hub_model=(
                    self.settings
                    .hub_model_key
                ),

                context_hooks=(
                    self.learning_hooks
                ),
            )
        )