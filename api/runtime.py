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

from learning.recorder import (
    TrajectoryRecorder,
)

from subagents.core.orchestrator import (
    Orchestrator,
)

from subagents.core.tool_gateway import (
    ToolGateway,
)

from subagents.llm.inference import (
    InferenceCoordinator,
)

from subagents.llm.model_manager import (
    ModelManager,
)

from subagents.llm.scheduler import (
    GpuScheduler,
)


@dataclass
class ApplicationRuntime:
    """
    Explicit process-owned runtime graph for the AI service.

    FastAPI lifespan creates exactly one instance.

    Tests may construct isolated runtime graphs independently.
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
        asyncio.Task | None
    ) = None

    ready: bool = False

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
            )
        )