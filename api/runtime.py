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

    This is NOT a Singleton Pattern. The application lifecycle
    owns this object and passes/retrieves it explicitly through
    FastAPI app.state.

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