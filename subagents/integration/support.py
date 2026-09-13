from dataclasses import (
    dataclass,
)

from pathlib import Path

from typing import (
    Any,
)

from agent.approvals import (
    ApprovalManager,
)

from agent.approval_store import (
    ApprovalStore,
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

from subagents.hub import (
    build_hub,
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
class HubIntegrationContext:
    """
    Explicit runtime graph used by integration tests.

    Each test owns its own approval database and dependency
    graph.

    Real-MCP tests may start/stop `mcp` themselves.

    Tests that must prove MCP is never reached may inject a
    guard/fake MCP object instead.
    """

    settings: Settings

    mcp: Any

    approval_store: (
        ApprovalStore
    )

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


def build_hub_integration_context(
    *,
    approval_db_path: Path,
    mcp: Any | None = None,
) -> HubIntegrationContext:
    settings = (
        Settings()
    )

    mcp_runtime = (
        mcp
        if mcp is not None
        else MCPRuntime()
    )

    approval_store = (
        ApprovalStore(
            approval_db_path
        )
    )

    approval_store.initialize()

    approvals = (
        ApprovalManager(
            store=(
                approval_store
            ),
            mcp=(
                mcp_runtime
            ),
        )
    )

    model_manager = (
        ModelManager(
            settings=(
                settings
            )
        )
    )

    gpu_scheduler = (
        GpuScheduler()
    )

    inference = (
        InferenceCoordinator(
            model_manager=(
                model_manager
            ),
            scheduler=(
                gpu_scheduler
            ),
        )
    )

    tool_gateway = (
        ToolGateway(
            approval_creator=(
                approvals
                .create_approval
            ),
            mcp=(
                mcp_runtime
            ),
        )
    )

    hub = (
        build_hub(
            settings=(
                settings
            ),
            model_manager=(
                model_manager
            ),
            inference=(
                inference
            ),
            tool_gateway=(
                tool_gateway
            ),
        )
    )

    return HubIntegrationContext(
        settings=(
            settings
        ),
        mcp=(
            mcp_runtime
        ),
        approval_store=(
            approval_store
        ),
        approvals=(
            approvals
        ),
        model_manager=(
            model_manager
        ),
        gpu_scheduler=(
            gpu_scheduler
        ),
        inference=(
            inference
        ),
        tool_gateway=(
            tool_gateway
        ),
        hub=(
            hub
        ),
    )