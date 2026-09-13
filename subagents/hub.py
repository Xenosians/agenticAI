from config import (
    Settings,
)

from subagents.core.loader import (
    load_agent_directory,
)

from subagents.core.registry import (
    AgentRegistry,
)

from subagents.core.llm_router import (
    LLMRouter,
)

from subagents.core.orchestrator import (
    Orchestrator,
)

from subagents.core.primary_assistant import (
    PrimaryAssistant,
)

from subagents.core.runtime import (
    AgentRuntime,
)

from subagents.core.tool_gateway import (
    ToolGateway,
)

from subagents.llm.inference import (
    InferenceEngine,
)

from subagents.llm.model_manager import (
    ModelManager,
)


def build_hub(
    *,
    settings: Settings,
    model_manager: ModelManager,
    inference: InferenceEngine,
    tool_gateway: ToolGateway,
) -> Orchestrator:
    """
    Build the Hub reasoning and specialist orchestration graph.

    This function owns no process-wide runtime resources.

    Runtime ownership belongs to the composition root:

        FastAPI lifespan
            |
            +-- Settings
            +-- ModelManager
            +-- GpuScheduler
            +-- InferenceCoordinator
            +-- MCPRuntime
            +-- ApprovalManager
            +-- ToolGateway
            |
            +-- build_hub(...)

    Tests may construct the same graph with isolated runtime
    dependencies.

    Security policy remains inside trusted ToolGateway/tool
    registry code rather than model configuration.
    """

    # ============================================================
    # HUB MODEL
    # ============================================================

    if not model_manager.exists(
        settings.hub_model_key
    ):
        raise RuntimeError(
            "HUB_MODEL_KEY references "
            "an unknown or disabled "
            "model profile: "
            f"{settings.hub_model_key}"
        )

    # ============================================================
    # AGENT DEFINITIONS
    # ============================================================

    agents_dir = (
        settings.require_path(
            settings.agents_dir,
            "AGENTS_DIR",
        )
    )

    configured_agents = (
        load_agent_directory(
            agents_dir
        )
    )

    enabled_agents = []

    for agent in (
        configured_agents
    ):
        profile = (
            settings.model_profile(
                agent.model
            )
        )

        if profile is None:
            raise RuntimeError(
                f"Agent '{agent.name}' "
                "references model profile "
                f"'{agent.model}', but that "
                "profile is not configured."
            )

        if not profile.enabled:
            print(
                "[HUB] Specialist disabled "
                f"agent='{agent.name}' "
                f"model='{agent.model}'"
            )

            continue

        if not model_manager.exists(
            agent.model
        ):
            raise RuntimeError(
                f"Agent '{agent.name}' "
                "references unavailable "
                f"model '{agent.model}'."
            )

        enabled_agents.append(
            agent
        )

    # ============================================================
    # AGENT REGISTRY
    # ============================================================

    agent_registry = (
        AgentRegistry()
    )

    agent_registry.register_many(
        enabled_agents
    )

    # ============================================================
    # SPECIALIST RUNTIME
    # ============================================================

    runtime = (
        AgentRuntime(
            agent_registry=(
                agent_registry
            ),
            inference=(
                inference
            ),
            tool_gateway=(
                tool_gateway
            ),
        )
    )

    # ============================================================
    # HUB ROUTER
    # ============================================================

    router = (
        LLMRouter(
            registry=(
                agent_registry
            ),
            inference=(
                inference
            ),
            model_key=(
                settings
                .hub_model_key
            ),
        )
    )

    # ============================================================
    # PRIMARY ASSISTANT
    # ============================================================

    primary_assistant = (
        PrimaryAssistant(
            inference=(
                inference
            ),
            model_key=(
                settings
                .hub_model_key
            ),
        )
    )

    return Orchestrator(
        router=(
            router
        ),
        runtime=(
            runtime
        ),
        primary_assistant=(
            primary_assistant
        ),
    )