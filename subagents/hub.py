from config import (
    get_settings,
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
    InferenceCoordinator,
)

from subagents.llm.model_manager import (
    ModelManager,
)

from subagents.llm.scheduler import (
    GpuScheduler,
)


def build_hub() -> Orchestrator:
    """
    Transitional AI composition point.

    Model ownership is now centralized behind:

        ModelManager
            ↓
        GpuScheduler
            ↓
        InferenceCoordinator

    Router, PrimaryAssistant, and AgentRuntime consume only the
    InferenceCoordinator boundary.

    FastAPI lifespan will become the final application
    composition root during the next cleanup stage.
    """

    settings = (
        get_settings()
    )

    # ============================================================
    # Model runtime
    # ============================================================

    model_manager = (
        ModelManager(
            settings=settings,
        )
    )

    scheduler = (
        GpuScheduler()
    )

    inference = (
        InferenceCoordinator(
            model_manager=(
                model_manager
            ),
            scheduler=(
                scheduler
            ),
        )
    )

    # ============================================================
    # Hub model
    #
    # Preserve current readiness behavior:
    # the Hub is loaded while build_hub() executes.
    #
    # Specialist models remain lazy.
    # ============================================================

    if not model_manager.exists(
        settings.hub_model_key
    ):
        raise RuntimeError(
            "HUB_MODEL_KEY references "
            "an unknown or disabled model "
            f"profile: {settings.hub_model_key}"
        )

    model_manager.load(
        settings.hub_model_key
    )

    # ============================================================
    # Load agent definitions
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

    for agent in configured_agents:
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
                "references unavailable model "
                f"'{agent.model}'."
            )

        enabled_agents.append(
            agent
        )

    # ============================================================
    # Agent registry
    # ============================================================

    agent_registry = (
        AgentRegistry()
    )

    agent_registry.register_many(
        enabled_agents
    )

    # ============================================================
    # Deterministic execution boundary
    # ============================================================

    tool_gateway = (
        ToolGateway()
    )

    # ============================================================
    # Specialist runtime
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
    # Hub router
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
    # Primary assistant
    #
    # Uses the same logical Hub model through the shared
    # inference boundary rather than owning the backend object.
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
        router=router,
        runtime=runtime,
        primary_assistant=(
            primary_assistant
        ),
    )