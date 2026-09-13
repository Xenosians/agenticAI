from config.settings import (
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

from subagents.llm.factory import (
    build_hub_backend,
    build_worker_backend,
)

from subagents.llm.registry import (
    ModelRegistry,
)


def build_hub() -> Orchestrator:
    """
    Build the complete Agentic Developer Hub runtime.

    Agent definitions determine which logical worker models are
    required.

    WORKER_MODELS provides deployment configuration for those
    logical model names.

    No specialist type is wired explicitly here.
    """

    settings = get_settings()

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

    # ============================================================
    # Select enabled agents
    #
    # An agent becomes available when:
    #
    # 1. its definition exists in AGENTS_DIR
    # 2. its logical model exists in WORKER_MODELS
    # 3. that worker model is enabled
    #
    # This removes specialist-specific enable flags and wiring.
    # ============================================================

    enabled_agents = []

    for agent in configured_agents:
        model_config = (
            settings.worker_model(
                agent.model
            )
        )

        if model_config is None:
            raise RuntimeError(
                f"Agent '{agent.name}' "
                "references worker model "
                f"'{agent.model}', but that "
                "model is not configured in "
                "WORKER_MODELS."
            )

        if not model_config.enabled:
            print(
                "[HUB] Specialist disabled "
                f"agent='{agent.name}' "
                f"model='{agent.model}'"
            )

            continue

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
    # Worker model registry
    #
    # Register each logical model exactly once, regardless of
    # how many agents share it.
    # ============================================================

    model_registry = (
        ModelRegistry()
    )

    required_model_keys = sorted(
        {
            agent.model
            for agent
            in enabled_agents
        }
    )

    for model_key in (
        required_model_keys
    ):
        model_config = (
            settings.require_worker_model(
                model_key
            )
        )

        backend_type = (
            model_config.backend
        )

        model_path = (
            model_config.model_path
        )

        def load_worker_backend(
            backend_type=backend_type,
            model_path=model_path,
        ):
            return build_worker_backend(
                backend_type=(
                    backend_type
                ),
                model_path=(
                    model_path
                ),
            )

        model_registry.register_lazy(
            model_key,
            load_worker_backend,
        )

    # ============================================================
    # Deterministic security / execution boundary
    # ============================================================

    tool_gateway = (
        ToolGateway()
    )

    # ============================================================
    # Specialist runtime
    # ============================================================

    runtime = AgentRuntime(
        agent_registry=(
            agent_registry
        ),
        model_registry=(
            model_registry
        ),
        tool_gateway=(
            tool_gateway
        ),
    )

    # ============================================================
    # Hub model
    #
    # The Hub remains separate because it has Hub-specific model
    # options and also powers the primary conversational path.
    # ============================================================

    hub_model_path = (
        settings.require_path(
            settings.hub_model_path,
            "HUB_MODEL_PATH",
        )
    )

    hub_backend = (
        build_hub_backend(
            backend_type=(
                settings.hub_backend
            ),
            model_path=(
                hub_model_path
            ),
            dequantize_fp8=(
                settings
                .hub_dequantize_fp8
            ),
            offload_folder=(
                settings
                .hub_offload_folder
            ),
        )
    )

    # ============================================================
    # Specialist router
    # ============================================================

    router = LLMRouter(
        registry=(
            agent_registry
        ),
        backend=(
            hub_backend
        ),
    )

    # ============================================================
    # Primary conversational assistant
    #
    # Shares the Hub backend.
    # ============================================================

    primary_assistant = (
        PrimaryAssistant(
            backend=(
                hub_backend
            ),
        )
    )

    # ============================================================
    # Complete orchestrator
    # ============================================================

    return Orchestrator(
        router=router,
        runtime=runtime,
        primary_assistant=(
            primary_assistant
        ),
    )