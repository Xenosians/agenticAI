from config import (
    Settings,
)

from subagents.core.definitions.loader import (
    load_agent_directory,
)

from subagents.core.definitions.registry import (
    AgentRegistry,
)

from subagents.core.orchestration.router import (
    LLMRouter,
)

from subagents.core.orchestration.orchestrator import (
    Orchestrator,
)

from subagents.core.orchestration.primary_assistant import (
    PrimaryAssistant,
)

from subagents.core.orchestration.runtime import (
    AgentRuntime,
)

from subagents.core.orchestration.semantic_guard import (
    SemanticGuard,
)

from subagents.core.tooling.gateway import (
    ToolGateway,
)

from subagents.llm.runtime.inference import (
    InferenceEngine,
)

from subagents.llm.runtime.model_manager import (
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

    Runtime ownership belongs to the composition root.

    Security boundaries:

        Hub model
            interprets user intent

        strict routing contract
            rejects malformed semantic execution plans

        SemanticGuard
            deterministically checks specialist behavior against
            structured intent + trusted capability metadata

        ToolGateway
            remains authoritative for authorization, grounding,
            approval, policy, and execution
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
    # SEMANTIC GUARD
    # ============================================================

    semantic_guard = (
        SemanticGuard()
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

            model_profile_resolver=(
                model_manager.model_profile
            ),

            semantic_guard=(
                semantic_guard
            ),
        )
    )

    # ============================================================
    # HUB ROUTER
    #
    # Production routing is strict:
    #
    # malformed semantic output cannot silently become ordinary
    # PrimaryAssistant conversation.
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

            strict_contract=True,
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

    return (
        Orchestrator(
            router=(
                router
            ),

            runtime=(
                runtime
            ),

            primary_assistant=(
                primary_assistant
            ),

            require_semantic_intent=True,
        )
    )
