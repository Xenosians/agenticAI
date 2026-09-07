from config.settings import get_settings

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

    The Hub model serves two roles:

    1. Decide whether specialist delegation is useful.
    2. Act as the primary conversational assistant when no
       specialist is required.

    Both roles share the same loaded model backend and weights.

    Specialists remain responsible for domain-specific reasoning
    and governed tool proposals.

    Deterministic Python policy remains responsible for
    validating whether actions are allowed.
    """

    settings = get_settings()

    # ============================================================
    # Agent registry
    # ============================================================

    agent_registry = AgentRegistry()

    agents_dir = settings.require_path(
        settings.agents_dir,
        "AGENTS_DIR",
    )

    agents = load_agent_directory(
        agents_dir
    )

    agent_registry.register_many(
        agents
    )

    # ============================================================
    # Model registry
    # ============================================================

    model_registry = ModelRegistry()

    # ============================================================
    # Account worker
    # ============================================================

    account_model_path = (
        settings.require_path(
            settings.account_model_path,
            "ACCOUNT_MODEL_PATH",
        )
    )

    account_backend = (
        build_worker_backend(
            backend_type=(
                settings.account_backend
            ),
            model_path=(
                account_model_path
            ),
        )
    )

    model_registry.register(
        settings.account_model_key,
        account_backend,
    )

    # ============================================================
    # Access worker
    # ============================================================

    if settings.access_enabled:
        access_model_path = (
            settings.require_path(
                settings.access_model_path,
                "ACCESS_MODEL_PATH",
            )
        )

        access_backend = (
            build_worker_backend(
                backend_type=(
                    settings.access_backend
                ),
                model_path=(
                    access_model_path
                ),
            )
        )

        model_registry.register(
            settings.access_model_key,
            access_backend,
        )

    # ============================================================
    # Deterministic security / execution boundary
    # ============================================================

    tool_gateway = ToolGateway()

    # ============================================================
    # Specialist runtime
    # ============================================================

    runtime = AgentRuntime(
        agent_registry=agent_registry,
        model_registry=model_registry,
        tool_gateway=tool_gateway,
    )

    # ============================================================
    # Hub model
    # ============================================================

    hub_model_path = (
        settings.require_path(
            settings.hub_model_path,
            "HUB_MODEL_PATH",
        )
    )

    hub_backend = build_hub_backend(
        backend_type=(
            settings.hub_backend
        ),
        model_path=(
            hub_model_path
        ),
        dequantize_fp8=(
            settings.hub_dequantize_fp8
        ),
        offload_folder=(
            settings.hub_offload_folder
        ),
    )

    # ============================================================
    # Specialist router
    # ============================================================

    router = LLMRouter(
        registry=agent_registry,
        backend=hub_backend,
    )

    # ============================================================
    # Primary conversational assistant
    #
    # IMPORTANT:
    # This deliberately shares hub_backend with the router.
    # No second copy of the Hub model is loaded.
    # ============================================================

    primary_assistant = PrimaryAssistant(
        backend=hub_backend,
    )

    # ============================================================
    # Complete orchestrator
    # ============================================================

    return Orchestrator(
        router=router,
        runtime=runtime,
        primary_assistant=primary_assistant,
    )