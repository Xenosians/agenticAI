from config.settings import get_settings

from subagents.core.loader import (
    load_agent_directory,
)
from subagents.core.registry import AgentRegistry
from subagents.core.llm_router import LLMRouter
from subagents.core.orchestrator import Orchestrator
from subagents.core.runtime import AgentRuntime
from subagents.core.tool_gateway import ToolGateway

from subagents.llm.factory import (
    build_hub_backend,
    build_worker_backend,
)
from subagents.llm.registry import ModelRegistry


def build_hub() -> Orchestrator:
    settings = get_settings()

    # ---------------------------------------------------------
    # Agents
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Worker models
    # ---------------------------------------------------------

    model_registry = ModelRegistry()

    account_model_path = settings.require_path(
        settings.account_model_path,
        "ACCOUNT_MODEL_PATH",
    )

    account_backend = build_worker_backend(
        backend_type=settings.account_backend,
        model_path=account_model_path,
    )

    model_registry.register(
        settings.account_model_key,
        account_backend,
    )

    # Access is intentionally optional until its
    # Qwen worker checkpoint is validated.
    if settings.access_enabled:
        access_model_path = settings.require_path(
            settings.access_model_path,
            "ACCESS_MODEL_PATH",
        )

        access_backend = build_worker_backend(
            backend_type=settings.access_backend,
            model_path=access_model_path,
        )

        model_registry.register(
            settings.access_model_key,
            access_backend,
        )

    # ---------------------------------------------------------
    # Security / tool boundary
    # ---------------------------------------------------------

    tool_gateway = ToolGateway()

    # ---------------------------------------------------------
    # Workers
    # ---------------------------------------------------------

    runtime = AgentRuntime(
        agent_registry=agent_registry,
        model_registry=model_registry,
        tool_gateway=tool_gateway,
    )

    # ---------------------------------------------------------
    # Hub
    # ---------------------------------------------------------

    hub_model_path = settings.require_path(
        settings.hub_model_path,
        "HUB_MODEL_PATH",
    )

    hub_backend = build_hub_backend(
        backend_type=settings.hub_backend,
        model_path=hub_model_path,
        dequantize_fp8=(
            settings.hub_dequantize_fp8
        ),
    )

    router = LLMRouter(
        registry=agent_registry,
        backend=hub_backend,
    )

    return Orchestrator(
        router=router,
        runtime=runtime,
    )