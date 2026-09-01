from pathlib import Path

from subagents.core.loader import load_agent_directory
from subagents.core.orchestrator import Orchestrator
from subagents.core.registry import AgentRegistry
from subagents.core.router import Router
from subagents.core.runtime import AgentRuntime
from subagents.core.tool_gateway import ToolGateway
from subagents.llm.qwen_funcall import QwenFuncCallBackend
from subagents.llm.registry import ModelRegistry
from subagents.core.llm_router import LLMRouter
from subagents.llm.qwen_hub import QwenHubBackend


ACCOUNT_MODEL_NAME = "qwen2.5-0.5b-funccall"

ACCOUNT_MODEL_PATH = Path(
    "/mnt/c/project/agenticaiPersonal/Models/"
    "qwen2.5-0.5b-funccall"
)

HUB_MODEL_PATH = Path(
    "/mnt/c/project/agenticaiPersonal/Models/Qwen3-0.6B"
)


def build_hub() -> Orchestrator:
    """
    Build the current hub-worker runtime.

    V1:
        deterministic router
        +
        real specialist models
        +
        real ToolGateway
    """

    # -----------------------------------------
    # Agent registry
    # -----------------------------------------

    agent_registry = AgentRegistry()

    agents = load_agent_directory(
        "subagents/agents"
    )

    agent_registry.register_many(
        agents
    )

    # -----------------------------------------
    # Model registry
    # -----------------------------------------

    model_registry = ModelRegistry()

    account_backend = QwenFuncCallBackend(
        ACCOUNT_MODEL_PATH
    )

    model_registry.register(
        ACCOUNT_MODEL_NAME,
        account_backend,
    )

    # -----------------------------------------
    # Security / execution layer
    # -----------------------------------------

    tool_gateway = ToolGateway()

    # -----------------------------------------
    # Worker runtime
    # -----------------------------------------

    runtime = AgentRuntime(
        agent_registry=agent_registry,
        model_registry=model_registry,
        tool_gateway=tool_gateway,
    )

    # -----------------------------------------
    # Temporary deterministic routing
    # -----------------------------------------
    
    hub_backend = QwenHubBackend(
        HUB_MODEL_PATH 
    )

    router = LLMRouter(
        registry=agent_registry,
        backend=hub_backend
    )

    # -----------------------------------------
    # Hub / orchestrator
    # -----------------------------------------

    return Orchestrator(
        router=router,
        runtime=runtime,
    )