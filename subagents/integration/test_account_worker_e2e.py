import asyncio
from pathlib import Path

from agent.mcp_client import mcp_runtime

from subagents.core.loader import load_agent_directory
from subagents.core.registry import AgentRegistry
from subagents.core.runtime import AgentRuntime
from subagents.core.tool_gateway import ToolGateway
from subagents.core.types import AgentTask
from subagents.llm.qwen_funcall import QwenFuncCallBackend
from subagents.llm.registry import ModelRegistry


MODEL_PATH = Path(
    "/mnt/c/project/agenticaiPersonal/Models/"
    "qwen2.5-0.5b-funccall"
)

MODEL_NAME = "qwen2.5-0.5b-funccall"


def fail_if_approval_requested(
    tool_name,
    arguments,
):
    """
    This integration test is intentionally read-only.

    If the worker unexpectedly proposes a mutating operation,
    fail instead of creating an approval.
    """

    raise AssertionError(
        f"Read-only test unexpectedly requested approval "
        f"for tool '{tool_name}' with arguments {arguments}"
    )


def build_runtime() -> AgentRuntime:
    # -----------------------------------------
    # Agent definitions
    # -----------------------------------------

    agent_registry = AgentRegistry()

    agents = load_agent_directory(
        "subagents/agents"
    )

    agent_registry.register_many(agents)

    # -----------------------------------------
    # Real worker model
    # -----------------------------------------

    model_registry = ModelRegistry()

    backend = QwenFuncCallBackend(
        MODEL_PATH
    )

    model_registry.register(
        MODEL_NAME,
        backend,
    )

    # -----------------------------------------
    # Real ToolGateway + real MCP
    # -----------------------------------------

    tool_gateway = ToolGateway(
        approval_creator=fail_if_approval_requested,
        mcp=mcp_runtime,
    )

    return AgentRuntime(
        agent_registry=agent_registry,
        model_registry=model_registry,
        tool_gateway=tool_gateway,
    )


async def run_e2e_test():
    await mcp_runtime.start()

    try:
        runtime = build_runtime()

        task = AgentTask(
            task_id="e2e-account-001",
            agent_name="account-specialist",
            user_request="Is jdoe locked?",
        )

        result = await runtime.run(task)

        print()
        print("===== E2E RESULT =====")
        print(f"status: {result.status}")
        print(f"agent: {result.agent_name}")
        print(f"tool: {result.proposed_tool}")
        print(
            f"arguments: "
            f"{result.proposed_arguments}"
        )
        print(f"answer: {result.answer}")
        print(f"error: {result.error}")
        print("======================")

        assert result.status == "success", (
            result.error
        )

        assert (
            result.agent_name
            == "account-specialist"
        )

        assert (
            result.proposed_tool
            == "account_status"
        )

        assert result.proposed_arguments == {
            "user_id": "jdoe"
        }

        assert result.answer is not None

        assert "jdoe" in result.answer

    finally:
        await mcp_runtime.stop()


def test_real_account_worker_e2e():
    asyncio.run(
        run_e2e_test()
    )