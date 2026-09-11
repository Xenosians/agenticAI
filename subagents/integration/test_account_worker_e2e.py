import asyncio

from config import (
    get_settings,
)

from agent.mcp_client import (
    mcp_runtime,
)

from subagents.core.loader import (
    load_agent_directory,
)

from subagents.core.registry import (
    AgentRegistry,
)

from subagents.core.runtime import (
    AgentRuntime,
)

from subagents.core.tool_gateway import (
    ToolGateway,
)

from subagents.core.types import (
    AgentTask,
)

from subagents.llm.qwen_funcall import (
    QwenFuncCallBackend,
)

from subagents.llm.registry import (
    ModelRegistry,
)


def fail_if_approval_requested(
    tool_name,
    arguments,
    risk=None,
):
    """
    This integration test is intentionally read-only.

    If the worker unexpectedly proposes a mutating operation,
    fail instead of creating an approval.
    """

    raise AssertionError(
        "Read-only test unexpectedly requested "
        f"approval for tool '{tool_name}' "
        f"with arguments {arguments} "
        f"and risk {risk}"
    )


def build_runtime() -> AgentRuntime:
    settings = get_settings()

    # -----------------------------------------
    # Agent definitions
    # -----------------------------------------

    agent_registry = (
        AgentRegistry()
    )

    agents_dir = (
        settings.require_path(
            settings.agents_dir,
            "AGENTS_DIR",
        )
    )

    agents = (
        load_agent_directory(
            agents_dir
        )
    )

    agent_registry.register_many(
        agents
    )

    # -----------------------------------------
    # Real worker model
    # -----------------------------------------

    model_registry = (
        ModelRegistry()
    )

    model_path = (
        settings.require_path(
            settings.account_model_path,
            "ACCOUNT_MODEL_PATH",
        )
    )

    backend = (
        QwenFuncCallBackend(
            model_path
        )
    )

    model_registry.register(
        settings.account_model_key,
        backend,
    )

    # -----------------------------------------
    # Real ToolGateway + real MCP
    # -----------------------------------------

    tool_gateway = (
        ToolGateway(
            approval_creator=(
                fail_if_approval_requested
            ),
            mcp=mcp_runtime,
        )
    )

    return AgentRuntime(
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


async def run_e2e_test():
    await mcp_runtime.start()

    try:
        runtime = (
            build_runtime()
        )

        task = AgentTask(
            task_id=(
                "e2e-account-001"
            ),
            agent_name=(
                "account-specialist"
            ),
            user_request=(
                "Is jdoe locked?"
            ),
        )

        result = (
            await runtime.run(
                task
            )
        )

        print()
        print(
            "===== E2E RESULT ====="
        )

        print(
            f"status: {result.status}"
        )

        print(
            f"agent: {result.agent_name}"
        )

        print(
            f"tool: {result.proposed_tool}"
        )

        print(
            "arguments: "
            f"{result.proposed_arguments}"
        )

        print(
            f"answer: {result.answer}"
        )

        print(
            f"error: {result.error}"
        )

        print(
            "======================"
        )

        assert (
            result.status
            == "success"
        ), result.error

        assert (
            result.agent_name
            == "account-specialist"
        )

        assert (
            result.proposed_tool
            == "account_status"
        )

        assert (
            result.proposed_arguments
            == {
                "user_id":
                    "jdoe"
            }
        )

        assert (
            result.answer
            is not None
        )

        assert (
            "jdoe"
            in result.answer
        )

    finally:
        await mcp_runtime.stop()


def test_real_account_worker_e2e():
    asyncio.run(
        run_e2e_test()
    )