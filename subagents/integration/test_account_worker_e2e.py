import asyncio

from pathlib import Path

from agent.mcp_client import (
    MCPRuntime,
)

from config import (
    Settings,
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

from subagents.llm.inference import (
    InferenceCoordinator,
)

from subagents.llm.model_manager import (
    ModelManager,
)

from subagents.llm.scheduler import (
    GpuScheduler,
)


def fail_if_approval_requested(
    tool_name,
    arguments,
    risk=None,
):
    """
    This integration test is intentionally read-only.

    A mutation proposal is a test failure.
    """

    raise AssertionError(
        "Read-only test unexpectedly "
        "requested approval for "
        f"tool '{tool_name}' "
        f"with arguments {arguments} "
        f"and risk {risk}"
    )


def build_runtime(
    *,
    settings: Settings,
    inference: InferenceCoordinator,
    mcp: MCPRuntime,
) -> AgentRuntime:
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

    account_agent = next(
        (
            agent

            for agent
            in agents

            if (
                agent.name
                == "account-specialist"
            )
        ),
        None,
    )

    if account_agent is None:
        raise RuntimeError(
            "account-specialist "
            "definition was not found."
        )

    if not (
        inference
        .model_manager
        .exists(
            account_agent.model
        )
    ):
        raise RuntimeError(
            "Account specialist model "
            f"'{account_agent.model}' "
            "is not registered."
        )

    tool_gateway = (
        ToolGateway(
            approval_creator=(
                fail_if_approval_requested
            ),
            mcp=(
                mcp
            ),
        )
    )

    return AgentRuntime(
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


async def run_e2e_test():
    settings = (
        Settings()
    )

    model_manager = (
        ModelManager(
            settings=(
                settings
            )
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

    mcp = (
        MCPRuntime()
    )

    await mcp.start()

    try:
        runtime = (
            build_runtime(
                settings=(
                    settings
                ),
                inference=(
                    inference
                ),
                mcp=(
                    mcp
                ),
            )
        )

        task = (
            AgentTask(
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
        await mcp.stop()


def test_real_account_worker_e2e():
    asyncio.run(
        run_e2e_test()
    )