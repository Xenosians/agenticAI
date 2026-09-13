import asyncio

from subagents.core.loader import (
    load_agent_directory,
)

from subagents.core.registry import (
    AgentRegistry,
)

from subagents.core.runtime import (
    AgentRuntime,
)

from subagents.core.types import (
    AgentTask,
)


class FakeInference:
    def __init__(
        self,
    ) -> None:
        self.calls = []

        self.response = """
        [
            {
                "name": "account_status",
                "arguments": {
                    "user_id": "jdoe"
                }
            }
        ]
        """

        self.error = None

    async def generate(
        self,
        *,
        model_key,
        messages,
        max_new_tokens,
        priority,
    ):
        self.calls.append(
            {
                "model_key":
                    model_key,

                "messages":
                    messages,

                "max_new_tokens":
                    max_new_tokens,

                "priority":
                    priority,
            }
        )

        if self.error is not None:
            raise self.error

        return self.response


class FakeToolGateway:
    def __init__(
        self,
    ) -> None:
        self.calls = []

        self.result = {
            "ok": True,

            "status":
                "success",

            "result": {
                "ok": True,
                "user_id": "jdoe",
                "enabled": True,
                "locked": False,
            },
        }

    async def execute(
        self,
        agent,
        user_input,
        tool_name,
        arguments,
    ):
        self.calls.append(
            {
                "agent":
                    agent.name,

                "user_input":
                    user_input,

                "tool_name":
                    tool_name,

                "arguments":
                    arguments,
            }
        )

        return self.result


def build_runtime():
    agent_registry = (
        AgentRegistry()
    )

    agents = (
        load_agent_directory(
            "subagents/agents"
        )
    )

    agent_registry.register_many(
        agents
    )

    inference = (
        FakeInference()
    )

    gateway = (
        FakeToolGateway()
    )

    runtime = AgentRuntime(
        agent_registry=(
            agent_registry
        ),
        inference=(
            inference
        ),
        tool_gateway=(
            gateway
        ),
    )

    return (
        runtime,
        inference,
        gateway,
    )


def test_runtime_executes_account_specialist():
    (
        runtime,
        inference,
        gateway,
    ) = build_runtime()

    task = AgentTask(
        task_id="task-001",
        agent_name=(
            "account-specialist"
        ),
        user_request=(
            "Is jdoe locked?"
        ),
    )

    result = asyncio.run(
        runtime.run(
            task
        )
    )

    assert (
        result.task_id
        == "task-001"
    )

    assert (
        result.agent_name
        == "account-specialist"
    )

    assert (
        result.status
        == "success"
    ), result.error

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

    assert gateway.calls == [
        {
            "agent":
                "account-specialist",

            "user_input":
                "Is jdoe locked?",

            "tool_name":
                "account_status",

            "arguments": {
                "user_id":
                    "jdoe"
            },
        }
    ]

    assert (
        len(
            inference.calls
        )
        == 1
    )

    inference_call = (
        inference.calls[0]
    )

    assert (
        inference_call[
            "model_key"
        ]
        == "qwen2.5-0.5b-funccall"
    )

    messages = (
        inference_call[
            "messages"
        ]
    )

    assert (
        messages[0][
            "role"
        ]
        == "system"
    )

    assert (
        "ITSM account specialist"
        in messages[0][
            "content"
        ]
    )

    assert messages[1] == {
        "role":
            "user",

        "content":
            "Is jdoe locked?",
    }


def test_runtime_rejects_unknown_agent():
    (
        runtime,
        inference,
        gateway,
    ) = build_runtime()

    task = AgentTask(
        task_id="task-002",
        agent_name=(
            "does-not-exist"
        ),
        user_request="Hello",
    )

    result = asyncio.run(
        runtime.run(
            task
        )
    )

    assert (
        result.status
        == "error"
    )

    assert (
        result.answer
        is None
    )

    assert (
        result.error
        is not None
    )

    assert (
        inference.calls
        == []
    )

    assert (
        gateway.calls
        == []
    )


def test_runtime_returns_model_failure():
    (
        runtime,
        inference,
        gateway,
    ) = build_runtime()

    inference.error = KeyError(
        "Model is unavailable."
    )

    task = AgentTask(
        task_id="task-003",
        agent_name=(
            "account-specialist"
        ),
        user_request=(
            "Is jdoe locked?"
        ),
    )

    result = asyncio.run(
        runtime.run(
            task
        )
    )

    assert (
        result.status
        == "error"
    )

    assert (
        result.error
        is not None
    )

    assert (
        "Worker model failed"
        in result.error
    )

    assert (
        gateway.calls
        == []
    )


def test_runtime_passes_additional_instructions():
    (
        runtime,
        inference,
        _gateway,
    ) = build_runtime()

    task = AgentTask(
        task_id="task-004",
        agent_name=(
            "account-specialist"
        ),
        user_request=(
            "Check jdoe."
        ),
        instructions=(
            "Preserve the identifier exactly."
        ),
    )

    result = asyncio.run(
        runtime.run(
            task
        )
    )

    assert (
        result.status
        == "success"
    ), result.error

    messages = (
        inference.calls[0][
            "messages"
        ]
    )

    assert (
        messages[2][
            "role"
        ]
        == "user"
    )

    assert (
        "Preserve the identifier exactly."
        in messages[2][
            "content"
        ]
    )


def test_runtime_rejects_invalid_worker_json():
    (
        runtime,
        inference,
        gateway,
    ) = build_runtime()

    inference.response = (
        "I think you should call "
        "account_status."
    )

    task = AgentTask(
        task_id="task-005",
        agent_name=(
            "account-specialist"
        ),
        user_request=(
            "Is jdoe locked?"
        ),
    )

    result = asyncio.run(
        runtime.run(
            task
        )
    )

    assert (
        result.status
        == "error"
    )

    assert (
        result.error
        is not None
    )

    assert (
        gateway.calls
        == []
    )


def test_runtime_rejects_multiple_tool_calls():
    (
        runtime,
        inference,
        gateway,
    ) = build_runtime()

    inference.response = """
    [
        {
            "name": "account_status",
            "arguments": {
                "user_id": "jdoe"
            }
        },
        {
            "name": "unlock_user",
            "arguments": {
                "user_id": "jdoe"
            }
        }
    ]
    """

    task = AgentTask(
        task_id="task-006",
        agent_name=(
            "account-specialist"
        ),
        user_request=(
            "Check and unlock jdoe."
        ),
    )

    result = asyncio.run(
        runtime.run(
            task
        )
    )

    assert (
        result.status
        == "error"
    )

    assert (
        "exactly one tool call"
        in result.error
    )

    assert (
        gateway.calls
        == []
    )


def test_runtime_returns_approval_required():
    (
        runtime,
        _inference,
        gateway,
    ) = build_runtime()

    gateway.result = {
        "ok": True,

        "status":
            "approval_required",

        "tool":
            "account_status",

        "approval_id":
            "approval-123",
    }

    task = AgentTask(
        task_id="task-007",
        agent_name=(
            "account-specialist"
        ),
        user_request=(
            "Is jdoe locked?"
        ),
    )

    result = asyncio.run(
        runtime.run(
            task
        )
    )

    assert (
        result.status
        == "approval_required"
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
        result.approval_id
        == "approval-123"
    )

    assert (
        "approval-123"
        in result.answer
    )


def test_runtime_returns_gateway_error():
    (
        runtime,
        _inference,
        gateway,
    ) = build_runtime()

    gateway.result = {
        "ok": False,
        "status": "denied",
        "error": (
            "Execution denied."
        ),
    }

    task = AgentTask(
        task_id="task-008",
        agent_name=(
            "account-specialist"
        ),
        user_request=(
            "Is jdoe locked?"
        ),
    )

    result = asyncio.run(
        runtime.run(
            task
        )
    )

    assert (
        result.status
        == "error"
    )

    assert (
        result.proposed_tool
        == "account_status"
    )

    assert (
        result.error
        == "Execution denied."
    )