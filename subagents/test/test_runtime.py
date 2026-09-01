import asyncio

from subagents.core.loader import load_agent_directory
from subagents.core.registry import AgentRegistry
from subagents.core.runtime import AgentRuntime
from subagents.core.types import AgentTask
from subagents.llm.base import LLMBackend
from subagents.llm.registry import ModelRegistry


class FakeLLMBackend(LLMBackend):
    def __init__(self) -> None:
        self.last_messages = None
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

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
    ) -> str:
        self.last_messages = messages
        return self.response


class FakeToolGateway:
    def __init__(self) -> None:
        self.calls = []

        self.result = {
            "ok": True,
            "status": "success",
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
                "agent": agent.name,
                "user_input": user_input,
                "tool_name": tool_name,
                "arguments": arguments,
            }
        )

        return self.result


def build_runtime():
    agent_registry = AgentRegistry()

    agents = load_agent_directory(
        "subagents/agents"
    )

    agent_registry.register_many(agents)

    model_registry = ModelRegistry()

    backend = FakeLLMBackend()

    model_registry.register(
        "qwen2.5-0.5b-funccall",
        backend,
    )

    gateway = FakeToolGateway()

    runtime = AgentRuntime(
        agent_registry=agent_registry,
        model_registry=model_registry,
        tool_gateway=gateway,
    )

    return runtime, backend, gateway


def test_runtime_executes_account_specialist():
    runtime, backend, gateway = build_runtime()

    task = AgentTask(
        task_id="task-001",
        agent_name="account-specialist",
        user_request="Is jdoe locked?",
    )

    result = asyncio.run(
        runtime.run(task)
    )

    assert result.task_id == "task-001"
    assert result.agent_name == "account-specialist"

    assert result.status == "success", result.error

    assert result.proposed_tool == "account_status"

    assert result.proposed_arguments == {
        "user_id": "jdoe"
    }

    assert gateway.calls == [
        {
            "agent": "account-specialist",
            "user_input": "Is jdoe locked?",
            "tool_name": "account_status",
            "arguments": {
                "user_id": "jdoe"
            },
        }
    ]

    assert backend.last_messages[0]["role"] == "system"

    assert "ITSM account specialist" in (
        backend.last_messages[0]["content"]
    )

    assert backend.last_messages[1] == {
        "role": "user",
        "content": "Is jdoe locked?",
    }


def test_runtime_rejects_unknown_agent():
    runtime, _, gateway = build_runtime()

    task = AgentTask(
        task_id="task-002",
        agent_name="does-not-exist",
        user_request="Hello",
    )

    result = asyncio.run(
        runtime.run(task)
    )

    assert result.status == "error"
    assert result.answer is None
    assert result.error is not None

    assert gateway.calls == []


def test_runtime_rejects_missing_model():
    agent_registry = AgentRegistry()

    agents = load_agent_directory(
        "subagents/agents"
    )

    agent_registry.register_many(agents)

    model_registry = ModelRegistry()

    gateway = FakeToolGateway()

    runtime = AgentRuntime(
        agent_registry=agent_registry,
        model_registry=model_registry,
        tool_gateway=gateway,
    )

    task = AgentTask(
        task_id="task-003",
        agent_name="account-specialist",
        user_request="Is jdoe locked?",
    )

    result = asyncio.run(
        runtime.run(task)
    )

    assert result.status == "error"
    assert result.error is not None

    assert gateway.calls == []


def test_runtime_passes_additional_instructions():
    runtime, backend, _ = build_runtime()

    task = AgentTask(
        task_id="task-004",
        agent_name="account-specialist",
        user_request="Check jdoe.",
        instructions="Preserve the identifier exactly.",
    )

    result = asyncio.run(
        runtime.run(task)
    )

    assert result.status == "success", result.error

    assert backend.last_messages[2]["role"] == "user"

    assert (
        "Preserve the identifier exactly."
        in backend.last_messages[2]["content"]
    )


def test_runtime_rejects_invalid_worker_json():
    runtime, backend, gateway = build_runtime()

    backend.response = (
        "I think you should call account_status."
    )

    task = AgentTask(
        task_id="task-005",
        agent_name="account-specialist",
        user_request="Is jdoe locked?",
    )

    result = asyncio.run(
        runtime.run(task)
    )

    assert result.status == "error"
    assert result.error is not None

    assert gateway.calls == []


def test_runtime_rejects_multiple_tool_calls():
    runtime, backend, gateway = build_runtime()

    backend.response = """
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
        agent_name="account-specialist",
        user_request="Check and unlock jdoe.",
    )

    result = asyncio.run(
        runtime.run(task)
    )

    assert result.status == "error"

    assert (
        "exactly one tool call"
        in result.error
    )

    assert gateway.calls == []


def test_runtime_returns_approval_required():
    runtime, _, gateway = build_runtime()

    gateway.result = {
        "ok": True,
        "status": "approval_required",
        "tool": "account_status",
        "approval_id": "approval-123",
    }

    task = AgentTask(
        task_id="task-007",
        agent_name="account-specialist",
        user_request="Is jdoe locked?",
    )

    result = asyncio.run(
        runtime.run(task)
    )

    assert result.status == "approval_required"

    assert result.proposed_tool == "account_status"

    assert result.proposed_arguments == {
        "user_id": "jdoe"
    }

    assert "approval-123" in result.answer


def test_runtime_returns_gateway_error():
    runtime, _, gateway = build_runtime()

    gateway.result = {
        "ok": False,
        "status": "denied",
        "error": "Execution denied.",
    }

    task = AgentTask(
        task_id="task-008",
        agent_name="account-specialist",
        user_request="Is jdoe locked?",
    )

    result = asyncio.run(
        runtime.run(task)
    )

    assert result.status == "error"

    assert result.proposed_tool == "account_status"

    assert result.error == "Execution denied."