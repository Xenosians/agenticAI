import asyncio

from subagents.core.definitions.loader import (
    load_agent_directory,
)

from subagents.core.definitions.registry import (
    AgentRegistry,
)

from subagents.core.definitions.types import (
    AgentTask,
)

from subagents.core.orchestration.runtime import (
    AgentRuntime,
)


class FakeInference:
    def __init__(
        self,
        response: str,
    ) -> None:

        self.response = (
            response
        )

    async def generate(
        self,
        *,
        model_key,
        messages,
        max_new_tokens,
        priority,
    ):

        return (
            self.response
        )


class FakeToolGateway:
    def __init__(
        self,
    ) -> None:

        self.calls = []

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

        return {
            "ok":
                True,

            "status":
                "success",

            "decision_code":
                "success",

            "result": {
                "ok":
                    True,

                "user_id":
                    "jdoe",

                "enabled":
                    True,

                "locked":
                    False,
            },
        }


def build_runtime(
    response: str,
):
    registry = (
        AgentRegistry()
    )

    registry.register_many(
        load_agent_directory(
            "subagents/agents"
        )
    )

    inference = (
        FakeInference(
            response
        )
    )

    gateway = (
        FakeToolGateway()
    )

    runtime = (
        AgentRuntime(
            agent_registry=(
                registry
            ),

            inference=(
                inference
            ),

            tool_gateway=(
                gateway
            ),
        )
    )

    return (
        runtime,
        gateway,
    )


def run_account(
    runtime: AgentRuntime,
):
    return (
        asyncio.run(
            runtime.run(
                AgentTask(
                    task_id=(
                        "task-output-evidence"
                    ),

                    agent_name=(
                        "account-specialist"
                    ),

                    user_request=(
                        "Check jdoe."
                    ),
                )
            )
        )
    )


def test_valid_single_call_uses_canonical_fields_only(
):
    raw = (
        '[{"name":"account_status",'
        '"arguments":{"user_id":"jdoe"}}]'
    )

    (
        runtime,
        gateway,
    ) = build_runtime(
        raw
    )

    result = (
        run_account(
            runtime
        )
    )

    assert (
        result.status
        == "success"
    )

    assert (
        result.outcome_code
        == "success"
    )

    assert (
        result.proposed_tool
        == "account_status"
    )

    assert (
        result.proposed_arguments
        == {
            "user_id":
                "jdoe",
        }
    )

    # Supplemental evidence remains empty when the canonical
    # exactly-one representation is already lossless.
    assert (
        result.raw_model_output
        is None
    )

    assert (
        result.proposed_tool_calls
        is None
    )

    assert (
        len(
            gateway.calls
        )
        == 1
    )


def test_parse_failure_preserves_raw_worker_output(
):
    raw = (
        "I think account_status should be called."
    )

    (
        runtime,
        gateway,
    ) = build_runtime(
        raw
    )

    result = (
        run_account(
            runtime
        )
    )

    assert (
        result.status
        == "error"
    )

    assert (
        result.outcome_code
        == "tool_parse_error"
    )

    assert (
        result.raw_model_output
        == raw
    )

    assert (
        result.proposed_tool_calls
        is None
    )

    assert (
        result.proposed_tool
        is None
    )

    assert (
        result.proposed_arguments
        is None
    )

    assert (
        gateway.calls
        == []
    )


def test_multiple_calls_preserve_complete_validated_call_set(
):
    raw = """
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
                "user_id": "alice"
            }
        }
    ]
    """

    (
        runtime,
        gateway,
    ) = build_runtime(
        raw
    )

    result = (
        run_account(
            runtime
        )
    )

    assert (
        result.status
        == "error"
    )

    assert (
        result.outcome_code
        == "invalid_tool_call_count"
    )

    assert (
        result.raw_model_output
        == raw
    )

    assert (
        result.proposed_tool_calls
        == [
            {
                "name":
                    "account_status",

                "arguments": {
                    "user_id":
                        "jdoe",
                },
            },

            {
                "name":
                    "unlock_user",

                "arguments": {
                    "user_id":
                        "alice",
                },
            },
        ]
    )

    # No single call is promoted into the canonical executable
    # proposal when the worker violated the exactly-one contract.
    assert (
        result.proposed_tool
        is None
    )

    assert (
        result.proposed_arguments
        is None
    )

    # Most important safety assertion: observing the calls does not
    # mean executing either call.
    assert (
        gateway.calls
        == []
    )
