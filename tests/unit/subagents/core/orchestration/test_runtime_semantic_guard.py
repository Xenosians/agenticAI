import asyncio

from subagents.core.definitions.registry import (
    AgentRegistry,
)

from subagents.core.definitions.types import (
    AgentDefinition,
    AgentTask,
    SemanticIntent,
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

        self.calls = []

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

        return (
            self.response
        )


class FakeGateway:
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
                    arguments.get(
                        "user_id"
                    ),

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

    registry.register(
        AgentDefinition(
            name=(
                "account-specialist"
            ),

            description=(
                "Handles user account operations."
            ),

            model=(
                "account-test-model"
            ),

            tools=[
                "account_status",
                "unlock_user",
                "reset_password",
            ],

            system_prompt=(
                "You are an ITSM account specialist."
            ),
        )
    )

    inference = (
        FakeInference(
            response
        )
    )

    gateway = (
        FakeGateway()
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
        inference,
        gateway,
    )


def intent_for_bob_status(
) -> SemanticIntent:

    return (
        SemanticIntent(
            summary=(
                "Check bob's account status "
                "without resetting it."
            ),

            effect=(
                "read"
            ),

            allowed_tools=[
                "account_status",
            ],

            forbidden_tools=[
                "reset_password",
            ],

            allowed_arguments={
                "user_id": [
                    "bob",
                ],
            },

            forbidden_arguments={
                "user_id": [
                    "alice",
                ],
            },

            max_tool_calls=1,

            clarification_required=False,
        )
    )


def test_runtime_blocks_wrong_tool_before_gateway(
):

    (
        runtime,
        _,
        gateway,
    ) = (
        build_runtime(
            """
            [
              {
                "name": "reset_password",
                "arguments": {
                  "user_id": "bob"
                }
              }
            ]
            """
        )
    )

    result = (
        asyncio.run(
            runtime.run(
                AgentTask(
                    task_id=(
                        "task-wrong-tool"
                    ),

                    agent_name=(
                        "account-specialist"
                    ),

                    user_request=(
                        "Do not reset bob's password. "
                        "Check bob's account status instead."
                    ),

                    instructions=(
                        "Check bob's account status."
                    ),

                    semantic_intent=(
                        intent_for_bob_status()
                    ),
                )
            )
        )
    )

    assert (
        result.status
        == "error"
    )

    assert (
        result.outcome_code
        == "semantic_tool_forbidden"
    )

    assert (
        result.proposed_tool
        == "reset_password"
    )

    assert (
        gateway.calls
        == []
    )


def test_runtime_blocks_wrong_identity_before_gateway(
):

    (
        runtime,
        _,
        gateway,
    ) = (
        build_runtime(
            """
            [
              {
                "name": "account_status",
                "arguments": {
                  "user_id": "alice"
                }
              }
            ]
            """
        )
    )

    result = (
        asyncio.run(
            runtime.run(
                AgentTask(
                    task_id=(
                        "task-wrong-target"
                    ),

                    agent_name=(
                        "account-specialist"
                    ),

                    user_request=(
                        "Alice reported the issue, but "
                        "check bob's account status only."
                    ),

                    instructions=(
                        "Check bob's account status."
                    ),

                    semantic_intent=(
                        intent_for_bob_status()
                    ),
                )
            )
        )
    )

    assert (
        result.status
        == "error"
    )

    assert (
        result.outcome_code
        == "semantic_argument_forbidden"
    )

    assert (
        result.proposed_arguments
        == {
            "user_id":
                "alice",
        }
    )

    assert (
        gateway.calls
        == []
    )


def test_runtime_allows_semantically_matching_call_to_gateway(
):

    (
        runtime,
        _,
        gateway,
    ) = (
        build_runtime(
            """
            [
              {
                "name": "account_status",
                "arguments": {
                  "user_id": "bob"
                }
              }
            ]
            """
        )
    )

    result = (
        asyncio.run(
            runtime.run(
                AgentTask(
                    task_id=(
                        "task-correct"
                    ),

                    agent_name=(
                        "account-specialist"
                    ),

                    user_request=(
                        "Check bob's account status."
                    ),

                    instructions=(
                        "Check bob's account status."
                    ),

                    semantic_intent=(
                        intent_for_bob_status()
                    ),
                )
            )
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

    assert gateway.calls == [
        {
            "agent":
                "account-specialist",

            "user_input":
                "Check bob's account status.",

            "tool_name":
                "account_status",

            "arguments": {
                "user_id":
                    "bob",
            },
        }
    ]
