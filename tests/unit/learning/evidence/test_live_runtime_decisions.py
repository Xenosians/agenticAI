import asyncio

from dataclasses import (
    asdict,
)

from learning.evidence.recorder import (
    TrajectoryRecorder,
)

from subagents.core.definitions.registry import (
    AgentRegistry,
)

from subagents.core.definitions.types import (
    AgentDefinition,
    AgentResult,
    SemanticIntent,
    SpecialistRequest,
)

from subagents.core.orchestration.orchestrator import (
    Orchestrator,
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

            "tool":
                tool_name,

            "result": {
                "ok":
                    True,

                "user_id":
                    arguments[
                        "user_id"
                    ],

                "enabled":
                    True,

                "locked":
                    False,
            },
        }


class FakeRouter:
    async def route(
        self,
        user_request: str,
    ):

        return [
            SpecialistRequest(
                agent_name=(
                    "account-specialist"
                ),

                instructions=(
                    "Check bob's account status."
                ),

                semantic_intent=(
                    SemanticIntent(
                        summary=(
                            "Check bob's account status."
                        ),

                        effect="read",

                        allowed_tools=[
                            "account_status",
                        ],

                        forbidden_tools=[],

                        allowed_arguments={
                            "user_id": [
                                "bob",
                            ],
                        },

                        forbidden_arguments={},

                        max_tool_calls=1,

                        clarification_required=False,
                    )
                ),
            )
        ]


class FakePrimaryAssistant:
    async def respond(
        self,
        user_request: str,
    ) -> str:

        return (
            "unused"
        )

    async def synthesize(
        self,
        user_request: str,
        results: list[
            AgentResult
        ],
    ) -> str:

        return (
            "bob is enabled and unlocked."
        )


def build_orchestrator(
    worker_response: str,
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
                "Handles account operations."
            ),

            model=(
                "test-model"
            ),

            tools=[
                "account_status",
                "reset_password",
            ],
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
                FakeInference(
                    worker_response
                )
            ),

            tool_gateway=(
                gateway
            ),
        )
    )

    orchestrator = (
        Orchestrator(
            router=(
                FakeRouter()
            ),

            runtime=(
                runtime
            ),

            primary_assistant=(
                FakePrimaryAssistant()
            ),

            require_semantic_intent=True,
        )
    )

    return (
        orchestrator,
        gateway,
    )


def test_successful_runtime_decisions_enter_trajectory_but_not_public_result(
    tmp_path,
):

    (
        orchestrator,
        gateway,
    ) = (
        build_orchestrator(
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

    hub_result = (
        asyncio.run(
            orchestrator.run(
                "Check bob's account status."
            )
        )
    )

    assert (
        hub_result.status
        == "success"
    )

    assert (
        len(
            gateway.calls
        )
        == 1
    )

    public_result = (
        asdict(
            hub_result.results[
                0
            ]
        )
    )

    # Private learning evidence MUST NOT become part of the public
    # AgentResult dataclass serialization.
    assert (
        "_learning_runtime_decisions"
        not in public_result
    )

    assert (
        "semantic_guard_decision"
        not in public_result
    )

    assert (
        "gateway_decision"
        not in public_result
    )

    recorder = (
        TrajectoryRecorder(
            path=(
                tmp_path
                / "trajectories.jsonl"
            ),

            enabled=True,

            hub_model="hub-main",
        )
    )

    trajectory = (
        recorder.build(
            job_id=(
                "job-live-decision"
            ),

            attempt=1,

            result=(
                hub_result
            ),
        )
    )

    step = (
        trajectory.steps[
            0
        ]
    )

    assert (
        step.semantic_guard_decision
        == {
            "allowed":
                True,

            "decision_code":
                "semantic_guard_allowed",

            "error":
                None,
        }
    )

    assert (
        step.gateway_decision
        == {
            "ok":
                True,

            "status":
                "success",

            "decision_code":
                "success",

            "tool":
                "account_status",
        }
    )

    assert (
        step.tool_result
        == {
            "ok":
                True,

            "user_id":
                "bob",

            "enabled":
                True,

            "locked":
                False,
        }
    )


def test_semantic_denial_is_preserved_without_gateway_decision(
    tmp_path,
):

    (
        orchestrator,
        gateway,
    ) = (
        build_orchestrator(
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

    hub_result = (
        asyncio.run(
            orchestrator.run(
                "Check bob's account status."
            )
        )
    )

    assert (
        hub_result.status
        == "partial_error"
    )

    assert (
        gateway.calls
        == []
    )

    recorder = (
        TrajectoryRecorder(
            path=(
                tmp_path
                / "trajectories.jsonl"
            ),

            enabled=True,

            hub_model="hub-main",
        )
    )

    trajectory = (
        recorder.build(
            job_id=(
                "job-live-denial"
            ),

            attempt=1,

            result=(
                hub_result
            ),
        )
    )

    step = (
        trajectory.steps[
            0
        ]
    )

    assert (
        step.semantic_guard_decision
        is not None
    )

    assert (
        step.semantic_guard_decision[
            "allowed"
        ]
        is False
    )

    assert (
        step.semantic_guard_decision[
            "decision_code"
        ]
        == "semantic_tool_not_allowed"
    )

    assert (
        step.gateway_decision
        is None
    )

    assert (
        step.tool_result
        is None
    )