import asyncio
import json

import pytest

from subagents.core.definitions.registry import (
    AgentRegistry,
)

from subagents.core.definitions.types import (
    AgentDefinition,
    AgentResult,
    ResultCondition,
    SemanticIntent,
    SpecialistRequest,
)

from subagents.core.orchestration.orchestrator import (
    Orchestrator,
)

from subagents.core.orchestration.router import (
    LLMRouter,
    RoutingContractError,
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


def access_registry(
) -> AgentRegistry:

    registry = (
        AgentRegistry()
    )

    registry.register(
        AgentDefinition(
            name=(
                "access-specialist"
            ),

            description=(
                "Handles governed access operations."
            ),

            model=(
                "test-access-model"
            ),

            tools=[
                "check_access",
                "grant_access",
                "revoke_access",
            ],
        )
    )

    return registry


def payload(
    *,
    result_field: str = "has_access",
):

    return {
        "delegations": [
            {
                "agent":
                    "access-specialist",

                "instructions":
                    "Check jdoe VPN access.",

                "intent": {
                    "summary":
                        "Check jdoe VPN access.",

                    "effect":
                        "read",

                    "allowed_tools": [
                        "check_access",
                    ],

                    "forbidden_tools": [
                        "grant_access",
                        "revoke_access",
                    ],

                    "allowed_arguments": {
                        "user_id": [
                            "jdoe",
                        ],

                        "resource": [
                            "VPN",
                        ],
                    },

                    "forbidden_arguments":
                        {},

                    "max_tool_calls":
                        1,

                    "clarification_required":
                        False,
                },
            },

            {
                "agent":
                    "access-specialist",

                "instructions":
                    "Grant jdoe VPN access only if the "
                    "trusted access check says it is absent.",

                "intent": {
                    "summary":
                        "Grant jdoe VPN access if absent.",

                    "effect":
                        "mutation",

                    "allowed_tools": [
                        "grant_access",
                    ],

                    "forbidden_tools": [
                        "revoke_access",
                    ],

                    "allowed_arguments": {
                        "user_id": [
                            "jdoe",
                        ],

                        "resource": [
                            "VPN",
                        ],
                    },

                    "forbidden_arguments":
                        {},

                    "max_tool_calls":
                        1,

                    "clarification_required":
                        False,
                },

                "when": {
                    "source_agent":
                        "access-specialist",

                    "result_field":
                        result_field,

                    "equals":
                        False,
                },
            },
        ]
    }


def strict_router(
    data: dict,
) -> LLMRouter:

    return (
        LLMRouter(
            registry=(
                access_registry()
            ),

            inference=(
                FakeInference(
                    json.dumps(
                        data
                    )
                )
            ),

            model_key=(
                "hub-main"
            ),

            strict_contract=True,
        )
    )


def test_router_accepts_result_aware_same_specialist_workflow(
):

    result = (
        asyncio.run(
            strict_router(
                payload()
            ).route(
                "Check jdoe's VPN access and grant "
                "VPN access if it is missing."
            )
        )
    )

    assert (
        len(
            result
        )
        == 2
    )

    assert (
        result[
            0
        ].condition
        is None
    )

    assert (
        result[
            1
        ].condition
        == ResultCondition(
            source_agent=(
                "access-specialist"
            ),

            result_field=(
                "has_access"
            ),

            equals=False,
        )
    )


def test_router_rejects_untrusted_condition_field(
):

    router = (
        strict_router(
            payload(
                result_field=(
                    "invented_field"
                )
            )
        )
    )

    with pytest.raises(
        RoutingContractError,
        match=(
            "invalid workflow condition"
        ),
    ):

        asyncio.run(
            router.route(
                "Check jdoe's VPN access and grant "
                "VPN access if it is missing."
            )
        )


def read_intent(
) -> SemanticIntent:

    return (
        SemanticIntent(
            summary=(
                "Check jdoe VPN access."
            ),

            effect="read",

            allowed_tools=[
                "check_access",
            ],

            forbidden_tools=[
                "grant_access",
                "revoke_access",
            ],

            allowed_arguments={
                "user_id": [
                    "jdoe",
                ],

                "resource": [
                    "VPN",
                ],
            },

            forbidden_arguments={},

            max_tool_calls=1,

            clarification_required=False,
        )
    )


def mutation_intent(
) -> SemanticIntent:

    return (
        SemanticIntent(
            summary=(
                "Grant jdoe VPN access if absent."
            ),

            effect="mutation",

            allowed_tools=[
                "grant_access",
            ],

            forbidden_tools=[
                "revoke_access",
            ],

            allowed_arguments={
                "user_id": [
                    "jdoe",
                ],

                "resource": [
                    "VPN",
                ],
            },

            forbidden_arguments={},

            max_tool_calls=1,

            clarification_required=False,
        )
    )


class WorkflowRouter:

    async def route(
        self,
        user_request,
    ):

        return [
            SpecialistRequest(
                agent_name=(
                    "access-specialist"
                ),

                instructions=(
                    "Check jdoe VPN access."
                ),

                semantic_intent=(
                    read_intent()
                ),
            ),

            SpecialistRequest(
                agent_name=(
                    "access-specialist"
                ),

                instructions=(
                    "Grant jdoe VPN access if absent."
                ),

                semantic_intent=(
                    mutation_intent()
                ),

                condition=(
                    ResultCondition(
                        source_agent=(
                            "access-specialist"
                        ),

                        result_field=(
                            "has_access"
                        ),

                        equals=False,
                    )
                ),
            ),
        ]


class WorkflowRuntime:

    def __init__(
        self,
        *,
        has_access: bool,
    ) -> None:

        self.has_access = (
            has_access
        )

        self.tasks = []

    async def run(
        self,
        task,
    ) -> AgentResult:

        self.tasks.append(
            task
        )

        if (
            task.semantic_intent.effect
            == "read"
        ):

            return (
                AgentResult(
                    task_id=(
                        task.task_id
                    ),

                    agent_name=(
                        task.agent_name
                    ),

                    status="success",

                    proposed_tool=(
                        "check_access"
                    ),

                    proposed_arguments={
                        "user_id":
                            "jdoe",

                        "resource":
                            "VPN",
                    },

                    outcome_code="success",

                    tool_result={
                        "ok":
                            True,

                        "user_id":
                            "jdoe",

                        "resource":
                            "VPN",

                        "has_access":
                            self.has_access,
                    },

                    answer=(
                        "Access checked."
                    ),
                )
            )

        return (
            AgentResult(
                task_id=(
                    task.task_id
                ),

                agent_name=(
                    task.agent_name
                ),

                status=(
                    "approval_required"
                ),

                proposed_tool=(
                    "grant_access"
                ),

                proposed_arguments={
                    "user_id":
                        "jdoe",

                    "resource":
                        "VPN",
                },

                approval_id=(
                    "approval-test"
                ),

                outcome_code=(
                    "approval_required"
                ),

                answer=(
                    "Granting access requires approval."
                ),
            )
        )


class FakePrimaryAssistant:

    def __init__(
        self,
    ) -> None:

        self.synthesis_calls = []

    async def respond(
        self,
        user_request,
    ):

        return "direct"

    async def synthesize(
        self,
        user_request,
        results,
    ):

        self.synthesis_calls.append(
            (
                user_request,
                results,
            )
        )

        return "synthesized"


def run_workflow(
    *,
    has_access: bool,
):

    runtime = (
        WorkflowRuntime(
            has_access=(
                has_access
            )
        )
    )

    primary = (
        FakePrimaryAssistant()
    )

    orchestrator = (
        Orchestrator(
            router=(
                WorkflowRouter()
            ),

            runtime=(
                runtime
            ),

            primary_assistant=(
                primary
            ),

            require_semantic_intent=True,
        )
    )

    result = (
        asyncio.run(
            orchestrator.run(
                "Check jdoe's VPN access and grant "
                "VPN access if it is missing."
            )
        )
    )

    return (
        result,
        runtime,
        primary,
    )


def test_missing_access_runs_conditional_mutation_to_approval(
):

    (
        result,
        runtime,
        primary,
    ) = (
        run_workflow(
            has_access=False
        )
    )

    assert (
        result.status
        == "approval_required"
    )

    assert (
        len(
            runtime.tasks
        )
        == 2
    )

    assert result.routes == [
        "access-specialist",
        "access-specialist",
    ]

    assert (
        result.results[
            1
        ].proposed_tool
        == "grant_access"
    )

    assert (
        primary.synthesis_calls
        == []
    )


def test_existing_access_skips_mutation_entirely(
):

    (
        result,
        runtime,
        primary,
    ) = (
        run_workflow(
            has_access=True
        )
    )

    assert (
        result.status
        == "success"
    )

    assert (
        len(
            runtime.tasks
        )
        == 1
    )

    assert result.routes == [
        "access-specialist",
    ]

    assert (
        len(
            result.results
        )
        == 1
    )

    assert (
        result.results[
            0
        ].proposed_tool
        == "check_access"
    )

    assert (
        len(
            primary.synthesis_calls
        )
        == 1
    )
