import asyncio

import pytest

from subagents.core.definitions.registry import (
    AgentRegistry,
)

from subagents.core.definitions.types import (
    AgentDefinition,
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


def registry(
) -> AgentRegistry:

    result = (
        AgentRegistry()
    )

    result.register(
        AgentDefinition(
            name=(
                "account-specialist"
            ),

            description=(
                "Handles account operations."
            ),

            model=(
                "account-model"
            ),

            tools=[
                "account_status",
                "unlock_user",
                "reset_password",
            ],
        )
    )

    return result


def strict_router(
    response: str,
) -> LLMRouter:

    return (
        LLMRouter(
            registry=(
                registry()
            ),

            inference=(
                FakeInference(
                    response
                )
            ),

            model_key=(
                "hub-main"
            ),

            strict_contract=True,
        )
    )


def test_strict_router_accepts_valid_empty_delegation(
):

    router = (
        strict_router(
            """
            {
              "delegations": []
            }
            """
        )
    )

    result = (
        asyncio.run(
            router.route(
                "Hello."
            )
        )
    )

    assert (
        result
        == []
    )


def test_strict_router_rejects_invalid_json(
):

    router = (
        strict_router(
            "not-json"
        )
    )

    with pytest.raises(
        RoutingContractError,
        match=(
            "invalid JSON"
        ),
    ):

        asyncio.run(
            router.route(
                "Check jdoe."
            )
        )


def test_strict_router_rejects_missing_semantic_intent(
):

    router = (
        strict_router(
            """
            {
              "delegations": [
                {
                  "agent": "account-specialist",
                  "instructions": "Check jdoe."
                }
              ]
            }
            """
        )
    )

    with pytest.raises(
        RoutingContractError,
        match=(
            "omitted the semantic intent"
        ),
    ):

        asyncio.run(
            router.route(
                "Check jdoe."
            )
        )


def test_strict_router_rejects_unknown_specialist(
):

    router = (
        strict_router(
            """
            {
              "delegations": [
                {
                  "agent": "invented-specialist",
                  "instructions": "Check jdoe.",
                  "intent": {
                    "summary": "Check jdoe.",
                    "effect": "read",
                    "allowed_tools": [],
                    "forbidden_tools": [],
                    "allowed_arguments": {},
                    "forbidden_arguments": {},
                    "max_tool_calls": 1,
                    "clarification_required": true
                  }
                }
              ]
            }
            """
        )
    )

    with pytest.raises(
        RoutingContractError,
        match=(
            "unknown specialist"
        ),
    ):

        asyncio.run(
            router.route(
                "Check jdoe."
            )
        )


def test_strict_router_rejects_duplicate_specialist_delegation(
):

    router = (
        strict_router(
            """
            {
              "delegations": [
                {
                  "agent": "account-specialist",
                  "instructions": "Check jdoe.",
                  "intent": {
                    "summary": "Check jdoe.",
                    "effect": "read",
                    "allowed_tools": [
                      "account_status"
                    ],
                    "forbidden_tools": [],
                    "allowed_arguments": {
                      "user_id": [
                        "jdoe"
                      ]
                    },
                    "forbidden_arguments": {},
                    "max_tool_calls": 1,
                    "clarification_required": false
                  }
                },
                {
                  "agent": "account-specialist",
                  "instructions": "Check alice.",
                  "intent": {
                    "summary": "Check alice.",
                    "effect": "read",
                    "allowed_tools": [
                      "account_status"
                    ],
                    "forbidden_tools": [],
                    "allowed_arguments": {
                      "user_id": [
                        "alice"
                      ]
                    },
                    "forbidden_arguments": {},
                    "max_tool_calls": 1,
                    "clarification_required": false
                  }
                }
              ]
            }
            """
        )
    )

    with pytest.raises(
        RoutingContractError,
        match=(
            "multiple delegations"
        ),
    ):

        asyncio.run(
            router.route(
                "Check jdoe and alice."
            )
        )


class RejectingRouter:
    async def route(
        self,
        user_request: str,
    ):

        raise RoutingContractError(
            "bad semantic plan"
        )


class NeverRuntime:
    async def run(
        self,
        task,
    ):

        raise AssertionError(
            "Runtime must not execute after "
            "routing contract rejection."
        )


class NeverPrimaryAssistant:
    async def respond(
        self,
        user_request: str,
    ):

        raise AssertionError(
            "Primary Assistant must not receive "
            "malformed governed routing output."
        )

    async def synthesize(
        self,
        user_request,
        results,
    ):

        raise AssertionError(
            "Synthesis must not execute after "
            "routing contract rejection."
        )


def test_orchestrator_fails_closed_on_routing_contract_error(
):

    orchestrator = (
        Orchestrator(
            router=(
                RejectingRouter()
            ),

            runtime=(
                NeverRuntime()
            ),

            primary_assistant=(
                NeverPrimaryAssistant()
            ),

            require_semantic_intent=True,
        )
    )

    result = (
        asyncio.run(
            orchestrator.run(
                "Reset jdoe's password."
            )
        )
    )

    assert (
        result.status
        == "error"
    )

    assert (
        result.routes
        == []
    )

    assert (
        result.results
        == []
    )

    assert (
        result.error
        == (
            "The request could not be safely interpreted "
            "into a valid governed execution plan."
        )
    )
