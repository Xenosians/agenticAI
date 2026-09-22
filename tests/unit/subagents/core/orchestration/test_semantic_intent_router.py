import asyncio

from subagents.core.definitions.registry import (
    AgentRegistry,
)

from subagents.core.definitions.types import (
    AgentDefinition,
)

from subagents.core.orchestration.router import (
    LLMRouter,
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
        )
    )

    result.register(
        AgentDefinition(
            name=(
                "access-specialist"
            ),

            description=(
                "Handles access checks."
            ),

            model=(
                "access-test-model"
            ),

            tools=[
                "check_access",
            ],
        )
    )

    return result


def router_for(
    response: str,
):

    inference = (
        FakeInference(
            response
        )
    )

    router = (
        LLMRouter(
            registry=(
                registry()
            ),

            inference=(
                inference
            ),

            model_key=(
                "hub-main"
            ),
        )
    )

    return (
        router,
        inference,
    )


def test_router_parses_semantic_intent_contract(
):

    (
        router,
        _,
    ) = (
        router_for(
            """
            {
              "delegations": [
                {
                  "agent": "account-specialist",
                  "instructions": "Check bob's account status.",
                  "intent": {
                    "summary": "Check bob's account status.",
                    "effect": "read",
                    "allowed_tools": [
                      "account_status"
                    ],
                    "forbidden_tools": [
                      "reset_password"
                    ],
                    "allowed_arguments": {
                      "user_id": [
                        "bob"
                      ]
                    },
                    "forbidden_arguments": {
                      "user_id": [
                        "alice"
                      ]
                    },
                    "max_tool_calls": 1,
                    "clarification_required": false
                  }
                }
              ]
            }
            """
        )
    )

    delegations = (
        asyncio.run(
            router.route(
                (
                    "Do not reset bob's password. "
                    "Check bob's account status instead. "
                    "Alice is only the requester."
                )
            )
        )
    )

    assert (
        len(
            delegations
        )
        == 1
    )

    intent = (
        delegations[
            0
        ].semantic_intent
    )

    assert (
        intent
        is not None
    )

    assert (
        intent.effect
        == "read"
    )

    assert (
        intent.allowed_tools
        == [
            "account_status",
        ]
    )

    assert (
        intent.forbidden_tools
        == [
            "reset_password",
        ]
    )

    assert (
        intent.allowed_arguments
        == {
            "user_id": [
                "bob",
            ],
        }
    )

    assert (
        intent.forbidden_arguments
        == {
            "user_id": [
                "alice",
            ],
        }
    )

    assert (
        intent.max_tool_calls
        == 1
    )

    assert (
        intent.clarification_required
        is False
    )


def test_router_prompt_contains_generic_trusted_intent_metadata(
):

    (
        router,
        inference,
    ) = (
        router_for(
            """
            {
              "delegations": []
            }
            """
        )
    )

    asyncio.run(
        router.route(
            "Hello."
        )
    )

    prompt = (
        inference.calls[
            0
        ][
            "messages"
        ][
            0
        ][
            "content"
        ]
    )

    assert (
        '"intent_metadata"'
        in prompt
    )

    assert (
        '"effect": "read"'
        in prompt
    )

    assert (
        '"effect": "mutation"'
        in prompt
    )

    assert (
        '"grounded_arguments"'
        in prompt
    )


    assert (
        '"policy_owns_preconditions"'
        in prompt
    )

    assert (
        '"user_id"'
        in prompt
    )

    assert (
        '"resource"'
        in prompt
    )

    # Router contract metadata should not expose internal
    # provider or executor implementation.
    assert (
        "result_formatter"
        not in prompt
    )

    assert (
        "approval_formatter"
        not in prompt
    )

    assert (
        "policy_resolver"
        not in prompt
    )


def test_router_rejects_intent_with_untrusted_capability(
):

    (
        router,
        _,
    ) = (
        router_for(
            """
            {
              "delegations": [
                {
                  "agent": "account-specialist",
                  "instructions": "Do something.",
                  "intent": {
                    "summary": "Do something.",
                    "effect": "mutation",
                    "allowed_tools": [
                      "delete_everything"
                    ],
                    "forbidden_tools": [],
                    "allowed_arguments": {},
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

    delegations = (
        asyncio.run(
            router.route(
                "Do something."
            )
        )
    )

    assert (
        delegations
        == []
    )


def test_router_rejects_effect_mismatch_against_trusted_tool_metadata(
):

    (
        router,
        _,
    ) = (
        router_for(
            """
            {
              "delegations": [
                {
                  "agent": "account-specialist",
                  "instructions": "Unlock jdoe.",
                  "intent": {
                    "summary": "Unlock jdoe.",
                    "effect": "read",
                    "allowed_tools": [
                      "unlock_user"
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
                }
              ]
            }
            """
        )
    )

    delegations = (
        asyncio.run(
            router.route(
                "Unlock jdoe."
            )
        )
    )

    assert (
        delegations
        == []
    )


def test_router_rejects_unknown_grounded_argument_name(
):

    (
        router,
        _,
    ) = (
        router_for(
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
                      "invented_identifier": [
                        "jdoe"
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

    delegations = (
        asyncio.run(
            router.route(
                "Check jdoe."
            )
        )
    )

    assert (
        delegations
        == []
    )


def test_router_supports_clarification_contract_without_safe_tool(
):

    (
        router,
        _,
    ) = (
        router_for(
            """
            {
              "delegations": [
                {
                  "agent": "account-specialist",
                  "instructions":
                    "Clarify which account should be changed.",
                  "intent": {
                    "summary":
                      "The requested account target is ambiguous.",
                    "effect": "unknown",
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

    delegations = (
        asyncio.run(
            router.route(
                "Fix the account."
            )
        )
    )

    assert (
        len(
            delegations
        )
        == 1
    )

    intent = (
        delegations[
            0
        ].semantic_intent
    )

    assert (
        intent
        is not None
    )

    assert (
        intent.clarification_required
        is True
    )

    assert (
        intent.effect
        == "unknown"
    )

    assert (
        intent.allowed_tools
        == []
    )


def test_legacy_delegation_remains_readable_during_transition(
):

    (
        router,
        _,
    ) = (
        router_for(
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

    delegations = (
        asyncio.run(
            router.route(
                "Check jdoe."
            )
        )
    )

    assert (
        len(
            delegations
        )
        == 1
    )

    assert (
        delegations[
            0
        ].semantic_intent
        is None
    )
