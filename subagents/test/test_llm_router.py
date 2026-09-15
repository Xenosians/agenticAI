import asyncio

from subagents.core.llm_router import (
    LLMRouter,
)

from subagents.core.registry import (
    AgentRegistry,
)

from subagents.core.types import (
    AgentDefinition,
    SpecialistRequest,
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


def build_registry(
) -> AgentRegistry:

    registry = (
        AgentRegistry()
    )

    registry.register(
        AgentDefinition(
            name=(
                "account-specialist"
            ),

            description=(
                "Handles user accounts."
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

    registry.register(
        AgentDefinition(
            name=(
                "access-specialist"
            ),

            description=(
                "Handles access and permissions."
            ),

            model=(
                "access-test-model"
            ),

            tools=[
                "check_access",
            ],
        )
    )

    return registry


def build_router(
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
                build_registry()
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


def test_llm_router_creates_structured_account_delegation():

    (
        router,
        inference,
    ) = build_router(
        """
        {
          "delegations": [
            {
              "agent": "account-specialist",
              "instructions":
                "Determine whether account jdoe is locked."
            }
          ]
        }
        """
    )

    delegations = (
        asyncio.run(
            router.route(
                "Is jdoe locked?"
            )
        )
    )

    assert delegations == [
        SpecialistRequest(
            agent_name=(
                "account-specialist"
            ),

            instructions=(
                "Determine whether account "
                "jdoe is locked."
            ),
        )
    ]

    assert (
        inference.calls[
            0
        ][
            "model_key"
        ]
        == "hub-main"
    )


def test_router_prompt_contains_trusted_capability_catalog():

    (
        router,
        inference,
    ) = build_router(
        """
        {
          "delegations": []
        }
        """
    )

    asyncio.run(
        router.route(
            "Hello."
        )
    )

    system_prompt = (
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
        "account-specialist"
        in system_prompt
    )

    assert (
        "account_status"
        in system_prompt
    )

    assert (
        "unlock_user"
        in system_prompt
    )

    assert (
        "reset_password"
        in system_prompt
    )

    assert (
        "check_access"
        in system_prompt
    )

    assert (
        "argument_schema"
        not in system_prompt
    )


def test_llm_router_supports_multiple_delegations():

    (
        router,
        _inference,
    ) = build_router(
        """
        {
          "delegations": [
            {
              "agent": "account-specialist",
              "instructions":
                "Check account jdoe."
            },
            {
              "agent": "access-specialist",
              "instructions":
                "Check VPN access for jdoe."
            }
          ]
        }
        """
    )

    delegations = (
        asyncio.run(
            router.route(
                "Check jdoe account "
                "and VPN access."
            )
        )
    )

    assert [
        item.agent_name

        for item
        in delegations
    ] == [
        "account-specialist",
        "access-specialist",
    ]

    assert (
        delegations[
            1
        ].instructions
        == "Check VPN access for jdoe."
    )


def test_llm_router_rejects_unknown_agent():

    (
        router,
        _inference,
    ) = build_router(
        """
        {
          "delegations": [
            {
              "agent": "account-specialist",
              "instructions": "Check jdoe."
            },
            {
              "agent": "fake-specialist",
              "instructions": "Do something."
            }
          ]
        }
        """
    )

    delegations = (
        asyncio.run(
            router.route(
                "Check jdoe."
            )
        )
    )

    assert [
        item.agent_name

        for item
        in delegations
    ] == [
        "account-specialist"
    ]


def test_llm_router_rejects_empty_instructions():

    (
        router,
        _inference,
    ) = build_router(
        """
        {
          "delegations": [
            {
              "agent": "account-specialist",
              "instructions": ""
            }
          ]
        }
        """
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


def test_llm_router_rejects_invalid_json():

    (
        router,
        _inference,
    ) = build_router(
        (
            "I think "
            "account-specialist "
            "should do it."
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


def test_llm_router_handles_no_match():

    (
        router,
        _inference,
    ) = build_router(
        """
        {
          "delegations": []
        }
        """
    )

    delegations = (
        asyncio.run(
            router.route(
                "Tell me a joke."
            )
        )
    )

    assert (
        delegations
        == []
    )