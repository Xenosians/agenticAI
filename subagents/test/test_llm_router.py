import asyncio

from subagents.core.llm_router import (
    LLMRouter,
)

from subagents.core.types import (
    AgentDefinition,
)

from subagents.core.registry import (
    AgentRegistry,
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

        return self.response


def build_registry():
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

    router = LLMRouter(
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

    return (
        router,
        inference,
    )


def test_llm_router_selects_account_agent():
    (
        router,
        inference,
    ) = build_router(
        '{"agents": ["account-specialist"]}'
    )

    routes = asyncio.run(
        router.route(
            "Is jdoe locked?"
        )
    )

    assert routes == [
        "account-specialist"
    ]

    assert (
        inference.calls[0][
            "model_key"
        ]
        == "hub-main"
    )


def test_llm_router_supports_multiple_agents():
    (
        router,
        _inference,
    ) = build_router(
        (
            '{"agents": ['
            '"account-specialist", '
            '"access-specialist"'
            ']}'
        )
    )

    routes = asyncio.run(
        router.route(
            "Check jdoe account "
            "and VPN access."
        )
    )

    assert routes == [
        "account-specialist",
        "access-specialist",
    ]


def test_llm_router_rejects_unknown_agent():
    (
        router,
        _inference,
    ) = build_router(
        (
            '{"agents": ['
            '"account-specialist", '
            '"fake-specialist"'
            ']}'
        )
    )

    routes = asyncio.run(
        router.route(
            "Check jdoe."
        )
    )

    assert routes == [
        "account-specialist"
    ]


def test_llm_router_rejects_invalid_json():
    (
        router,
        _inference,
    ) = build_router(
        (
            "I think account-specialist "
            "should do it."
        )
    )

    routes = asyncio.run(
        router.route(
            "Check jdoe."
        )
    )

    assert routes == []


def test_llm_router_handles_no_match():
    (
        router,
        _inference,
    ) = build_router(
        '{"agents": []}'
    )

    routes = asyncio.run(
        router.route(
            "Tell me a joke."
        )
    )

    assert routes == []