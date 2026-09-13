from subagents.core.types import (
    AgentDefinition,
)

from subagents.core.registry import (
    AgentRegistry,
)


def build_test_agent(
) -> AgentDefinition:
    return AgentDefinition(
        name=(
            "account-specialist"
        ),
        description=(
            "Handles account-related "
            "requests."
        ),
        model=(
            "test-model"
        ),
        tools=[
            "account_status",
        ],
    )


def test_register_and_get_agent():
    registry = (
        AgentRegistry()
    )

    agent = (
        build_test_agent()
    )

    registry.register(
        agent
    )

    assert (
        registry.exists(
            "account-specialist"
        )
        is True
    )

    assert (
        registry.get(
            "account-specialist"
        )
        == agent
    )

    assert (
        registry.list_agents()
        == [
            agent
        ]
    )


def test_duplicate_agent_raises_error():
    registry = (
        AgentRegistry()
    )

    agent = (
        build_test_agent()
    )

    registry.register(
        agent
    )

    try:
        registry.register(
            agent
        )

        assert (
            False
        ), (
            "Expected ValueError "
            "for duplicate agent"
        )

    except ValueError:
        pass


def test_unknown_agent_raises_error():
    registry = (
        AgentRegistry()
    )

    try:
        registry.get(
            "does-not-exist"
        )

        assert (
            False
        ), (
            "Expected KeyError "
            "for unknown agent"
        )

    except KeyError:
        pass