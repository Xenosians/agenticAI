from pathlib import (
    Path,
)

from subagents.core.definitions.loader import (
    load_agent_directory,
)

from subagents.core.definitions.types import (
    SemanticIntent,
)

from subagents.core.orchestration.semantic_guard import (
    SemanticGuard,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[4]
)


def access_agent():

    agents = {
        agent.name:
            agent

        for agent
        in load_agent_directory(
            PROJECT_ROOT
            / "subagents"
            / "agents"
        )
    }

    return (
        agents[
            "access-specialist"
        ]
    )


def test_access_specialist_exposes_read_and_mutations():

    agent = (
        access_agent()
    )

    assert (
        set(
            agent.tools
        )
        == {
            "check_access",
            "grant_access",
            "revoke_access",
        }
    )


def test_semantic_guard_allows_exact_grant():

    agent = (
        access_agent()
    )

    guard = (
        SemanticGuard()
    )

    intent = (
        SemanticIntent(
            summary=(
                "Grant jdoe access to VPN."
            ),

            effect="mutation",

            allowed_tools=[
                "grant_access",
            ],

            forbidden_tools=[],

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

    result = (
        guard.evaluate(
            agent=(
                agent
            ),

            intent=(
                intent
            ),

            tool_name=(
                "grant_access"
            ),

            arguments={
                "user_id":
                    "jdoe",

                "resource":
                    "VPN",
            },
        )
    )

    assert result.allowed is True


def test_semantic_guard_blocks_grant_when_only_check_requested():

    agent = (
        access_agent()
    )

    guard = (
        SemanticGuard()
    )

    intent = (
        SemanticIntent(
            summary=(
                "Check jdoe's VPN access."
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

    result = (
        guard.evaluate(
            agent=(
                agent
            ),

            intent=(
                intent
            ),

            tool_name=(
                "grant_access"
            ),

            arguments={
                "user_id":
                    "jdoe",

                "resource":
                    "VPN",
            },
        )
    )

    assert result.allowed is False

    assert (
        result.decision_code
        == "semantic_tool_forbidden"
    )


def test_semantic_guard_blocks_wrong_resource():

    agent = (
        access_agent()
    )

    guard = (
        SemanticGuard()
    )

    intent = (
        SemanticIntent(
            summary=(
                "Grant jdoe access to VPN."
            ),

            effect="mutation",

            allowed_tools=[
                "grant_access",
            ],

            forbidden_tools=[],

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

    result = (
        guard.evaluate(
            agent=(
                agent
            ),

            intent=(
                intent
            ),

            tool_name=(
                "grant_access"
            ),

            arguments={
                "user_id":
                    "jdoe",

                "resource":
                    "admin",
            },
        )
    )

    assert result.allowed is False

    assert (
        result.decision_code
        == "semantic_argument_not_allowed"
    )


def test_semantic_guard_blocks_grant_revoke_inversion():

    agent = (
        access_agent()
    )

    guard = (
        SemanticGuard()
    )

    intent = (
        SemanticIntent(
            summary=(
                "Revoke jdoe's VPN access."
            ),

            effect="mutation",

            allowed_tools=[
                "revoke_access",
            ],

            forbidden_tools=[
                "grant_access",
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

    result = (
        guard.evaluate(
            agent=(
                agent
            ),

            intent=(
                intent
            ),

            tool_name=(
                "grant_access"
            ),

            arguments={
                "user_id":
                    "jdoe",

                "resource":
                    "VPN",
            },
        )
    )

    assert result.allowed is False

    assert (
        result.decision_code
        == "semantic_tool_forbidden"
    )
