import pytest

from subagents.core.definitions.types import (
    AgentDefinition,
)

from subagents.core.orchestration.intent_contract import (
    build_router_semantic_agent_spec,
    parse_semantic_intent,
)


TOOLS = {
    "inspect_repository": {
        "description": (
            "Inspect one configured logical repository."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments": [
            "repository",
        ],

        "parameters": {
            "repository": {
                "type":
                    "str",
            },
        },

        "argument_values_resolver": (
            lambda: {
                "repository": [
                    "ai",
                    "backend",
                    "frontend",
                ],
            }
        ),
    },

    "check_resource": {
        "description": (
            "Check one exact resource."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments": [
            "resource",
        ],

        "parameters": {
            "resource": {
                "type":
                    "str",
            },
        },
    },
}


def lookup(
    name: str,
):

    return (
        TOOLS.get(
            name
        )
    )


def repository_agent(
) -> AgentDefinition:

    return (
        AgentDefinition(
            name=(
                "repository-specialist"
            ),

            description=(
                "Handles repository inspection."
            ),

            model=(
                "test-model"
            ),

            tools=[
                "inspect_repository",
            ],
        )
    )


def resource_agent(
) -> AgentDefinition:

    return (
        AgentDefinition(
            name=(
                "resource-specialist"
            ),

            description=(
                "Handles exact resource checks."
            ),

            model=(
                "test-model"
            ),

            tools=[
                "check_resource",
            ],
        )
    )


def test_router_spec_exposes_trusted_bounded_argument_values(
):

    spec = (
        build_router_semantic_agent_spec(
            repository_agent(),

            tool_lookup=(
                lookup
            ),
        )
    )

    capability = (
        spec[
            "capabilities"
        ][
            0
        ]
    )

    metadata = (
        capability[
            "intent_metadata"
        ]
    )

    assert (
        metadata[
            "bounded_argument_values"
        ]
        == {
            "repository": [
                "ai",
                "backend",
                "frontend",
            ],
        }
    )


def test_bounded_argument_is_canonicalized_case_insensitively(
):

    intent = (
        parse_semantic_intent(
            {
                "summary":
                    "Inspect the AI repository.",

                "effect":
                    "read",

                "allowed_tools": [
                    "inspect_repository",
                ],

                "forbidden_tools":
                    [],

                "allowed_arguments": {
                    "repository": [
                        "AI",
                    ],
                },

                "forbidden_arguments":
                    {},

                "max_tool_calls":
                    1,

                "clarification_required":
                    False,
            },

            agent=(
                repository_agent()
            ),

            tool_lookup=(
                lookup
            ),
        )
    )

    assert (
        intent
        is not None
    )

    assert (
        intent.allowed_arguments
        == {
            "repository": [
                "ai",
            ],
        }
    )


def test_exact_bounded_value_remains_unchanged(
):

    intent = (
        parse_semantic_intent(
            {
                "summary":
                    "Inspect the backend repository.",

                "effect":
                    "read",

                "allowed_tools": [
                    "inspect_repository",
                ],

                "forbidden_tools":
                    [],

                "allowed_arguments": {
                    "repository": [
                        "backend",
                    ],
                },

                "forbidden_arguments":
                    {},

                "max_tool_calls":
                    1,

                "clarification_required":
                    False,
            },

            agent=(
                repository_agent()
            ),

            tool_lookup=(
                lookup
            ),
        )
    )

    assert (
        intent
        is not None
    )

    assert (
        intent.allowed_arguments[
            "repository"
        ]
        == [
            "backend",
        ]
    )


def test_unknown_bounded_value_fails_closed(
):

    with pytest.raises(
        ValueError,
        match=(
            "does not match any trusted bounded value"
        ),
    ):

        parse_semantic_intent(
            {
                "summary":
                    "Inspect the production repository.",

                "effect":
                    "read",

                "allowed_tools": [
                    "inspect_repository",
                ],

                "forbidden_tools":
                    [],

                "allowed_arguments": {
                    "repository": [
                        "production",
                    ],
                },

                "forbidden_arguments":
                    {},

                "max_tool_calls":
                    1,

                "clarification_required":
                    False,
            },

            agent=(
                repository_agent()
            ),

            tool_lookup=(
                lookup
            ),
        )


def test_unbounded_grounded_value_is_not_normalized(
):

    intent = (
        parse_semantic_intent(
            {
                "summary":
                    "Check resource admin.",

                "effect":
                    "read",

                "allowed_tools": [
                    "check_resource",
                ],

                "forbidden_tools":
                    [],

                "allowed_arguments": {
                    "resource": [
                        "resource admin",
                    ],
                },

                "forbidden_arguments":
                    {},

                "max_tool_calls":
                    1,

                "clarification_required":
                    False,
            },

            agent=(
                resource_agent()
            ),

            tool_lookup=(
                lookup
            ),
        )
    )

    assert (
        intent
        is not None
    )

    assert (
        intent.allowed_arguments
        == {
            "resource": [
                "resource admin",
            ],
        }
    )


def test_allowed_and_forbidden_values_conflict_after_canonicalization(
):

    with pytest.raises(
        ValueError,
        match=(
            "cannot both allow and forbid"
        ),
    ):

        parse_semantic_intent(
            {
                "summary":
                    "Inspect AI but somehow forbid ai.",

                "effect":
                    "read",

                "allowed_tools": [
                    "inspect_repository",
                ],

                "forbidden_tools":
                    [],

                "allowed_arguments": {
                    "repository": [
                        "AI",
                    ],
                },

                "forbidden_arguments": {
                    "repository": [
                        "ai",
                    ],
                },

                "max_tool_calls":
                    1,

                "clarification_required":
                    False,
            },

            agent=(
                repository_agent()
            ),

            tool_lookup=(
                lookup
            ),
        )