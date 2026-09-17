from subagents.core.definitions.types import (
    AgentDefinition,
    SemanticIntent,
)

from subagents.core.orchestration.semantic_guard import (
    SemanticGuard,
)


TOOLS = {
    "inspect_thing": {
        "description":
            "Inspect a thing.",

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments": [
            "thing_id",
        ],

        "parameters": {
            "thing_id": {
                "type":
                    "str",
            },
        },
    },

    "change_thing": {
        "description":
            "Change a thing.",

        "risk":
            "high",

        "requires_approval":
            True,

        "grounded_arguments": [
            "thing_id",
        ],

        "parameters": {
            "thing_id": {
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


def agent(
) -> AgentDefinition:

    return (
        AgentDefinition(
            name=(
                "thing-specialist"
            ),

            description=(
                "Handles generic things."
            ),

            model=(
                "thing-model"
            ),

            tools=[
                "inspect_thing",
                "change_thing",
            ],
        )
    )


def read_intent(
) -> SemanticIntent:

    return (
        SemanticIntent(
            summary=(
                "Inspect thing-123 only."
            ),

            effect=(
                "read"
            ),

            allowed_tools=[
                "inspect_thing",
            ],

            forbidden_tools=[
                "change_thing",
            ],

            allowed_arguments={
                "thing_id": [
                    "thing-123",
                ],
            },

            forbidden_arguments={
                "thing_id": [
                    "thing-999",
                ],
            },

            max_tool_calls=1,

            clarification_required=False,
        )
    )


def guard(
) -> SemanticGuard:

    return (
        SemanticGuard(
            tool_lookup=(
                lookup
            )
        )
    )


def test_guard_allows_matching_generic_capability_and_target(
):

    decision = (
        guard().evaluate(
            agent=(
                agent()
            ),

            intent=(
                read_intent()
            ),

            tool_name=(
                "inspect_thing"
            ),

            arguments={
                "thing_id":
                    "thing-123",
            },
        )
    )

    assert (
        decision.allowed
        is True
    )

    assert (
        decision.decision_code
        == "semantic_guard_allowed"
    )


def test_guard_blocks_explicitly_forbidden_capability(
):

    decision = (
        guard().evaluate(
            agent=(
                agent()
            ),

            intent=(
                read_intent()
            ),

            tool_name=(
                "change_thing"
            ),

            arguments={
                "thing_id":
                    "thing-123",
            },
        )
    )

    assert (
        decision.allowed
        is False
    )

    assert (
        decision.decision_code
        == "semantic_tool_forbidden"
    )


def test_guard_blocks_explicitly_forbidden_target(
):

    decision = (
        guard().evaluate(
            agent=(
                agent()
            ),

            intent=(
                read_intent()
            ),

            tool_name=(
                "inspect_thing"
            ),

            arguments={
                "thing_id":
                    "thing-999",
            },
        )
    )

    assert (
        decision.allowed
        is False
    )

    assert (
        decision.decision_code
        == "semantic_argument_forbidden"
    )


def test_guard_blocks_other_unapproved_grounded_target(
):

    decision = (
        guard().evaluate(
            agent=(
                agent()
            ),

            intent=(
                read_intent()
            ),

            tool_name=(
                "inspect_thing"
            ),

            arguments={
                "thing_id":
                    "thing-456",
            },
        )
    )

    assert (
        decision.allowed
        is False
    )

    assert (
        decision.decision_code
        == "semantic_argument_not_allowed"
    )


def test_guard_blocks_unbound_grounded_argument(
):

    intent = (
        read_intent()
    )

    intent.allowed_arguments = {}

    decision = (
        guard().evaluate(
            agent=(
                agent()
            ),

            intent=(
                intent
            ),

            tool_name=(
                "inspect_thing"
            ),

            arguments={
                "thing_id":
                    "thing-123",
            },
        )
    )

    assert (
        decision.allowed
        is False
    )

    assert (
        decision.decision_code
        == "semantic_argument_unbound"
    )


def test_guard_blocks_effect_mismatch(
):

    intent = (
        read_intent()
    )

    intent.effect = (
        "mutation"
    )

    decision = (
        guard().evaluate(
            agent=(
                agent()
            ),

            intent=(
                intent
            ),

            tool_name=(
                "inspect_thing"
            ),

            arguments={
                "thing_id":
                    "thing-123",
            },
        )
    )

    assert (
        decision.allowed
        is False
    )

    assert (
        decision.decision_code
        == "semantic_effect_mismatch"
    )


def test_guard_blocks_clarification_required(
):

    intent = (
        read_intent()
    )

    intent.clarification_required = (
        True
    )

    decision = (
        guard().evaluate(
            agent=(
                agent()
            ),

            intent=(
                intent
            ),

            tool_name=(
                "inspect_thing"
            ),

            arguments={
                "thing_id":
                    "thing-123",
            },
        )
    )

    assert (
        decision.allowed
        is False
    )

    assert (
        decision.decision_code
        == "semantic_clarification_required"
    )


def test_guard_has_no_domain_specific_dependency(
):

    decision = (
        guard().evaluate(
            agent=(
                agent()
            ),

            intent=(
                read_intent()
            ),

            tool_name=(
                "inspect_thing"
            ),

            arguments={
                "thing_id":
                    "thing-123",
            },
        )
    )

    assert (
        decision.allowed
        is True
    )

    assert (
        "account"
        not in decision.decision_code
    )

    assert (
        "ticket"
        not in decision.decision_code
    )

    assert (
        "access"
        not in decision.decision_code
    )
