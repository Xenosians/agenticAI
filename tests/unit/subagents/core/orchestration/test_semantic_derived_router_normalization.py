from inspect import signature

import pytest

from subagents.core.definitions.loader import (
    load_agent_definition,
)

from subagents.core.orchestration.intent_contract import (
    parse_semantic_intent,
)


def ticket_agent():
    return (
        load_agent_definition(
            "subagents/agents/ticket-specialist.md"
        )
    )


def parse_for_request(
    payload: dict,
    request: str,
):
    """
    Keep this regression compatible with the current parser API while
    the request-origin contract evolves.
    """

    kwargs = {
        "agent":
            ticket_agent(),
    }

    parameters = (
        signature(
            parse_semantic_intent
        )
        .parameters
    )

    for candidate in (
        "user_request",
        "current_user_request",
        "request",
    ):

        if candidate in parameters:
            kwargs[
                candidate
            ] = request
            break

    return (
        parse_semantic_intent(
            payload,
            **kwargs,
        )
    )


def base_payload():
    return {
        "summary":
            (
                "Create an internal meeting "
                "ticket for next Sunday"
            ),

        "effect":
            "mutation",

        "allowed_tools": [
            "ticket_create",
        ],

        "forbidden_tools":
            [],

        "allowed_arguments": {
            "project_key": [
                "KAN",
            ],

            # Hub mistake:
            #
            # summary is trusted DERIVED content,
            # not an authority-bearing semantic binding.
            "summary": [
                "Internal meeting next Sunday",
            ],
        },

        "forbidden_arguments":
            {},

        "max_tool_calls":
            1,

        "clarification_required":
            False,
    }


def test_known_derived_argument_is_narrowed_to_absence():

    intent = (
        parse_for_request(
            base_payload(),

            (
                "create an internal meeting ticket "
                "for next sunday in project KAN"
            ),
        )
    )

    assert (
        intent.allowed_arguments[
            "project_key"
        ]
        == [
            "KAN",
        ]
    )

    assert (
        "summary"
        not in intent.allowed_arguments
    )


def test_unknown_argument_still_fails_closed():

    payload = (
        base_payload()
    )

    payload[
        "allowed_arguments"
    ][
        "priority"
    ] = [
        "High",
    ]

    with pytest.raises(
        ValueError,
        match=(
            "untrusted|irrelevant|grounded"
        ),
    ):

        parse_for_request(
            payload,

            (
                "create an internal meeting ticket "
                "for next sunday in project KAN"
            ),
        )
