from pathlib import (
    Path,
)

import pytest

from subagents.core.definitions.loader import (
    load_agent_directory,
)

from subagents.core.orchestration.intent_contract import (
    parse_semantic_intent,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[5]
)

AGENTS_DIR = (
    PROJECT_ROOT
    / "subagents"
    / "agents"
)


def knowledge_agent():

    agents = {
        agent.name:
            agent

        for agent
        in load_agent_directory(
            AGENTS_DIR
        )
    }

    return (
        agents[
            "knowledge-specialist"
        ]
    )


def test_non_grounded_capability_accepts_empty_semantic_argument_maps():

    intent = (
        parse_semantic_intent(
            {
                "summary": (
                    "Search trusted knowledge "
                    "for VPN troubleshooting."
                ),

                "effect":
                    "read",

                "allowed_tools": [
                    "knowledge_search",
                ],

                "forbidden_tools":
                    [],

                "allowed_arguments":
                    {},

                "forbidden_arguments":
                    {},

                "max_tool_calls":
                    1,

                "clarification_required":
                    False,
            },

            agent=(
                knowledge_agent()
            ),
        )
    )

    assert (
        intent
        is not None
    )

    assert (
        intent.allowed_arguments
        == {}
    )

    assert (
        intent.forbidden_arguments
        == {}
    )


def test_non_grounded_capability_rejects_untrusted_argument_binding():

    with pytest.raises(
        ValueError,
        match=(
            "untrusted or irrelevant grounded argument"
        ),
    ):

        parse_semantic_intent(
            {
                "summary": (
                    "Search trusted knowledge "
                    "for VPN troubleshooting."
                ),

                "effect":
                    "read",

                "allowed_tools": [
                    "knowledge_search",
                ],

                "forbidden_tools":
                    [],

                "allowed_arguments": {
                    "query": [
                        "VPN troubleshooting",
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
                knowledge_agent()
            ),
        )
