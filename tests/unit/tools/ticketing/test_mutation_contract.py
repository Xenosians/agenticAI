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

from tools.registry import (
    get_tool,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[4]
)

AGENTS_DIR = (
    PROJECT_ROOT
    / "subagents"
    / "agents"
)


def ticket_agent():
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
            "ticket-specialist"
        ]
    )


def test_ticket_specialist_exposes_all_governed_ticket_mutations():

    agent = (
        ticket_agent()
    )

    assert {
        "ticket_add_comment",
        "ticket_create",
        "ticket_assign",
        "ticket_transition",
    }.issubset(
        set(
            agent.tools
        )
    )


def test_ticket_mutation_risk_and_approval_contract():

    expected = {
        "ticket_add_comment": [
            "ticket_key",
            "comment",
        ],

        "ticket_create": [
            "project_key",
            "summary",
            "ticket_type",
        ],

        "ticket_assign": [
            "ticket_key",
            "assignee",
        ],

        "ticket_transition": [
            "ticket_key",
            "status",
        ],
    }

    for (
        tool_name,
        grounded_arguments,
    ) in expected.items():

        tool = (
            get_tool(
                tool_name
            )
        )

        assert (
            tool
            is not None
        )

        expected_risk = (
            "medium"
            if tool_name
            == "ticket_create"
            else "low"
        )

        assert (
            tool[
                "risk"
            ]
            == expected_risk
        )

        assert (
            tool[
                "requires_approval"
            ]
            is True
        )

        assert (
            tool[
                "grounded_arguments"
            ]
            == grounded_arguments
        )


def test_semantic_guard_allows_exact_ticket_transition():

    agent = (
        ticket_agent()
    )

    guard = (
        SemanticGuard()
    )

    intent = (
        SemanticIntent(
            summary=(
                "Move ITSM-101 to Resolved."
            ),

            effect=(
                "mutation"
            ),

            allowed_tools=[
                "ticket_transition",
            ],

            forbidden_tools=[],

            allowed_arguments={
                "ticket_key": [
                    "ITSM-101",
                ],

                "status": [
                    "Resolved",
                ],
            },

            forbidden_arguments={},

            max_tool_calls=1,

            clarification_required=False,
        )
    )

    decision = (
        guard.evaluate(
            agent=(
                agent
            ),

            intent=(
                intent
            ),

            tool_name=(
                "ticket_transition"
            ),

            arguments={
                "ticket_key":
                    "ITSM-101",

                "status":
                    "Resolved",
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


def test_semantic_guard_blocks_wrong_transition_status():

    agent = (
        ticket_agent()
    )

    guard = (
        SemanticGuard()
    )

    intent = (
        SemanticIntent(
            summary=(
                "Move ITSM-101 to Resolved."
            ),

            effect=(
                "mutation"
            ),

            allowed_tools=[
                "ticket_transition",
            ],

            forbidden_tools=[],

            allowed_arguments={
                "ticket_key": [
                    "ITSM-101",
                ],

                "status": [
                    "Resolved",
                ],
            },

            forbidden_arguments={},

            max_tool_calls=1,

            clarification_required=False,
        )
    )

    decision = (
        guard.evaluate(
            agent=(
                agent
            ),

            intent=(
                intent
            ),

            tool_name=(
                "ticket_transition"
            ),

            arguments={
                "ticket_key":
                    "ITSM-101",

                "status":
                    "Closed",
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


def test_semantic_guard_blocks_wrong_ticket_assignee():

    agent = (
        ticket_agent()
    )

    guard = (
        SemanticGuard()
    )

    intent = (
        SemanticIntent(
            summary=(
                "Assign ITSM-101 to alice."
            ),

            effect=(
                "mutation"
            ),

            allowed_tools=[
                "ticket_assign",
            ],

            forbidden_tools=[],

            allowed_arguments={
                "ticket_key": [
                    "ITSM-101",
                ],

                "assignee": [
                    "alice",
                ],
            },

            forbidden_arguments={
                "assignee": [
                    "bob",
                ],
            },

            max_tool_calls=1,

            clarification_required=False,
        )
    )

    decision = (
        guard.evaluate(
            agent=(
                agent
            ),

            intent=(
                intent
            ),

            tool_name=(
                "ticket_assign"
            ),

            arguments={
                "ticket_key":
                    "ITSM-101",

                "assignee":
                    "bob",
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


def test_semantic_guard_blocks_rewritten_ticket_summary():

    agent = (
        ticket_agent()
    )

    guard = (
        SemanticGuard()
    )

    intent = (
        SemanticIntent(
            summary=(
                "Create a ticket in ITSM titled "
                "'Laptop onboarding failure'."
            ),

            effect=(
                "mutation"
            ),

            allowed_tools=[
                "ticket_create",
            ],

            forbidden_tools=[],

            allowed_arguments={
                "project_key": [
                    "ITSM",
                ],

                "summary": [
                    "Laptop onboarding failure",
                ],
            },

            forbidden_arguments={},

            max_tool_calls=1,

            clarification_required=False,
        )
    )

    decision = (
        guard.evaluate(
            agent=(
                agent
            ),

            intent=(
                intent
            ),

            tool_name=(
                "ticket_create"
            ),

            arguments={
                "project_key":
                    "ITSM",

                "summary":
                    "Employee laptop is broken",
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
