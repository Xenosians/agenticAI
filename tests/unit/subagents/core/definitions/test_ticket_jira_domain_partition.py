from subagents.core.definitions.loader import (
    load_agent_definition,
)


def test_jira_specialist_is_provider_specific_superset():

    jira = (
        load_agent_definition(
            "subagents/agents/jira-specialist.md"
        )
    )

    ticket = (
        load_agent_definition(
            "subagents/agents/ticket-specialist.md"
        )
    )

    assert (
        set(
            ticket.tools
        )
        .issubset(
            set(
                jira.tools
            )
        )
    )


def test_generic_ticket_specialist_declares_fallback_role():

    ticket = (
        load_agent_definition(
            "subagents/agents/ticket-specialist.md"
        )
    )

    description = (
        ticket.description
        .lower()
    )

    prompt = (
        ticket.system_prompt
        .lower()
    )

    assert (
        "provider-neutral"
        in description
    )

    assert (
        "provider-specific"
        in prompt
    )

    assert (
        "more specific semantic domain"
        in prompt
    )


def test_role_prompts_do_not_hardcode_capability_names():

    agents = [
        load_agent_definition(
            "subagents/agents/jira-specialist.md"
        ),

        load_agent_definition(
            "subagents/agents/ticket-specialist.md"
        ),
    ]

    for agent in agents:

        for tool_name in agent.tools:

            assert (
                tool_name
                not in agent.system_prompt
            )
