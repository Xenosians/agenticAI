from subagents.core.definitions.loader import (
    load_agent_definition,
)


JIRA_PROJECT_TOOLS = {
    "jira_project_list",
    "jira_project_get",
    "jira_project_create",
    "jira_project_update",
    "jira_project_archive",
    "jira_project_delete",
}


JIRA_TICKET_TOOLS = {
    "ticket_get",
    "ticket_search",
    "ticket_history",
    "ticket_comments",
    "ticket_add_comment",
    "ticket_create",
    "ticket_assign",
    "ticket_transition",
}


def test_load_real_jira_specialist():

    agent = (
        load_agent_definition(
            "subagents/agents/jira-specialist.md"
        )
    )

    assert (
        agent.name
        == "jira-specialist"
    )

    assert (
        set(
            agent.tools
        )
        == (
            JIRA_PROJECT_TOOLS
            | JIRA_TICKET_TOOLS
        )
    )

    # jira-func is activated only after its isolated model
    # protocol evaluation passes.
    assert (
        agent.model
        == "hub-main"
    )

    assert (
        agent.max_steps
        == 2
    )

    for tool_name in agent.tools:

        assert (
            tool_name
            not in agent.system_prompt
        )


def test_jira_specialist_owns_projects_and_issues_as_distinct_resources():

    agent = (
        load_agent_definition(
            "subagents/agents/jira-specialist.md"
        )
    )

    description = (
        agent.description
        .lower()
    )

    prompt = (
        agent.system_prompt
        .lower()
    )

    assert (
        "project administration"
        in description
    )

    assert (
        "issue"
        in description
    )

    assert (
        "ticket"
        in description
    )

    assert (
        "primary resource"
        in prompt
    )

    assert (
        "issue search as project search"
        in prompt
    )

    assert (
        "project search as issue search"
        in prompt
    )
