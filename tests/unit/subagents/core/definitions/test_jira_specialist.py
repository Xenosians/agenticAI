from subagents.core.definitions.loader import (
    load_agent_definition,
)


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
        == {
            "jira_project_list",
            "jira_project_get",
            "jira_project_create",
            "jira_project_update",
            "jira_project_archive",
        }
    )

    assert (
        agent.model
        == "hub-main"
    )

    assert (
        agent.max_steps
        == 2
    )

    # Exact capability IDs stay in YAML/runtime metadata only,
    # never the free-form role prompt.
    for tool_name in (
        agent.tools
    ):
        assert (
            tool_name
            not in agent.system_prompt
        )
