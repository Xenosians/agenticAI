from subagents.core.definitions.loader import (
    load_agent_definition,
)

from subagents.core.orchestration.intent_contract import (
    build_router_semantic_agent_spec,
)

from subagents.prompts.prompt_loader import (
    load_prompt,
)

from tools.registry import (
    get_tool,
)


def _capability_description(
    specialist_path: str,
    tool_name: str,
) -> str:
    agent = (
        load_agent_definition(
            specialist_path
        )
    )

    spec = (
        build_router_semantic_agent_spec(
            agent
        )
    )

    for capability in spec[
        "capabilities"
    ]:
        if capability[
            "name"
        ] == tool_name:
            return capability[
                "description"
            ].lower()

    raise AssertionError(
        f"Capability not found: {tool_name}"
    )


def test_jira_specialist_description_preserves_ticket_primary_resource():
    agent = (
        load_agent_definition(
            "subagents/agents/jira-specialist.md"
        )
    )

    description = (
        agent.description
        .lower()
    )

    assert "primary resource" in description
    assert "project key" in description
    assert "scope or filter" in description
    assert "project administration" in description


def test_ticket_search_router_metadata_is_contrastive_against_project_lookup():
    description = (
        _capability_description(
            "subagents/agents/jira-specialist.md",
            "ticket_search",
        )
    )

    # The ticket collection remains the primary resource.
    assert (
        "primary resources are tickets/issues"
        in description
    )

    # A Jira project is contextual scope, not the operation target.
    assert (
        "scope or filter"
        in description
    )

    # Ticket search must not collapse into project operations.
    assert (
        "project-list operation"
        in description
    )

    assert (
        "do not reinterpret"
        in description
    )

    assert (
        "project lookup"
        in description
    )


def test_ticket_create_router_metadata_is_contrastive_against_project_creation():
    description = (
        _capability_description(
            "subagents/agents/jira-specialist.md",
            "ticket_create",
        )
    )

    assert "primary resource being created" in description
    assert "ticket/issue" in description
    assert "project creation" in description
    assert "project lookup" in description


def test_jira_project_metadata_excludes_issue_member_operations():
    get_description = (
        _capability_description(
            "subagents/agents/jira-specialist.md",
            "jira_project_get",
        )
    )

    create_description = (
        _capability_description(
            "subagents/agents/jira-specialist.md",
            "jira_project_create",
        )
    )

    # Project lookup acts on the project resource itself.
    assert (
        "project itself is the primary resource"
        in get_description
    )

    # It explicitly excludes issue/ticket retrieval.
    assert (
        "ticket or issue"
        in get_description
    )

    assert (
        "tickets/issues inside"
        in get_description
    )

    # Project creation must remain distinct from ticket creation.
    assert (
        "project itself"
        in create_description
    )

    assert (
        "do not use it when the user asks to"
        in create_description
    )

    assert (
        "task, bug, story"
        in create_description
    )


def test_developer_specialist_description_preserves_git_operation_identity():
    agent = (
        load_agent_definition(
            "subagents/agents/developer-specialist.md"
        )
    )

    description = (
        agent.description
        .lower()
    )

    for phrase in [
        "overall status",
        "commit history",
        "unstaged diff",
        "staged diff",
        "branch creation",
        "branch switching",
        "commit",
        "push",
    ]:
        assert phrase in description

    assert "repository identity is scope" in description


def test_git_read_capability_descriptions_are_contrastive():
    expectations = {
        "workspace_git_status": [
            "current overall git status",
            "do not substitute commit history",
        ],
        "workspace_git_branches": [
            "list existing local git branches",
            "do not use it to create",
        ],
        "workspace_git_log": [
            "recent commit history",
            "do not substitute current git status",
        ],
        "workspace_git_diff": [
            "unstaged git patch/diff",
            "do not substitute a changed-file name overview",
        ],
        "workspace_git_staged_diff": [
            "staged/index git patch/diff",
            "do not substitute unstaged diff",
        ],
        "workspace_git_changed_files": [
            "names/overview",
            "do not substitute patch/diff contents",
        ],
    }

    for tool_name, phrases in expectations.items():
        tool = get_tool(tool_name)
        assert tool is not None

        description = (
            tool[
                "description"
            ]
            .lower()
        )

        for phrase in phrases:
            assert phrase in description


def test_generic_hub_prompt_has_member_vs_container_rule_without_tool_names():
    prompt = (
        load_prompt(
            "hub_router.txt"
        )
    )

    normalized = (
        " ".join(
            prompt.split()
        )
    )

    assert "<operation> <primary-resource> in/inside/within/under <context>" in normalized
    assert "containing resource remains context" in (
        " ".join(
            load_prompt(
                "hub_router_repair.txt"
            ).split()
        )
    )

    # Generic Hub templates remain capability-name agnostic.
    assert "ticket_create" not in prompt
    assert "jira_project_get" not in prompt
    assert "workspace_git_status" not in prompt
