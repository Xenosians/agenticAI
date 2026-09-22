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


def test_workspace_mkdir_explicitly_excludes_git_staging():

    tool = (
        get_tool(
            "workspace_mkdir"
        )
    )

    assert (
        tool
        is not None
    )

    description = (
        tool[
            "description"
        ]
        .lower()
    )

    assert (
        "directory"
        in description
    )

    assert (
        "git staging"
        in description
        or "git stage"
        in description
    )


def test_git_stage_capability_explicitly_describes_git_index_operation():

    tool = (
        get_tool(
            "workspace_git_stage_files"
        )
    )

    assert (
        tool
        is not None
    )

    description = (
        tool[
            "description"
        ]
        .lower()
    )

    assert (
        "git stage"
        in description
    )

    assert (
        "git index"
        in description
    )

    assert (
        "does not create directories"
        in description
    )


def test_developer_router_spec_contains_both_contrastive_capabilities():

    agent = (
        load_agent_definition(
            "subagents/agents/developer-specialist.md"
        )
    )

    spec = (
        build_router_semantic_agent_spec(
            agent
        )
    )

    capabilities = {
        capability[
            "name"
        ]:
            capability

        for capability
        in spec[
            "capabilities"
        ]
    }

    assert (
        "workspace_mkdir"
        in capabilities
    )

    assert (
        "workspace_git_stage_files"
        in capabilities
    )

    mkdir_description = (
        capabilities[
            "workspace_mkdir"
        ][
            "description"
        ]
        .lower()
    )

    stage_description = (
        capabilities[
            "workspace_git_stage_files"
        ][
            "description"
        ]
        .lower()
    )

    assert (
        "directory"
        in mkdir_description
    )

    assert (
        "git stage"
        in stage_description
    )


def test_hub_prompt_requires_operation_preservation():

    prompt = (
        load_prompt(
            "hub_router.txt"
        )
    )

    assert (
        "CAPABILITY SELECTION DISAMBIGUATION"
        in prompt
    )

    assert (
        "Preserve the user's requested OPERATION"
        in prompt
    )

    assert (
        "Stage src/app.py in the AI repository."
        in prompt
    )

    assert (
        "Create a directory named artifacts."
        in prompt
    )


def test_conditional_workflow_prompt_forbids_synthesized_preflight_reads():

    prompt = (
        load_prompt(
            "hub_conditional_workflow.txt"
        )
    )

    assert (
        "DO NOT SYNTHESIZE PREFLIGHT WORKFLOWS"
        in prompt
    )

    assert (
        "Do NOT invent a read-before-mutation workflow"
        in prompt
    )

    assert (
        "Did the USER explicitly request a result-dependent sequence?"
        in prompt
    )

    assert (
        "Switch the AI repository to feature/test."
        in prompt
    )


def test_git_switch_description_owns_trusted_preconditions():

    tool = (
        get_tool(
            "workspace_git_switch_branch"
        )
    )

    assert tool is not None

    description = (
        tool[
            "description"
        ]
        .lower()
    )

    assert (
        "select this mutation capability directly"
        in description
    )

    assert (
        "trusted policy validates local branch existence"
        in description
    )

    assert (
        "without a separate read delegation"
        in description
    )
