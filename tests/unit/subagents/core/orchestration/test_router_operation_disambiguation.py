from pathlib import (
    Path,
)

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
    TOOLS,
    get_tool,
)


def test_workspace_mkdir_explicitly_excludes_git_staging():

    tool = (
        get_tool(
            "workspace_mkdir"
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

    assert tool is not None

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


def test_developer_router_spec_contains_contrastive_capabilities():

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

    assert (
        "workspace_git_switch_branch"
        in capabilities
    )

    assert (
        capabilities[
            "workspace_git_switch_branch"
        ][
            "intent_metadata"
        ][
            "policy_owns_preconditions"
        ]
        is True
    )

    assert (
        capabilities[
            "workspace_git_branches"
        ][
            "intent_metadata"
        ][
            "policy_owns_preconditions"
        ]
        is False
    )


def test_hub_prompt_uses_generic_operation_preservation():

    prompt = (
        load_prompt(
            "hub_router.txt"
        )
    )

    normalized_prompt = (
        " ".join(
            prompt.split()
        )
    )

    assert (
        "Preserve the user's requested OPERATION"
        in normalized_prompt
    )

    assert (
        "must contain exactly ONE capability"
        in normalized_prompt
    )

    assert (
        "policy_owns_preconditions"
        in normalized_prompt
    )

    assert (
        "do NOT add a read delegation"
        in normalized_prompt
    )

    assert (
        "Never turn an unconditional user request "
        "into a conditional workflow."
        in normalized_prompt
    )


def test_generic_hub_templates_do_not_hardcode_registered_capability_names():

    templates = [
        load_prompt(
            "hub_router.txt"
        ),

        load_prompt(
            "hub_conditional_workflow.txt"
        ),

        load_prompt(
            "hub_router_repair.txt"
        ),
    ]

    for template in templates:

        for tool_name in TOOLS:

            assert (
                tool_name
                not in template
            ), (
                "Generic Hub prompt hardcodes registered "
                f"capability '{tool_name}'."
            )


def test_generic_hub_templates_do_not_embed_domain_examples():

    templates = "\n".join(
        [
            load_prompt(
                "hub_router.txt"
            ),

            load_prompt(
                "hub_conditional_workflow.txt"
            ),

            load_prompt(
                "hub_router_repair.txt"
            ),
        ]
    )

    forbidden_examples = [
        "AI repository",
        "VPN",
        "src/app.py",
        "test/agentic-git-smoke",
        "Git branch",
        "Git staging",
    ]

    for value in forbidden_examples:

        assert (
            value
            not in templates
        ), (
            "Generic Hub prompt contains domain-specific "
            f"example '{value}'."
        )


def test_generic_orchestration_runtime_does_not_hardcode_capability_names():

    paths = [
        Path(
            "subagents/core/orchestration/router.py"
        ),
        Path(
            "subagents/core/orchestration/orchestrator.py"
        ),
        Path(
            "subagents/core/orchestration/semantic_guard.py"
        ),
        Path(
            "subagents/core/orchestration/condition_contract.py"
        ),
        Path(
            "subagents/core/orchestration/intent_contract.py"
        ),
    ]

    for path in paths:

        source = path.read_text(
            encoding="utf-8"
        )

        for tool_name in TOOLS:

            assert (
                tool_name
                not in source
            ), (
                f"{path} hardcodes capability "
                f"'{tool_name}'."
            )



def test_hub_prompt_preserves_primary_resource_over_context():

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

    assert (
        "PRIMARY RESOURCE or object"
        in normalized
    )

    assert (
        "scope, filter, grouping value, or contextual resource"
        in normalized
    )

    assert (
        "requested operation"
        in normalized
    )

    assert (
        "primary resource"
        in normalized
    )
