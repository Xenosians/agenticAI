"""Folder-name aliases for configured repositories and upper-case Jira keys.

Users name repositories by their folders ("agenticaiPersonal") and type Jira
keys in any case ("sup-11"). Both resolve to the one trusted canonical value
behind the scenes; nothing outside the configured repositories or the user's
own words becomes reachable.
"""
from pathlib import Path

import pytest

from services.git_repositories import GitRepositoryRegistry
from subagents.core.definitions.types import AgentDefinition
from subagents.core.orchestration.intent_contract import parse_semantic_intent
from subagents.core.tooling.capabilities import (
    normalize_argument_values,
    trusted_argument_aliases,
)
from subagents.core.tooling.gateway import validate_grounded_arguments
from tools.registry import TOOLS

JIRA_KEY_ARGUMENTS = {"ticket_key", "other_ticket_key", "project_key", "request_key", "project_id_or_key"}


@pytest.fixture
def registry(tmp_path):
    layout = {
        "ai": "agenticaiPersonal/itsm-agent",
        "backend": "agenticBackend/itsm_backend",
        "frontend": "agenticFrontend",
        "mvp": "projectOps/.runtime/mvp-workspace/workspace",
        "tools": "shared/tools",
        "docs": "shared/docs",
    }
    for relative in layout.values():
        (tmp_path / relative / ".git").mkdir(parents=True)
    return GitRepositoryRegistry(
        root=tmp_path,
        repositories={name: Path(relative) for name, relative in layout.items()},
        default_repository="ai",
    )


def test_folder_names_on_a_configured_path_name_that_repository(registry):
    aliases = registry.aliases()
    assert aliases["ai"] == ["agenticaipersonal", "itsm-agent"]
    assert aliases["backend"] == ["agenticbackend", "itsm_backend"]
    assert aliases["frontend"] == ["agenticfrontend"]


def test_a_repository_inside_a_hidden_folder_is_named_only_by_its_configured_name(registry):
    # Its folders are generic words ("workspace") and internal runtime paths.
    assert registry.aliases()["mvp"] == []
    with pytest.raises(ValueError):
        registry.resolve("workspace")


def test_folder_names_shared_by_two_repositories_name_neither(registry):
    # "shared" holds both tools and docs, so it identifies no single repository.
    aliases = registry.aliases()
    assert "shared" not in aliases["tools"] + aliases["docs"]


def test_a_folder_name_resolves_to_its_canonical_repository(registry):
    for spoken, canonical in (("agenticaiPersonal", "ai"), ("itsm-agent", "ai"),
                              ("AgenticBackend", "backend"), ("agenticFrontend", "frontend")):
        assert registry.resolve(spoken).name == canonical
    with pytest.raises(ValueError):
        registry.resolve("shared")


def read_capability(description, parameters, argument):
    return {
        "description": description, "effect": "read", "resource_type": "fixture", "operation_kind": "read",
        "permission": "fixture.read", "risk": "read", "requires_approval": False,
        "required_arguments": [argument], "grounded_arguments": [argument], "parameters": parameters,
    }


def repository_tool(registry):
    def values():
        return {"repository": {name: registry.aliases()[name] for name in registry.available()}}
    return {
        **read_capability("Inspect one repository.", {"repository": {"type": "str"}}, "repository"),
        "argument_values_resolver": values,
    }


def ticket_tool():
    return read_capability("Read one ticket.", {"ticket_key": {"type": "str", "normalize": "uppercase"}},
                           "ticket_key")


def intent_for(tool_name, tool, arguments, request):
    agent = AgentDefinition(name="fixture-specialist", description="Fixture.", model="fixture", tools=[tool_name])
    return parse_semantic_intent(
        {"summary": "Fixture read.", "effect": "read", "allowed_tools": [tool_name], "forbidden_tools": [],
         "allowed_arguments": arguments, "forbidden_arguments": {}, "max_tool_calls": 1,
         "clarification_required": False},
        agent=agent, tool_lookup=lambda name: tool if name == tool_name else None, user_request=request,
    )


def test_a_repository_named_by_its_folder_binds_the_canonical_repository(registry):
    tool = repository_tool(registry)
    intent = intent_for("inspect_repository", tool, {"repository": ["agenticaiPersonal"]},
                        "check item in agenticaiPersonal repository please")
    assert intent.allowed_arguments == {"repository": ["ai"]}
    # The model-facing choices stay the canonical names only.
    assert trusted_argument_aliases("inspect_repository", tool)["repository"]["agenticaipersonal"] == "ai"


def test_the_canonical_name_binds_when_the_user_typed_its_folder_name(registry):
    intent = intent_for("inspect_repository", repository_tool(registry), {"repository": ["ai"]},
                        "check item in agenticaiPersonal repository please")
    assert intent.allowed_arguments == {"repository": ["ai"]}


def test_a_canonical_name_the_user_never_named_is_still_dropped(registry):
    intent = intent_for("inspect_repository", repository_tool(registry), {"repository": ["backend"]},
                        "check item in agenticaiPersonal repository please")
    assert intent.allowed_arguments == {}
    assert intent.clarification_required is True


def test_an_unconfigured_folder_name_is_still_refused(registry):
    with pytest.raises(ValueError, match="trusted bounded value"):
        intent_for("inspect_repository", repository_tool(registry), {"repository": ["platform-kit"]},
                   "check platform-kit repository")


def test_execution_grounding_accepts_the_folder_name_the_user_typed(registry):
    tool = repository_tool(registry)
    aliases = trusted_argument_aliases("inspect_repository", tool)
    request = "check item in agenticaiPersonal repository please"
    assert validate_grounded_arguments(request, {"repository": "ai"}, ["repository"], aliases=aliases) == (True, None)
    # Another configured repository the user did not name is not grounded.
    valid, _ = validate_grounded_arguments(request, {"repository": "backend"}, ["repository"], aliases=aliases)
    assert valid is False


def test_jira_keys_in_any_case_bind_upper_case_keys():
    for spoken, hub_value in (("sla of sup-11", "SUP-11"), ("sla of sup-11", "sup-11"), ("sla of SUP-11", "SUP-11")):
        intent = intent_for("read_ticket", ticket_tool(), {"ticket_key": [hub_value]}, spoken)
        assert intent.allowed_arguments == {"ticket_key": ["SUP-11"]}


def test_a_key_the_user_never_typed_is_still_dropped():
    intent = intent_for("read_ticket", ticket_tool(), {"ticket_key": ["SUP-12"]}, "sla of sup-11")
    assert intent.allowed_arguments == {}
    assert intent.clarification_required is True


def test_proposed_keys_are_upper_cased_before_any_check_or_provider_call():
    assert normalize_argument_values(ticket_tool(), {"ticket_key": "sup-11", "other": "keep"}) == {
        "ticket_key": "SUP-11", "other": "keep"}


def test_every_registered_jira_key_argument_is_upper_cased():
    declared = {
        (tool_name, argument_name)
        for tool_name, tool in TOOLS.items()
        for argument_name, parameter in (tool.get("parameters") or {}).items()
        if argument_name in JIRA_KEY_ARGUMENTS
    }
    assert declared
    missing = sorted(pair for pair in declared
                     if TOOLS[pair[0]]["parameters"][pair[1]].get("normalize") != "uppercase")
    assert missing == []
