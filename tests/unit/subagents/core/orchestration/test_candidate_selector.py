"""Candidate retrieval reduces disclosure, never the trusted authority registry."""
from dataclasses import replace
from pathlib import Path

import pytest

from services.extensions.budget import estimate_text_tokens
from subagents.core.definitions.loader import load_agent_directory
from subagents.core.definitions.registry import AgentRegistry
from subagents.core.definitions.types import AgentDefinition
from subagents.core.orchestration import candidate_selector as module
from subagents.core.orchestration.candidate_selector import (
    Candidate,
    CandidateSet,
    CandidateValidationError,
    DeterministicCandidateSelector,
    build_candidate_agent_specs,
    validate_candidate_set,
)
from subagents.core.orchestration.router_catalog import render_router_catalog
from tools.capability_contract import required_arguments_for
from tools.registry import get_tool


@pytest.fixture
def production_registry():
    registry = AgentRegistry()
    registry.register_many(load_agent_directory(Path(__file__).resolve().parents[5] / "subagents/agents"))
    return registry


@pytest.mark.parametrize("user_request,expected", [
    ("Show Jira projects", "jira_project_list"),
    ("Check account status for alice", "account_status"),
    ("Get Jira ticket OPS-12", "ticket_get"),
    ("Create Jira ticket in OPS", "ticket_create"),
    ("Check OpenWrt firmware", "network_system_info"),
    ("List jira projects and account status", "account_status"),
    ("Show git status for /tmp/demo", "workspace_git_status"),
    ("Show git log for /tmp/demo", "workspace_git_log"),
    ("Execute a native process", "process_exec"),
])
def test_tiny_production_requests_have_small_trusted_packets(production_registry, user_request, expected):
    selection = DeterministicCandidateSelector().select(user_request, production_registry)
    assert not selection.fallback_required
    assert expected in {name for candidate in selection.candidates for name in candidate.capability_names}
    assert selection.capability_count <= 12
    specs = build_candidate_agent_specs(selection, production_registry)
    assert estimate_text_tokens(render_router_catalog(specs)) <= 2500
    for spec in specs:
        for descriptor in spec["capabilities"]:
            assert descriptor["name"] in production_registry.get(spec["name"]).tools
            tool = get_tool(descriptor["name"])
            assert descriptor["risk"] == tool["risk"]
            for field in ("grounded_arguments", "derived_arguments", "condition_fields"):
                assert descriptor["intent_metadata"][field] == tool.get(field, [])
            assert descriptor["intent_metadata"]["required_arguments"] == required_arguments_for(descriptor["name"], tool)
            assert descriptor["intent_metadata"]["requires_approval"] == tool["requires_approval"]


def test_shared_capabilities_keep_all_owners(production_registry):
    selection = DeterministicCandidateSelector().select("Get Jira ticket OPS-12", production_registry)
    expected = {agent.name for agent in production_registry.list_agents() if "ticket_get" in agent.tools}
    actual = {candidate.agent_name for candidate in selection.candidates if "ticket_get" in candidate.capability_names}
    assert actual == expected


@pytest.mark.parametrize("user_request", [
    "Is alice disabled?",
    "Is bob unlocked?",
    "Is charlie enabled?",
    "Is dana locked?",
    "Is employee47 locked or disabled?",
    "Check whether blair is disabled",
])
def test_account_security_state_language_discloses_the_read_capability(
        production_registry, user_request):
    selection = DeterministicCandidateSelector().select(user_request, production_registry)
    disclosed = {name for candidate in selection.candidates for name in candidate.capability_names}
    # These are retrieval hints, not a plan: ambiguity may still require broad
    # routing. A mutation whose description mentions a disabled/locked account
    # must not crowd the actual status read out of the candidate packet.
    assert "account_status" in disclosed
    tool = get_tool("account_status")
    assert tool["effect"] == "read"
    assert tool["requires_approval"] is False
    assert tool["grounded_arguments"] == ["user_id"]


def test_account_state_alternatives_keep_unresolved_clause_fallback(production_registry):
    selection = DeterministicCandidateSelector().select(
        "Is employee47 locked or disabled?", production_registry,
    )
    # Disclosure of the right tool does not erase the selector's unresolved
    # clause evidence or turn the candidate into an executable route.
    assert selection.fallback_required
    assert selection.reason == "uncovered_request"


def test_registry_order_does_not_change_ranking(production_registry):
    reverse_registry = AgentRegistry()
    reverse_registry.register_many(list(reversed(production_registry.list_agents())))
    selector = DeterministicCandidateSelector()
    request = "Create Jira ticket in OPS"
    expected = selector.select(request, production_registry)
    assert selector.select(request, reverse_registry) == expected
    assert selector.select(request, production_registry) == expected


@pytest.mark.parametrize("user_request,expected", [
    ("Create Jira ticket in OPS", {"ticket_create"}),
    ("Get Jira ticket OPS-12", {"ticket_get"}),
    ("List jira projects and account status", {"jira_project_list", "account_status"}),
])
def test_unrelated_catalog_growth_preserves_small_supported_requests(production_registry, monkeypatch, user_request, expected):
    original_get_tool = module.get_tool
    extra_tools = {
        "canopy_register": {"description": "Create one local orchard canopy record.", "operation_kind": "create",
                            "effect": "mutation", "requires_approval": True},
        "canopy_status": {"description": "Read the presence of a local orchard canopy record.", "operation_kind": "read",
                          "effect": "read", "requires_approval": False},
    }
    for tool in extra_tools.values():
        tool.update({"resource_type": "orchard.canopy", "risk": "read", "parameters": {}, "required_arguments": []})
    production_registry.register(AgentDefinition("canopy-helper", "Handles local orchard canopy records.", "local", list(extra_tools)))
    monkeypatch.setattr(module, "get_tool", lambda name: extra_tools.get(name) or original_get_tool(name))
    selection = DeterministicCandidateSelector().select(user_request, production_registry)
    assert not selection.fallback_required
    assert expected <= {name for candidate in selection.candidates for name in candidate.capability_names}
    assert estimate_text_tokens(render_router_catalog(build_candidate_agent_specs(selection, production_registry))) <= 2500


@pytest.mark.parametrize("user_request", ["", "hello", "quux frobnication", "' OR * NEAR( )", "👋"])
def test_no_lexical_evidence_requests_local_fallback(production_registry, user_request):
    selection = DeterministicCandidateSelector().select(user_request, production_registry)
    assert selection.fallback_required
    assert selection.candidates == ()


def test_low_confidence_widens_and_falls_back(production_registry):
    selection = DeterministicCandidateSelector(top_k=1, widen_k=3).select("project", production_registry)
    assert selection.fallback_required
    assert selection.confidence < 0.6
    assert selection.widening_count == 1
    assert selection.capability_count == 3


def test_query_limit_does_not_silently_drop_later_requested_operations(production_registry):
    selection = DeterministicCandidateSelector(max_query_terms=2).select("jira project account status", production_registry)
    assert selection.fallback_required
    assert selection.reason == "query_limit"
    assert not selection.candidates


@pytest.mark.parametrize("user_request,expected", [
    ("List Jira projects and account status", {"jira_project_list", "account_status"}),
    ("Show git status and execute native process", {"workspace_git_status", "process_exec"}),
])
def test_independent_operations_are_disclosed_or_require_local_fallback(production_registry, user_request, expected):
    selection = DeterministicCandidateSelector().select(user_request, production_registry)
    disclosed = {name for candidate in selection.candidates for name in candidate.capability_names}
    # Retrieval may prefer one task so strongly that the other falls outside the
    # compact packet. It must then request broad local routing, never silently
    # present the incomplete subset as the whole request.
    assert expected <= disclosed or selection.fallback_required


def test_request_coverage_widens_before_falling_back(production_registry):
    selection = DeterministicCandidateSelector().select("List jira projects and account status", production_registry)
    disclosed = {name for candidate in selection.candidates for name in candidate.capability_names}
    assert not selection.fallback_required
    assert {"jira_project_list", "account_status"} <= disclosed
    assert selection.capability_count <= 12
    assert estimate_text_tokens(render_router_catalog(build_candidate_agent_specs(selection, production_registry))) <= 2500


@pytest.mark.parametrize("user_request", [
    "Inspect the current developer workspace project and report its project type and the governed developer operations available.",
    "Inspect the workspace project and report its project type and available governed developer operations.",
])
def test_description_backed_output_clauses_do_not_force_local_fallback(production_registry, user_request):
    selection = DeterministicCandidateSelector().select(user_request, production_registry)
    assert not selection.fallback_required
    assert "workspace_project_info" in {
        name for candidate in selection.candidates for name in candidate.capability_names
    }
    assert estimate_text_tokens(render_router_catalog(build_candidate_agent_specs(selection, production_registry))) <= 2500


@pytest.mark.parametrize("literal", ["ls", "unregistered_executable"])
def test_unknown_exact_code_literal_cannot_be_replaced_by_generic_prose(production_registry, literal):
    selection = DeterministicCandidateSelector().select(f"Run the exact bounded command `{literal}`", production_registry)
    assert selection.fallback_required


def test_catalog_known_code_literal_remains_retrievable(production_registry):
    selection = DeterministicCandidateSelector().select("Show `git` status", production_registry)
    assert not selection.fallback_required
    assert "workspace_git_status" in {
        name for candidate in selection.candidates for name in candidate.capability_names
    }


@pytest.mark.parametrize("user_request", [
    "List Jira projects and frobnicate quux",
    "List Jira projects and frobnicate",
    "List Jira projects; frobnicate",
    "List Jira projects then frobnicate",
    "Show Git branches and run a shell command",
])
def test_strong_first_task_does_not_hide_unresolved_later_task(production_registry, user_request):
    selection = DeterministicCandidateSelector().select(user_request, production_registry)
    assert selection.fallback_required
    assert selection.reason == "uncovered_request"


def test_oversized_disclosure_falls_back_without_dropping_candidates(production_registry):
    request = "Create Jira ticket in OPS"
    normal = DeterministicCandidateSelector().select(request, production_registry)
    bounded = DeterministicCandidateSelector(max_packet_tokens=100).select(request, production_registry)
    assert not normal.fallback_required
    assert bounded.fallback_required
    assert bounded.reason == "packet_limit"
    assert bounded.candidates == normal.candidates
    assert bounded.widening_count == normal.widening_count


@pytest.mark.parametrize("candidate", [
    Candidate("invented-specialist", ("ticket_get",)),
    Candidate("jira-specialist", ("invented-tool",)),
    Candidate("jira-specialist", ("account_status",)),
    Candidate("jira-specialist", ("ticket_get", "ticket_get")),
    Candidate("jira-specialist", ()),
    Candidate(" jira-specialist", ("ticket_get",)),
])
def test_unknown_or_unowned_selector_output_is_rejected_before_disclosure(production_registry, candidate):
    selection = CandidateSet((candidate,), confidence=1.0)
    with pytest.raises(CandidateValidationError):
        build_candidate_agent_specs(selection, production_registry)


def test_mixed_valid_and_invalid_output_rejects_whole_selection(production_registry):
    selection = CandidateSet((
        Candidate("jira-specialist", ("ticket_get",)),
        Candidate("account-specialist", ("ticket_get",)),
    ), confidence=1.0)
    with pytest.raises(CandidateValidationError, match="unowned"):
        validate_candidate_set(selection, production_registry)


@pytest.mark.parametrize("overrides", [
    {"confidence": float("nan")}, {"confidence": float("inf")}, {"confidence": 1.1},
    {"confidence": True}, {"widening_count": -1}, {"widening_count": True},
    {"fallback_required": "false"}, {"candidates": []},
])
def test_untrusted_candidate_accounting_is_validated(production_registry, overrides):
    with pytest.raises(CandidateValidationError):
        validate_candidate_set(replace(CandidateSet(), **overrides), production_registry)


def test_disclosure_rechecks_registry_after_selection(production_registry):
    selection = CandidateSet((Candidate("jira-specialist", ("ticket_get",)),))
    production_registry.get("jira-specialist").tools.remove("ticket_get")
    with pytest.raises(CandidateValidationError):
        build_candidate_agent_specs(selection, production_registry)


def test_compact_packet_never_resolves_or_discloses_bounded_values(production_registry, monkeypatch):
    tool = get_tool("ticket_get")
    def forbidden_resolver():
        raise AssertionError("Candidate disclosure must not resolve runtime argument values")
    monkeypatch.setitem(tool, "argument_values_resolver", forbidden_resolver)
    monkeypatch.setitem(tool, "examples", ["PRIVATE_EXAMPLE" * 1000])
    monkeypatch.setitem(tool, "bounded_argument_values", {"ticket_id": ["PRIVATE_VALUE"]})
    selection = CandidateSet((Candidate("jira-specialist", ("ticket_get",)),))
    specs = build_candidate_agent_specs(selection, production_registry)
    packet = render_router_catalog(specs)
    assert "PRIVATE_EXAMPLE" not in packet
    assert "PRIVATE_VALUE" not in packet
    assert "bounded_argument_values" not in packet
    assert "argument_schema" not in packet
    assert "argument_values_resolver" not in packet
    metadata = specs[0]["capabilities"][0]["intent_metadata"]
    assert metadata["condition_fields"] == tool.get("condition_fields", [])
    assert metadata["grounded_argument_descriptions"]


def test_arbitrary_catalog_names_are_retrieved_without_domain_routes(monkeypatch):
    registry = AgentRegistry()
    registry.register(AgentDefinition("orchard-helper", "Looks after orchard harvests.", "local", ["inspect_pomelo"]))
    tool = {"description": "Inspect pomelo ripeness", "resource_type": "orchard.pomelo",
            "operation_kind": "read", "risk": "read", "effect": "read", "permission": "orchard.read", "requires_approval": False,
            "parameters": {}, "required_arguments": []}
    monkeypatch.setattr(module, "get_tool", lambda name: tool if name == "inspect_pomelo" else None)
    selection = DeterministicCandidateSelector().select("inspect pomelo", registry)
    assert not selection.fallback_required
    assert selection.candidates == (Candidate("orchard-helper", ("inspect_pomelo",)),)


def test_arbitrary_description_outputs_support_conjunctions_without_identity_tokens(monkeypatch):
    registry = AgentRegistry()
    registry.register(AgentDefinition("orchard-helper", "Looks after orchard harvests.", "local", ["inspect_pomelo"]))
    tool = {"description": "Inspect pomelo ripeness and report harvest quality and available orchard operations.",
            "resource_type": "orchard.pomelo", "operation_kind": "read", "risk": "read", "effect": "read",
            "permission": "orchard.read", "requires_approval": False, "parameters": {}, "required_arguments": []}
    monkeypatch.setattr(module, "get_tool", lambda name: tool if name == "inspect_pomelo" else None)
    selection = DeterministicCandidateSelector().select(
        "Inspect pomelo and report its harvest quality and available orchard operations", registry)
    assert not selection.fallback_required
    assert selection.candidates == (Candidate("orchard-helper", ("inspect_pomelo",)),)


def test_weaker_tail_ties_do_not_make_stronger_candidates_ambiguous(monkeypatch):
    registry = AgentRegistry()
    names = ["inspect_pomelo", *(f"observe_pomelo_{index}" for index in range(4))]
    registry.register(AgentDefinition("orchard-helper", "Looks after orchard harvests.", "local", names))
    tool = {"description": "Inspect pomelo ripeness", "resource_type": "orchard.pomelo", "operation_kind": "read",
            "risk": "read", "effect": "read", "permission": "orchard.read", "requires_approval": False,
            "parameters": {}, "required_arguments": []}
    monkeypatch.setattr(module, "get_tool", lambda name: tool if name in names else None)
    selection = DeterministicCandidateSelector(top_k=2, widen_k=3).select("inspect pomelo", registry)
    assert not selection.fallback_required
    assert "inspect_pomelo" in selection.candidates[0].capability_names
    assert selection.capability_count == 2
    assert selection.widening_count == 0


def test_unresolved_identical_capabilities_force_fallback(monkeypatch):
    registry = AgentRegistry()
    names = [f"opaque{index}" for index in range(4)]
    registry.register(AgentDefinition("helper", "Inspect pomelo ripeness.", "local", names))
    other_names = [f"other{index}" for index in range(5)]
    registry.register(AgentDefinition("other", "Unrelated tasks.", "local", other_names))
    tool = {"description": "Inspect pomelo ripeness", "resource_type": "orchard.pomelo", "operation_kind": "read"}
    monkeypatch.setattr(module, "get_tool", lambda name: tool if name in names else {"description": "Unrelated task"} if name in other_names else None)
    selection = DeterministicCandidateSelector(top_k=1, widen_k=2).select("inspect pomelo", registry)
    assert selection.fallback_required
    assert selection.widening_count == 1
    assert selection.reason == "ambiguous_boundary"
    assert selection.capability_count == 2


@pytest.mark.parametrize("kwargs", [{"top_k": 0}, {"widen_k": 0}, {"top_k": True}, {"top_k": 4, "widen_k": 2},
                                    {"max_packet_tokens": 0}, {"max_packet_tokens": True}])
def test_invalid_limits_are_rejected(kwargs):
    with pytest.raises(ValueError):
        DeterministicCandidateSelector(**kwargs)
