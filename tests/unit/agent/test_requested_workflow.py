"""Reviewed scope must survive model routing without gaining authority.

These tests use registered callable metadata only; they do not start transports,
instantiate deployment providers, generate model output, or execute ITSM tools.
"""
from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from agent.requested_workflow import (
    RequestedWorkflowContract,
    admit_requested_workflow,
    requested_workflow_prompt,
    validate_requested_plan,
    validate_requested_step,
    verify_workflow_binding,
)
from subagents.core.definitions.registry import AgentRegistry
from subagents.core.definitions.types import (
    AgentDefinition,
    ResultCondition,
    SemanticIntent,
    SpecialistRequest,
)


def step(capability="account_create", *, step_id="identity", arguments=None, after=None):
    return {
        "id": step_id,
        "capability": capability,
        "grounded_arguments": {} if arguments is None else arguments,
        "after_success_of": after,
    }


def account_ticket_payload():
    return {"version": 1, "steps": [
        step(arguments={"given_name": ["Mvp"], "family_name": ["Fixture"]}),
        step("ticket_create", step_id="ticket", arguments={"project_key": ["SUP"]}, after="identity"),
    ]}


def contract_for(*steps):
    return RequestedWorkflowContract.model_validate({"version": 1, "steps": list(steps)})


@pytest.fixture
def registry():
    from mcp_server import register_capability_signatures

    register_capability_signatures()
    result = AgentRegistry()
    for name, tools in (
        ("account-specialist", ["account_create", "enable_user", "reset_password"]),
        ("jira-specialist", ["ticket_create", "ticket_get", "jira_project_list"]),
        ("operations-specialist", ["notify_user", "service_health", "container_logs"]),
    ):
        result.register(AgentDefinition(name=name, description=name, model="fixture", tools=tools))
    return result


def delegations(contract, owners, carriers=None):
    from tools.capability_contract import build_capability_contract
    from tools.registry import get_tool

    carriers = carriers or {}
    result = []
    carried_by = {}
    for item in contract.steps:
        metadata = build_capability_contract(item.capability, get_tool(item.capability))
        agent = carriers.get(item.capability, owners[item.capability][0])
        result.append(SpecialistRequest(
            agent_name=agent,
            instructions="Perform the requested operation through normal policy and approval checks.",
            semantic_intent=SemanticIntent(
                summary="Requested operation",
                effect=metadata.effect,
                allowed_tools=[item.capability],
                allowed_arguments=deepcopy(item.grounded_arguments),
            ),
            after_success_of=carried_by.get(item.after_success_of),
        ))
        carried_by[item.id] = agent
    return result


@pytest.mark.parametrize("version", [True, False, "1", 1.0, 0, 2, None])
def test_version_is_exact_integer_one(version):
    with pytest.raises(ValidationError):
        RequestedWorkflowContract.model_validate({"version": version, "steps": [step()]})


@pytest.mark.parametrize("payload", [
    {},
    {"version": 1, "steps": []},
    {"version": 1, "steps": [step()], "owner": "invented-authority"},
    {"version": 1, "steps": [dict(step(), approved=True)]},
    {"version": 1, "steps": [dict(step(), id=True)]},
    {"version": 1, "steps": [dict(step(), id="1starts-with-a-number")]},
    {"version": 1, "steps": [dict(step(), capability="has whitespace")]},
    {"version": 1, "steps": [dict(step(), grounded_arguments={"invalid-key": ["value"]})]},
    {"version": 1, "steps": [dict(step(), grounded_arguments={"given_name": "value"})]},
    {"version": 1, "steps": [dict(step(), grounded_arguments={"given_name": [True]})]},
    {"version": 1, "steps": [dict(step(), grounded_arguments={"given_name": []})]},
    {"version": 1, "steps": [dict(step(), grounded_arguments={"given_name": [""]})]},
    {"version": 1, "steps": [dict(step(), after_success_of=False)]},
    {"version": 1, "steps": (step(),)},
])
def test_contract_rejects_extra_or_coerced_shapes(payload):
    with pytest.raises(ValidationError):
        RequestedWorkflowContract.model_validate(payload)


@pytest.mark.parametrize("missing", ["version", "steps"])
def test_contract_fields_are_required(missing):
    payload = account_ticket_payload()
    del payload[missing]
    with pytest.raises(ValidationError):
        RequestedWorkflowContract.model_validate(payload)


@pytest.mark.parametrize("missing", ["id", "capability", "grounded_arguments", "after_success_of"])
def test_step_fields_are_required_even_when_dependency_is_null(missing):
    payload = {"version": 1, "steps": [step()]}
    del payload["steps"][0][missing]
    with pytest.raises(ValidationError):
        RequestedWorkflowContract.model_validate(payload)


@pytest.mark.parametrize("changes", [
    {"id": "a" * 81},
    {"capability": "a" * 151},
    {"grounded_arguments": {"a" * 151: ["value"]}},
    {"grounded_arguments": {"given_name": ["a" * 1025]}},
    {"grounded_arguments": {"given_name": ["value"] * 17}},
    {"grounded_arguments": {f"key{index}": ["value"] for index in range(33)}},
])
def test_field_bounds_are_enforced(changes):
    with pytest.raises(ValidationError):
        contract_for(dict(step(), **changes))


def test_maximum_step_count_is_bounded_independently_of_admission():
    rows = [step(f"fixture_{index}", step_id=f"step{index}") for index in range(9)]
    assert len(contract_for(*rows[:8]).steps) == 8
    with pytest.raises(ValidationError):
        contract_for(*rows)


@pytest.mark.parametrize("rows", [
    [step(), step("ticket_create")],
    [step(), step("account_create", step_id="another")],
    [step(after="ticket"), step("ticket_create", step_id="ticket")],
    [step(after="missing")],
    [step(after="identity")],
])
def test_graph_rejects_duplicate_identity_capability_and_nonpreceding_dependencies(rows):
    with pytest.raises(ValidationError):
        contract_for(*rows)


def encoded_size(payload):
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def test_total_size_boundary_is_inclusive_and_not_a_character_limit():
    # Every individual field remains within its bound, but the combined body
    # must fit the declared UTF-8 byte budget.
    arguments = {"first": ["a" * 1024] * 16, "second": ["b" * 1024] * 16}
    payload = {"version": 1, "steps": [step(arguments=arguments)]}
    excess = encoded_size(payload) - 32768
    assert 0 < excess < 1024
    arguments["second"][-1] = arguments["second"][-1][:-excess]
    assert encoded_size(payload) == 32768
    assert contract_for(payload["steps"][0]).steps
    arguments["second"][-1] += "c"
    assert encoded_size(payload) == 32769
    with pytest.raises(ValidationError, match="size limit"):
        RequestedWorkflowContract.model_validate(payload)

    multibyte = {"version": 1, "steps": [step(arguments={"names": ["界" * 1024] * 11})]}
    assert len(json.dumps(multibyte, ensure_ascii=False, separators=(",", ":"))) < 32768
    assert encoded_size(multibyte) > 32768
    with pytest.raises(ValidationError, match="size limit"):
        RequestedWorkflowContract.model_validate(multibyte)


def unicode_vector():
    return {"version": 1, "steps": [
        step(arguments={"given_name": ["Zoë", "李"], "family_name": ["山田"]}),
        step("ticket_create", step_id="ticket", arguments={"project_key": ["SUP"]}, after="identity"),
    ]}


def test_canonical_hash_has_shared_unicode_vector_and_ignores_object_key_order():
    payload = unicode_vector()
    expected = "37d95a5bf9eb19950cbd72bb05bc960aa1a283dc54d6783db73226054aa8f5cf"
    assert RequestedWorkflowContract.model_validate(payload).content_hash() == expected
    first = payload["steps"][0]
    first["grounded_arguments"] = dict(reversed(list(first["grounded_arguments"].items())))
    payload["steps"][0] = dict(reversed(list(first.items())))
    assert RequestedWorkflowContract.model_validate(dict(reversed(list(payload.items())))).content_hash() == expected


@pytest.mark.parametrize("change", ["target_order", "target_text", "step_order", "dependency"])
def test_canonical_hash_changes_when_reviewed_meaning_changes(change):
    payload = unicode_vector()
    original = RequestedWorkflowContract.model_validate(payload).content_hash()
    if change == "target_order":
        payload["steps"][0]["grounded_arguments"]["given_name"].reverse()
    elif change == "target_text":
        payload["steps"][0]["grounded_arguments"]["given_name"][0] = "Zoe"
    elif change == "step_order":
        payload["steps"][1]["after_success_of"] = None
        payload["steps"].reverse()
    else:
        payload["steps"][1]["after_success_of"] = None
    assert RequestedWorkflowContract.model_validate(payload).content_hash() != original


def test_binding_accepts_only_absent_pair_or_exact_contract_and_hash():
    contract = contract_for(step())
    verify_workflow_binding(None, None)
    verify_workflow_binding(contract, contract.content_hash())


@pytest.mark.parametrize("case", ["contract_only", "hash_only", "wrong_hash", "uppercase", "dict", "bool"])
def test_binding_rejects_missing_wrong_or_unparsed_pair(case):
    contract = contract_for(step())
    value, digest = contract, contract.content_hash()
    if case == "contract_only":
        digest = None
    elif case == "hash_only":
        value = None
    elif case == "wrong_hash":
        digest = "0" * 64
    elif case == "uppercase":
        digest = digest.upper()
    elif case == "dict":
        value = contract.model_dump()
    else:
        digest = True
    with pytest.raises(ValueError, match="durable hash"):
        verify_workflow_binding(value, digest)


def test_admission_uses_actual_metadata_for_single_and_supported_success_pair(registry):
    single = contract_for(step("service_health", step_id="health"))
    assert admit_requested_workflow(single, registry) == {"service_health": ("operations-specialist",)}
    pair = RequestedWorkflowContract.model_validate(account_ticket_payload())
    assert admit_requested_workflow(pair, registry) == {
        "account_create": ("account-specialist",), "ticket_create": ("jira-specialist",),
    }


def test_real_specialist_definitions_admit_the_account_ticket_workflow():
    # Deployed definitions share ticket capabilities between a provider-neutral
    # and a provider-specific specialist; the core workflow must stay admissible.
    from pathlib import Path

    from mcp_server import register_capability_signatures
    from subagents.core.definitions.loader import load_agent_directory

    register_capability_signatures()
    deployed = AgentRegistry()
    deployed.register_many(load_agent_directory(Path(__file__).resolve().parents[3] / "subagents/agents"))
    owners = admit_requested_workflow(RequestedWorkflowContract.model_validate(account_ticket_payload()), deployed)
    assert owners["account_create"] == ("account-specialist",)
    assert "jira-specialist" in owners["ticket_create"]


@pytest.mark.parametrize("capability", ["unregistered_operation", "account_status"])
def test_admission_rejects_unknown_tools_and_registered_tools_without_owner(registry, capability):
    with pytest.raises(ValueError, match="one registered specialist owner"):
        admit_requested_workflow(contract_for(step(capability)), registry)


def test_shared_capability_admits_every_registered_owner_and_no_other(registry):
    registry.register(AgentDefinition(name="neutral-tickets", description="fixture", model="fixture",
                                      tools=["ticket_create"]))
    contract = RequestedWorkflowContract.model_validate(account_ticket_payload())
    owners = admit_requested_workflow(contract, registry)
    assert owners["ticket_create"] == ("jira-specialist", "neutral-tickets")
    for carrier in owners["ticket_create"]:
        validate_requested_plan(contract, delegations(contract, owners, {"ticket_create": carrier}), owners)
    stranger = delegations(contract, owners, {"ticket_create": "operations-specialist"})
    with pytest.raises(ValueError, match="differs from the reviewed workflow step"):
        validate_requested_plan(contract, stranger, owners)


def test_admission_rejects_steps_that_could_land_on_one_specialist(registry):
    registry.register(AgentDefinition(name="generalist", description="fixture", model="fixture",
                                      tools=["account_create", "service_health"]))
    contract = contract_for(step(), step("service_health", step_id="health"))
    with pytest.raises(ValueError, match="multiple operations from one specialist"):
        admit_requested_workflow(contract, registry)


def test_dependency_names_the_specialist_that_carries_the_source_step(registry):
    registry.register(AgentDefinition(name="directory-alt", description="fixture", model="fixture",
                                      tools=["account_create"]))
    contract = RequestedWorkflowContract.model_validate(account_ticket_payload())
    owners = admit_requested_workflow(contract, registry)
    proposals = delegations(contract, owners, {"account_create": "account-specialist"})
    proposals[1].after_success_of = "directory-alt"
    # Each step alone is plausible; only the whole plan shows the dependency
    # names a specialist other than the one performing the account step.
    validate_requested_step(contract, 1, proposals[1], owners)
    with pytest.raises(ValueError, match="dependency"):
        validate_requested_plan(contract, proposals, owners)


def test_admission_rejects_multiple_operations_for_one_specialist(registry):
    contract = contract_for(step(), step("enable_user", step_id="enable", arguments={"user_id": ["fixture"]}))
    with pytest.raises(ValueError, match="multiple operations from one specialist"):
        admit_requested_workflow(contract, registry)


@pytest.mark.parametrize(("capability", "argument"), [
    ("account_create", "expected_username"),
    ("ticket_create", "expected_project_id"),
    ("ticket_create", "summary"),
    ("container_logs", "lines"),
    ("service_health", "invented"),
])
def test_admission_rejects_policy_derived_or_undeclared_fields(registry, capability, argument):
    with pytest.raises(ValueError, match="trusted grounded argument fields"):
        admit_requested_workflow(contract_for(step(capability, arguments={argument: ["fixture"]})), registry)


def test_admission_uses_successor_metadata_instead_of_accepting_arbitrary_dependencies(registry):
    contract = contract_for(
        step("service_health", step_id="health"),
        step("ticket_create", step_id="ticket", after="health"),
    )
    with pytest.raises(ValueError, match="not declared by the capability"):
        admit_requested_workflow(contract, registry)


def test_admission_rejects_larger_graph_even_when_its_one_pair_is_supported(registry):
    payload = account_ticket_payload()
    payload["steps"].append(step("service_health", step_id="health"))
    with pytest.raises(ValueError, match="durable continuation engine"):
        admit_requested_workflow(RequestedWorkflowContract.model_validate(payload), registry)


@pytest.mark.parametrize("pair", [False, True])
def test_complete_matching_plan_is_validated_without_mutating_proposal(registry, pair):
    contract = (RequestedWorkflowContract.model_validate(account_ticket_payload()) if pair
                else contract_for(step("service_health", step_id="health")))
    owners = admit_requested_workflow(contract, registry)
    proposals = delegations(contract, owners)
    before = deepcopy(proposals)
    validate_requested_plan(contract, proposals, owners)
    assert proposals == before


@pytest.mark.parametrize("change", ["missing", "extra", "wrong_order"])
def test_whole_plan_rejects_missing_extra_and_reordered_operations(registry, change):
    contract = RequestedWorkflowContract.model_validate(account_ticket_payload())
    owners = admit_requested_workflow(contract, registry)
    proposals = delegations(contract, owners)
    if change == "missing":
        proposals.pop()
    elif change == "extra":
        proposals.append(deepcopy(proposals[-1]))
    else:
        proposals.reverse()
    with pytest.raises(ValueError):
        validate_requested_plan(contract, proposals, owners)


@pytest.mark.parametrize("change", [
    "agent", "tool", "extra_tool", "effect", "no_intent", "clarification", "calls",
    "condition", "dependency", "target", "extra_target", "missing_target", "forbidden_target",
])
def test_whole_plan_rejects_changed_operation_targets_or_success_condition(registry, change):
    contract = RequestedWorkflowContract.model_validate(account_ticket_payload())
    owners = admit_requested_workflow(contract, registry)
    proposals = delegations(contract, owners)
    first = proposals[0]
    intent = first.semantic_intent
    if change == "agent":
        first.agent_name = "jira-specialist"
    elif change == "tool":
        intent.allowed_tools = ["enable_user"]
    elif change == "extra_tool":
        intent.allowed_tools.append("enable_user")
    elif change == "effect":
        intent.effect = "read"
    elif change == "no_intent":
        first.semantic_intent = None
    elif change == "clarification":
        intent.clarification_required = True
    elif change == "calls":
        intent.max_tool_calls = 2
    elif change == "condition":
        proposals[1].condition = ResultCondition("account-specialist", "ok", True)
    elif change == "dependency":
        proposals[1].after_success_of = None
    elif change == "target":
        intent.allowed_arguments["given_name"] = ["Different"]
    elif change == "extra_target":
        intent.allowed_arguments["given_name"].append("Another")
    elif change == "missing_target":
        del intent.allowed_arguments["given_name"]
    else:
        intent.forbidden_arguments["given_name"] = ["Mvp"]
    with pytest.raises(ValueError):
        validate_requested_plan(contract, proposals, owners)


def test_dependency_names_the_reviewed_source_specialist_not_its_step_id(registry):
    contract = RequestedWorkflowContract.model_validate(account_ticket_payload())
    owners = admit_requested_workflow(contract, registry)
    proposals = delegations(contract, owners)
    proposals[1].after_success_of = "identity"
    with pytest.raises(ValueError, match="dependency"):
        validate_requested_plan(contract, proposals, owners)


def test_optional_target_pins_do_not_become_an_authorization_allowlist(registry):
    contract = contract_for(step(arguments={"given_name": ["Mvp"]}))
    owners = admit_requested_workflow(contract, registry)
    proposals = delegations(contract, owners)
    # Fields without explicit pins are still interpreted by the existing
    # semantic grounding and gateway checks; scope validation alone cannot
    # establish that these invented values are authorized or executable.
    proposals[0].semantic_intent.allowed_arguments["family_name"] = ["Interpreted elsewhere"]
    validate_requested_plan(contract, proposals, owners)


def test_step_validation_preserves_exact_pin_value_order(registry):
    contract = contract_for(step(arguments={"given_name": ["Zoë", "李"]}))
    owners = admit_requested_workflow(contract, registry)
    proposal = delegations(contract, owners)[0]
    proposal.semantic_intent.allowed_arguments["given_name"].reverse()
    with pytest.raises(ValueError, match="targets"):
        validate_requested_step(contract, 0, proposal, owners)


def test_prompt_contains_exact_reviewed_scope_and_does_not_grant_approval():
    contract = RequestedWorkflowContract.model_validate(unicode_vector())
    prompt = requested_workflow_prompt(contract)
    assert json.loads(prompt[prompt.index("{\"version\""):]) == contract.model_dump()
    assert "grants no permission or approval" in prompt
    assert "individual approvals still apply" in prompt
