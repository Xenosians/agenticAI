"""A reviewed workflow cannot dispatch a plausible but incomplete model plan."""
import asyncio
from copy import deepcopy
import json

import pytest

from agent.requested_workflow import RequestedWorkflowContract
from services.extensions.context import execution_profile
from subagents.core.definitions.registry import AgentRegistry
from subagents.core.definitions.types import AgentDefinition, AgentResult, SemanticIntent, SpecialistRequest
from subagents.core.orchestration.orchestrator import Orchestrator
from subagents.core.orchestration.router import LLMRouter
from subagents.llm.runtime.stages import InferenceStage


REQUEST = "Create account given name Mvp family name Test, then after success create a Jira ticket in SUP for this account."


@pytest.fixture
def registry():
    value = AgentRegistry()
    for agent, tools in (("account-specialist", ["account_create", "account_status"]),
                         ("jira-specialist", ["ticket_create", "ticket_get"])):
        value.register(AgentDefinition(name=agent, description=agent, model="isolated", tools=tools))
    return value


def contract():
    return RequestedWorkflowContract.model_validate({"version": 1, "steps": [
        {"id": "identity", "capability": "account_create", "grounded_arguments": {
            "given_name": ["Mvp"], "family_name": ["Test"],
        }, "after_success_of": None},
        {"id": "issue", "capability": "ticket_create", "grounded_arguments": {
            "project_key": ["SUP"],
        }, "after_success_of": "identity"},
    ]})


def plan():
    rows = []
    for agent, capability, targets in (
        ("account-specialist", "account_create", {"given_name": ["Mvp"], "family_name": ["Test"]}),
        ("jira-specialist", "ticket_create", {"project_key": ["SUP"]}),
    ):
        rows.append({"agent": agent, "instructions": "Perform the requested operation.", "intent": {
            "summary": "Perform the requested operation.", "effect": "mutation",
            "allowed_tools": [capability], "allowed_arguments": targets,
            "forbidden_tools": [], "forbidden_arguments": {},
            "max_tool_calls": 1, "clarification_required": False,
        }})
    rows[1]["after_success_of"] = "account-specialist"
    return {"delegations": rows}


class Inference:
    def __init__(self, responses):
        self.responses, self.calls = responses, []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        assert len(self.calls) <= len(self.responses), "Repair budget exceeded"
        return json.dumps(self.responses[len(self.calls) - 1])


class Runtime:
    def __init__(self, registry):
        self.agent_registry, self.tasks = registry, []

    async def run(self, task):
        self.tasks.append(task)
        return AgentResult(task_id=task.task_id, agent_name=task.agent_name,
                           status="approval_required", proposed_tool="account_create",
                           approval_id="isolated-exact-proposal")


class Primary:
    async def respond(self, *args, **kwargs):
        pytest.fail("An empty reviewed workflow must not become a successful chat answer.")


@pytest.mark.parametrize("external", [False, True])
def test_missing_step_uses_one_shared_repair_and_only_then_proposes_first_approval(registry, external):
    complete = plan()
    incomplete = {"delegations": complete["delegations"][:1]}
    inference = Inference([incomplete, complete])
    router = LLMRouter(registry, inference, "isolated", strict_contract=True)
    runtime = Runtime(registry)
    hub = Orchestrator(router, runtime, Primary(), require_semantic_intent=True)
    token = execution_profile.set({"backend": "http", "model": "isolated"} if external else None)
    try:
        result = asyncio.run(hub.run(REQUEST, workflow_contract=contract()))
    finally:
        execution_profile.reset(token)
    assert result.status == "approval_required"
    assert len(runtime.tasks) == 1
    assert runtime.tasks[0].semantic_intent.allowed_tools == ["account_create"]
    assert getattr(result, "_pending_onboarding")["ticket"]["semantic_intent"]["allowed_tools"] == ["ticket_create"]
    assert len(inference.calls) == 2
    expected_stage = InferenceStage.SEMANTIC_ROUTING if external else InferenceStage.CONTROL_ROUTING
    assert inference.calls[0]["stage"] is expected_stage
    assert inference.calls[1]["stage"] is InferenceStage.REPAIR
    for call in inference.calls:
        assert call["messages"][-1] == {"role": "user", "content": REQUEST}
        assert "USER-REVIEWED REQUESTED WORKFLOW" in call["messages"][0]["content"]
        assert "account_status" not in call["messages"][0]["content"]


@pytest.mark.parametrize("kind", ["missing", "empty", "dependency", "extra_value"])
@pytest.mark.parametrize("external", [False, True])
def test_repeated_mismatch_never_dispatches_any_step(registry, kind, external):
    bad = deepcopy(plan())
    if kind == "missing":
        bad["delegations"].pop()
    elif kind == "empty":
        bad["delegations"] = []
    elif kind == "dependency":
        bad["delegations"][1].pop("after_success_of")
    else:
        bad["delegations"][0]["intent"]["allowed_arguments"]["given_name"].append("Other")
    inference = Inference([bad, bad])
    runtime = Runtime(registry)
    hub = Orchestrator(LLMRouter(registry, inference, "isolated", strict_contract=True),
                       runtime, Primary(), require_semantic_intent=True)
    token = execution_profile.set({"backend": "http", "model": "isolated"} if external else None)
    try:
        # Both values must survive existing literal-origin narrowing so this
        # case exercises the reviewed pin, rather than invented-value removal.
        request = REQUEST + " Reference name Other belongs to a different person." if kind == "extra_value" else REQUEST
        result = asyncio.run(hub.run(request, workflow_contract=contract()))
    finally:
        execution_profile.reset(token)
    assert result.status == "error"
    assert result.routes == [] and result.results == []
    assert "No action was executed" in result.answer
    assert runtime.tasks == []
    assert len(inference.calls) == 2


def test_custom_router_cannot_bypass_completeness_check(registry):
    class Router:
        async def route(self, *args, **kwargs):
            return [SpecialistRequest(
                agent_name="account-specialist", instructions="Create requested account.",
                semantic_intent=SemanticIntent(summary="Create requested account.", effect="mutation",
                    allowed_tools=["account_create"], allowed_arguments={
                        "given_name": ["Mvp"], "family_name": ["Test"],
                    }),
            )]
    runtime = Runtime(registry)
    result = asyncio.run(Orchestrator(Router(), runtime, Primary()).run(REQUEST, workflow_contract=contract()))
    assert result.status == "error"
    assert not runtime.tasks


def test_legacy_router_cannot_silently_drop_reviewed_contract(registry):
    class Legacy:
        async def route(self, text):
            pytest.fail("A contract-ignorant router must fail before generation.")
    runtime = Runtime(registry)
    result = asyncio.run(Orchestrator(Legacy(), runtime, Primary()).run(REQUEST, workflow_contract=contract()))
    assert result.status == "error"
    assert not runtime.tasks


def test_without_contract_legitimate_account_only_request_remains_compatible(registry):
    inference = Inference([{"delegations": plan()["delegations"][:1]}])
    runtime = Runtime(registry)
    hub = Orchestrator(LLMRouter(registry, inference, "isolated", strict_contract=True), runtime, Primary())
    result = asyncio.run(hub.run("Create account given name Mvp family name Test."))
    assert result.status == "approval_required"
    assert len(runtime.tasks) == 1 and len(inference.calls) == 1


@pytest.mark.parametrize("carrier", ["jira-specialist", "ticket-specialist"])
def test_shared_ticket_capability_routes_through_either_registered_owner(registry, carrier):
    # Deployed definitions register ticket_create for the provider-specific and
    # the provider-neutral specialist; a reviewed workflow must stay usable.
    registry.register(AgentDefinition(name="ticket-specialist", description="ticket-specialist",
                                      model="isolated", tools=["ticket_create", "ticket_get"]))
    complete = plan()
    complete["delegations"][1]["agent"] = carrier
    inference = Inference([complete])
    runtime = Runtime(registry)
    hub = Orchestrator(LLMRouter(registry, inference, "isolated", strict_contract=True),
                       runtime, Primary(), require_semantic_intent=True)
    result = asyncio.run(hub.run(REQUEST, workflow_contract=contract()))
    assert result.status == "approval_required"
    assert [task.semantic_intent.allowed_tools for task in runtime.tasks] == [["account_create"]]
    pending = getattr(result, "_pending_onboarding")["ticket"]
    assert pending["agent_name"] == carrier
    assert pending["semantic_intent"]["allowed_tools"] == ["ticket_create"]
    assert len(inference.calls) == 1
    disclosed = inference.calls[0]["messages"][0]["content"]
    assert "ticket-specialist" in disclosed and "jira-specialist" in disclosed


def test_caller_cannot_change_nested_reviewed_scope_during_routing(registry):
    original = contract()

    class Router:
        async def route(self, request, *, workflow_contract, **kwargs):
            original.steps.pop()
            # Return an account-only proposal matching the caller's now-mutated
            # object. The orchestrator must still require its initial snapshot.
            return [SpecialistRequest(
                agent_name="account-specialist", instructions="Create requested account.",
                semantic_intent=SemanticIntent(summary="Create requested account.", effect="mutation",
                    allowed_tools=["account_create"], allowed_arguments={
                        "given_name": ["Mvp"], "family_name": ["Test"],
                    }),
            )]

    runtime = Runtime(registry)
    result = asyncio.run(Orchestrator(Router(), runtime, Primary()).run(REQUEST, workflow_contract=original))
    assert len(original.steps) == 1
    assert result.status == "error"
    assert not runtime.tasks
