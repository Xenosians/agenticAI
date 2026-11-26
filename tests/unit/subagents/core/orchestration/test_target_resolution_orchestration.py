"""A vague Proxmox request reaches the specialist with the exact guest live inventory names.

Owner decision, 26 November 2026: one clear match is proposed with its real ID and name; several, the
owner picks from a short list; none, Janus says so. Only the guest identity is ever looked up, from the
account's own Proxmox; the rest of the path (binding, SemanticGuard, ToolGateway, policy freeze,
approval) runs unchanged on the real values.
"""
import asyncio
from pathlib import Path

import pytest

from subagents.core.definitions.loader import load_agent_directory
from subagents.core.definitions.types import AgentResult, SemanticIntent, SpecialistRequest
from subagents.core.orchestration import orchestrator as orchestrator_module
from subagents.core.orchestration.orchestrator import Orchestrator
from subagents.core.orchestration.semantic_guard import SemanticGuard
from subagents.core.orchestration.target_resolution import GuestRecord
from subagents.core.tooling.gateway import validate_grounded_arguments
from tools.registry import get_tool

AGENTS = Path(__file__).resolve().parents[5] / "subagents/agents"

GUESTS = [
    GuestRecord(vmid=990, node="hummingbird", type="lxc", name="janus-lab", status="stopped", tags=("janus",)),
    GuestRecord(vmid=991, node="hummingbird", type="lxc", name="janus-trainer", status="running",
                tags=("janus",)),
    GuestRecord(vmid=992, node="hummingbird", type="lxc", name="janus-control", status="running", tags=()),
]


def unresolved(tool="proxmox_guest_start", effect="mutation"):
    """What the routing contract leaves when the request named the operation but no exact guest."""
    return SpecialistRequest(
        agent_name="proxmox-specialist", instructions="Start the guest the user named.",
        semantic_intent=SemanticIntent(
            summary="Start the guest.", effect=effect, allowed_tools=[tool], forbidden_tools=[],
            allowed_arguments={}, forbidden_arguments={}, max_tool_calls=1, clarification_required=True,
            missing_required_arguments=["node", "vmid"]))


class PlannedRouter:
    def __init__(self, delegations):
        self.delegations = delegations

    async def route_with_repair(self, user_request, *, context=None, workflow_contract=None):
        return list(self.delegations)


class RecordingRuntime:
    def __init__(self):
        self.tasks = []

    async def run(self, task):
        self.tasks.append(task)
        return AgentResult(task_id=task.task_id, agent_name=task.agent_name, status="approval_required",
                           proposed_tool=task.semantic_intent.allowed_tools[0],
                           proposed_arguments={"node": "hummingbird", "vmid": 990},
                           answer="Approval required: Start LXC 990 (janus-lab) on Proxmox node hummingbird.")


class NeverPrimary:
    async def respond(self, *args, **kwargs):
        raise AssertionError("a Proxmox request is never answered without its specialist")

    async def synthesize(self, *args, **kwargs):
        raise AssertionError("approval and clarification answers are deterministic")


@pytest.fixture
def inventory(monkeypatch):
    calls = []

    def read():
        calls.append(1)
        return list(GUESTS)

    monkeypatch.setattr(orchestrator_module, "proxmox_inventory", read)
    return calls


def run(request, delegations, runtime=None):
    runtime = runtime or RecordingRuntime()
    orchestrator = Orchestrator(router=PlannedRouter(delegations), runtime=runtime, primary_assistant=NeverPrimary(),
                                require_semantic_intent=True)
    return asyncio.run(orchestrator.run(request)), runtime


def test_one_clear_match_runs_the_specialist_on_the_real_guest(inventory):
    result, runtime = run("start the lab container", [unresolved()])

    assert inventory == [1]
    [task] = runtime.tasks
    intent = task.semantic_intent
    assert intent.clarification_required is False
    assert intent.missing_required_arguments == []
    assert intent.allowed_arguments == {"node": ["hummingbird"], "vmid": ["990"]}
    # The gateway grounds against the user's own words plus the trusted line naming the exact guest.
    assert task.user_request.startswith("start the lab container")
    assert "LXC 990 (janus-lab) on node hummingbird" in task.user_request
    assert result.status == "approval_required"
    # The answer says which guest was found, so the owner sees the real identity before approving.
    assert "janus-lab" in result.answer and "990" in result.answer


def test_the_resolved_target_passes_the_unchanged_guard_and_gateway_grounding(inventory):
    _, runtime = run("start the lab container", [unresolved()])
    [task] = runtime.tasks
    agent = next(agent for agent in load_agent_directory(AGENTS) if agent.name == "proxmox-specialist")
    decision = SemanticGuard().evaluate(agent=agent, intent=task.semantic_intent, tool_name="proxmox_guest_start",
                                        arguments={"node": "hummingbird", "vmid": 990})
    assert decision.allowed, decision.decision_code
    tool = get_tool("proxmox_guest_start")
    assert validate_grounded_arguments(task.user_request, {"node": "hummingbird", "vmid": 990},
                                       tool["grounded_arguments"]) == (True, None)
    # Another guest is still refused: resolution named one guest, not the whole inventory.
    valid, _ = validate_grounded_arguments(task.user_request, {"node": "hummingbird", "vmid": 991},
                                           tool["grounded_arguments"])
    assert valid is False


def test_several_matches_ask_the_owner_and_run_nothing(inventory):
    result, runtime = run("restart the janus container", [unresolved("proxmox_guest_reboot")])

    assert runtime.tasks == []
    assert result.status == "success"
    [step] = result.results
    assert step.status == "clarification_required"
    assert step.outcome_code == "target_ambiguous"
    for name, vmid in (("janus-lab", 990), ("janus-trainer", 991), ("janus-control", 992)):
        assert name in result.answer and str(vmid) in result.answer


def test_no_match_says_so_and_runs_nothing(inventory):
    result, runtime = run("start the minecraft server", [unresolved()])

    assert runtime.tasks == []
    [step] = result.results
    assert step.status == "clarification_required" and step.outcome_code == "target_not_found"
    assert "No Proxmox guest" in result.answer and "janus-lab" in result.answer


def test_an_exact_request_is_never_looked_up(inventory):
    exact = SpecialistRequest(
        agent_name="proxmox-specialist", instructions="Start guest 990.",
        semantic_intent=SemanticIntent(
            summary="Start guest 990.", effect="mutation", allowed_tools=["proxmox_guest_start"],
            allowed_arguments={"node": ["hummingbird"], "vmid": ["990"]}, max_tool_calls=1))
    _, runtime = run("start VMID 990 on node hummingbird", [exact])

    assert inventory == []
    [task] = runtime.tasks
    assert task.user_request == "start VMID 990 on node hummingbird"


def test_other_missing_values_stay_a_normal_question(inventory):
    snapshot = SpecialistRequest(
        agent_name="proxmox-specialist", instructions="Roll back a snapshot.",
        semantic_intent=SemanticIntent(
            summary="Roll back.", effect="mutation", allowed_tools=["proxmox_snapshot_rollback"],
            allowed_arguments={}, max_tool_calls=1, clarification_required=True,
            missing_required_arguments=["node", "snapshot", "vmid"]))
    result, runtime = run("roll back the lab container", [snapshot])

    assert inventory == []
    assert runtime.tasks == []
    assert result.results[0].outcome_code == "semantic_clarification_required"


def test_only_proxmox_capabilities_are_looked_up(inventory):
    account = SpecialistRequest(
        agent_name="account-specialist", instructions="Unlock the account.",
        semantic_intent=SemanticIntent(
            summary="Unlock.", effect="mutation", allowed_tools=["unlock_user"], allowed_arguments={},
            max_tool_calls=1, clarification_required=True, missing_required_arguments=["user_id"]))
    result, _ = run("unlock the lab account", [account])

    assert inventory == []
    assert result.results[0].outcome_code == "semantic_clarification_required"


def test_inventory_is_read_once_and_only_the_resolved_step_gains_the_trusted_line(inventory):
    other = SpecialistRequest(
        agent_name="account-specialist", instructions="Read the account.",
        semantic_intent=SemanticIntent(summary="Read.", effect="read", allowed_tools=["account_status"],
                                       allowed_arguments={}, max_tool_calls=1))
    _, runtime = run("start the lab container and check my account", [unresolved(), other])

    assert inventory == [1]
    proxmox, account = runtime.tasks
    assert "[Janus resolved from live Proxmox inventory]" in proxmox.user_request
    # Another specialist in the same request can never ground a value against the resolved line.
    assert account.user_request == "start the lab container and check my account"


def test_an_unreadable_inventory_asks_and_runs_nothing(monkeypatch):
    def broken():
        raise RuntimeError("connection refused")

    monkeypatch.setattr(orchestrator_module, "proxmox_inventory", broken)
    result, runtime = run("start the lab container", [unresolved()])

    assert runtime.tasks == []
    [step] = result.results
    assert step.status == "clarification_required" and step.outcome_code == "target_unavailable"
    assert "could not read your Proxmox inventory" in result.answer
    assert "connection refused" not in result.answer


def test_a_router_description_in_an_identity_argument_is_replaced_by_the_real_value(inventory):
    described = SpecialistRequest(
        agent_name="proxmox-specialist", instructions="Start the lab container.",
        semantic_intent=SemanticIntent(
            summary="Start.", effect="mutation", allowed_tools=["proxmox_guest_start"],
            allowed_arguments={"vmid": ["lab"]}, max_tool_calls=1, clarification_required=True,
            missing_required_arguments=["node"]))
    _, runtime = run("start the lab container", [described])

    [task] = runtime.tasks
    assert task.semantic_intent.allowed_arguments == {"node": ["hummingbird"], "vmid": ["990"]}


def test_a_reviewed_workflow_keeps_its_own_targets(inventory, monkeypatch):
    # A reviewed workflow's targets are the owner's reviewed values (validated against the contract):
    # inventory never adds or changes one.
    from agent.requested_workflow import RequestedWorkflowContract

    monkeypatch.setattr(orchestrator_module, "admit_requested_workflow", lambda contract, registry: {})
    monkeypatch.setattr(orchestrator_module, "validate_requested_plan", lambda contract, delegations, owners: None)
    runtime = RecordingRuntime()
    runtime.agent_registry = object()
    orchestrator = Orchestrator(router=PlannedRouter([unresolved()]), runtime=runtime,
                                primary_assistant=NeverPrimary(), require_semantic_intent=True)
    contract = RequestedWorkflowContract.model_validate({"version": 1, "steps": [
        {"id": "start", "capability": "proxmox_guest_start", "grounded_arguments": {}, "after_success_of": None}]})

    result = asyncio.run(orchestrator.run("start the lab container", workflow_contract=contract))

    assert inventory == []
    assert runtime.tasks == []
    [step] = result.results
    assert step.outcome_code == "semantic_clarification_required"
