"""The approval a person reviews is the exact action that approval executes.

Trusted capability policy may bind extra execution facts to a proposal before it is
approved: account_create derives the username, email, container and policy version
from the requested name. The durable approval stores and executes those facts, so the
job proposal shown on the approval card must carry the same arguments.

This drives the real specialist runtime, ToolGateway and ApprovalManager with the
registered account_create capability. Only the directory-backed policy resolver is
replaced by a fixed identity, so no directory is contacted.
"""

import asyncio
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from agent.approval_store import ApprovalStore
from agent.approvals import ApprovalManager
from api.app import completion_payload
from subagents.core.definitions.loader import load_agent_directory
from subagents.core.definitions.registry import AgentRegistry
from subagents.core.definitions.types import AgentResult, AgentTask, HubResult
from subagents.core.orchestration.runtime import AgentRuntime
from subagents.core.tooling.gateway import ToolGateway
from tools.registry import get_tool


REQUESTED = {"given_name": "Grace", "family_name": "Hopper", "department": "IT", "role": "Engineer"}
DERIVED = {
    "expected_username": "ghopper",
    "expected_email": "ghopper@corp.example",
    "expected_container_dn": "CN=Users,DC=corp,DC=example",
    "identity_policy_version": "test-policy",
}


def account_capability(name):
    tool = get_tool(name)
    if name != "account_create":
        return tool
    return {**tool, "policy_resolver": lambda **requested: {
        "ok": True, "status": "ready", "risk": "high", "requires_approval": True,
        "execution_arguments": {**requested, **DERIVED}, "error": None,
    }}


class AccountProposal:
    async def generate(self, **kwargs):
        return json.dumps([{"name": "account_create", "arguments": REQUESTED}])


class NoProvider:
    async def call_tool(self, name, arguments):
        raise AssertionError("A proposal must not execute before approval")


@pytest.fixture
def proposal(tmp_path):
    store = ApprovalStore(tmp_path / "approvals.sqlite3")
    store.initialize()
    approvals = ApprovalManager(store=store, mcp=NoProvider(), tool_lookup=account_capability)
    gateway = ToolGateway(approval_creator=approvals.create_approval, mcp=NoProvider(),
                          tool_lookup=account_capability)
    registry = AgentRegistry()
    registry.register_many(load_agent_directory("subagents/agents"))
    runtime = AgentRuntime(agent_registry=registry, inference=AccountProposal(), tool_gateway=gateway)
    result = asyncio.run(runtime.run(AgentTask(
        task_id="account-binding", agent_name="account-specialist",
        user_request="Create a corporate account for Grace Hopper, department IT, role Engineer.",
    )))
    return result, store


def job_proposal(*results):
    """The proposal Phoenix stores for the job and the approval card renders."""
    hub = asdict(HubResult(status=results[0].status, user_request="request",
                           routes=[item.agent_name for item in results], results=list(results)))
    return completion_payload(SimpleNamespace(attempt=1), hub)["proposed_tool"]


def test_reviewed_proposal_is_the_exact_durable_approval(proposal):
    result, store = proposal

    assert result.status == "approval_required"
    durable = store.get(result.approval_id or "")
    assert durable is not None and durable.status == "pending"
    assert durable.arguments == {**REQUESTED, **DERIVED}
    # The job proposal, the approval card and the durable approval are one action.
    assert job_proposal(result) == {"tool": "account_create", "arguments": durable.arguments,
                                    "approval_id": result.approval_id}
    assert result.approval_arguments == durable.arguments


def test_specialist_arguments_stay_separate_from_policy_bound_facts(proposal):
    result, _store = proposal

    # Audit and learning evidence keep what the specialist proposed. Trusted
    # policy facts are never recorded as model output.
    assert result.proposed_arguments == REQUESTED


def test_results_without_an_approval_keep_their_proposed_arguments():
    read = AgentResult(task_id="read", agent_name="account-specialist", status="success",
                       proposed_tool="account_status", proposed_arguments={"user_id": "ghopper"})

    assert job_proposal(read) == {"tool": "account_status", "arguments": {"user_id": "ghopper"},
                                  "approval_id": None}
