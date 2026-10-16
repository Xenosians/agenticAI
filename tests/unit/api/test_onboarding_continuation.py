from contextlib import asynccontextmanager
from functools import wraps
from dataclasses import asdict
from types import SimpleNamespace
import asyncio
import json

import pytest
from pydantic import ValidationError

from agent.approval_store import ApprovalStore
from agent.approvals import ApprovalManager
from agent.onboarding import AccountTicketPlan, WorkflowContinuation, account_ticket_plan, encode_plan
from agent.workflow_store import WorkflowStore
from api.workflows import continue_after_approval, persist_workflow_continuation
from services.extensions.context import execution_profile, owner_execution
from services.extensions.store import ExtensionStore
from subagents.core.definitions.types import AgentResult, HubResult, SemanticIntent, SpecialistRequest
from subagents.core.orchestration.orchestrator import Orchestrator
from subagents.core.definitions.registry import AgentRegistry
from subagents.core.definitions.types import AgentDefinition
from subagents.core.orchestration.router import LLMRouter, RoutingContractError


def run_async(test):
    @wraps(test)
    def wrapped(*args, **kwargs):
        return asyncio.run(test(*args, **kwargs))
    return wrapped


def delegation(tool, agent, values):
    return SpecialistRequest(
        agent_name=agent, instructions=f"Perform {tool} for the requested identity.",
        semantic_intent=SemanticIntent(
            summary=f"Perform {tool} for requested identity.", effect="mutation",
            allowed_tools=[tool], allowed_arguments=values,
        ),
    )


def account_result(**changes):
    return {
        "ok": True, "status": "success", "operation": "account_create",
        "user_id": "mtest", "email": "mtest@itsm.test", "account_created": True,
        "verification_ok": True, "credential_persisted": True,
        "credential_id": "safe-reference", **changes,
    }


class Provider:
    def __init__(self):
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return account_result()


class TicketRuntime:
    def __init__(self, approvals):
        self.approvals = approvals
        self.calls = []
        self.summary = "Create requested onboarding ticket for mtest@itsm.test"
        self.project_key = "KAN"
        self.entered = None
        self.release = None

    async def run(self, task):
        self.calls.append((task, execution_profile.get()))
        if self.entered is not None:
            self.entered.set()
            await self.release.wait()
        arguments = {"project_key": self.project_key, "summary": self.summary,
                     "expected_project_key": "KAN", "expected_project_id": "10001"}
        proposal = self.approvals.create_approval("ticket_create", arguments)
        return AgentResult(
            task_id=task.task_id, agent_name=task.agent_name, status="approval_required",
            proposed_tool="ticket_create", proposed_arguments={"project_key": "KAN", "summary": self.summary},
            approval_id=proposal["id"], answer="The Jira ticket requires approval.",
        )


@pytest.fixture
def env(tmp_path, monkeypatch):
    approvals = ApprovalStore(tmp_path / "approvals.sqlite3")
    approvals.initialize()
    provider = Provider()
    manager = ApprovalManager(store=approvals, mcp=provider)
    ticket_runtime = TicketRuntime(manager)
    runtime = SimpleNamespace(approvals=manager, settings=SimpleNamespace(),
                              hub=SimpleNamespace(runtime=ticket_runtime))
    extensions = ExtensionStore(tmp_path / "extensions")
    connection = extensions.put("owner-1", "model", {"backend": "http", "name": "unit-connection", "state": "ready"}, "private-unit-secret")
    extensions.select("owner-1", model_id=connection["id"], dataset_id=None)
    monkeypatch.setattr("services.extensions.service.extension_store", lambda: extensions)

    @asynccontextmanager
    async def no_progress(*args):
        yield

    monkeypatch.setattr("services.job_progress.job_progress", no_progress)
    source = manager.create_approval("account_create", {
        "given_name": "Mvp", "family_name": "Test", "expected_username": "mtest",
        "expected_email": "mtest@itsm.test",
    })
    ticket = delegation("ticket_create", "jira-specialist", {"project_key": ["KAN"]})
    ticket.after_success_of = "account-specialist"
    hub_result = HubResult(
        status="approval_required", user_request="request", routes=["account-specialist"],
        results=[AgentResult(
            task_id="task-1", agent_name="account-specialist", status="approval_required",
            proposed_tool="account_create", approval_id=source["id"],
        )],
    )
    setattr(hub_result, "_pending_onboarding", encode_plan(AccountTicketPlan(ticket)))
    payload = SimpleNamespace(user_id="owner-1", job_id="job-1", attempt=1, context=[],
                              message="Create an account for Mvp Test, then create a Jira ticket in KAN for this account.")
    with owner_execution("owner-1", payload.message):
        marker = persist_workflow_continuation(runtime, payload, hub_result)
    return SimpleNamespace(runtime=runtime, source=source["id"], store=approvals, manager=manager,
                           provider=provider, ticket=ticket_runtime, extensions=extensions,
                           connection=connection, payload=payload, hub_result=hub_result, marker=marker)


async def approve_source(env):
    outcome = await env.manager.approve_approval(env.source)
    assert outcome["ok"] is True
    return {"approval_id": env.source, "protocol_status": "processed",
            "execution_status": "succeeded", "approval_status": "approved",
            "replayed": False, "result": outcome["result"], "error": None}


def test_only_exact_validated_two_operation_plan_is_admitted():
    account = delegation("account_create", "account-specialist", {"given_name": ["Mvp"], "family_name": ["Test"]})
    ticket = delegation("ticket_create", "jira-specialist", {"project_key": ["KAN"]})
    assert account_ticket_plan([account, ticket]) is None


@pytest.mark.parametrize("after", ["account-specialist", "unrelated-specialist", None])
@run_async
async def test_real_router_validates_explicit_dependency_without_routing_keywords(after):
    registry = AgentRegistry()
    for name, tool in (("account-specialist", "account_create"), ("jira-specialist", "ticket_create")):
        registry.register(AgentDefinition(name=name, description=name, model="test-model", tools=[tool]))
    account = delegation("account_create", "account-specialist", {"given_name": ["Mvp"], "family_name": ["Test"]})
    ticket = delegation("ticket_create", "jira-specialist", {"project_key": ["KAN"]})
    rows = [{"agent": item.agent_name, "instructions": item.instructions,
             "intent": {key: value for key, value in asdict(item.semantic_intent).items()
                        if key != "missing_required_arguments"}}
            for item in (account, ticket)]
    if after is not None:
        rows[1]["after_success_of"] = after

    class Inference:
        async def generate(self, **kwargs):
            return json.dumps({"delegations": rows})

    router = LLMRouter(registry, Inference(), "test-model", strict_contract=True)
    request = "Create account given name Mvp family name Test, then create a Jira ticket in KAN for the created account."
    if after == "unrelated-specialist":
        with pytest.raises(RoutingContractError, match="continuation"):
            await router.route(request)
    else:
        routed = await router.route(request)
        assert routed[1].after_success_of == after
        assert bool(account_ticket_plan(routed)) == (after is not None)
    ticket.after_success_of = account.agent_name
    assert account_ticket_plan([account, ticket]) is not None
    assert account_ticket_plan([ticket, account]) is None
    assert account_ticket_plan([account, ticket, ticket]) is None
    ticket.semantic_intent.clarification_required = True
    assert account_ticket_plan([account, ticket]) is None


@pytest.mark.parametrize("source_status", ["approval_required", "error"])
@run_async
async def test_orchestrator_suspends_saved_ticket_until_source_is_approved(source_status):
    account = delegation("account_create", "account-specialist", {"given_name": ["Mvp"], "family_name": ["Test"]})
    ticket = delegation("ticket_create", "jira-specialist", {"project_key": ["KAN"]})
    ticket.after_success_of = account.agent_name

    class Router:
        async def route(self, request):
            return [account, ticket]

    class Runtime:
        calls = []

        async def run(self, task):
            self.calls.append(task)
            return AgentResult(
                task_id=task.task_id, agent_name=task.agent_name, status=source_status,
                proposed_tool="account_create", approval_id="source-1" if source_status == "approval_required" else None,
                error="Provider unavailable" if source_status == "error" else None,
            )

    runtime = Runtime()
    result = await Orchestrator(Router(), runtime, None, require_semantic_intent=True).run("original request")
    assert [task.agent_name for task in runtime.calls] == ["account-specialist"]
    assert bool(getattr(result, "_pending_onboarding", None)) == (source_status == "approval_required")
    assert "_pending_onboarding" not in asdict(result)


@run_async
async def test_account_result_advances_to_separately_approved_ticket_and_replays(env):
    protocol = await approve_source(env)
    # Changing preferences cannot silently change the selected workflow model.
    env.extensions.select("owner-1", model_id=None, dataset_id=None)
    first = await continue_after_approval(env.runtime, env.source, protocol)
    continuation = first["continuation"]
    assert continuation["status"] == "waiting_approval"
    assert continuation["source_approval_id"] == env.source
    target = continuation["proposed_tool"]
    assert target["tool"] == "ticket_create"
    assert target["approval_id"] != env.source
    assert target["arguments"] == env.store.get(target["approval_id"]).arguments
    assert continuation["result"]["workflow"]["account"] == {
        "user_id": "mtest", "email": "mtest@itsm.test"
    }
    assert env.store.get(target["approval_id"]).status == "pending"
    assert [call[0] for call in env.provider.calls] == ["account_create"]
    task, profile = env.ticket.calls[0]
    assert asdict(task.context["verified_account"]) == {"user_id": "mtest", "email": "mtest@itsm.test"}
    assert profile["id"] == env.connection["id"]
    assert profile["api_key"] == "private-unit-secret"
    assert "private-unit-secret" not in json.dumps(WorkflowStore(env.store.path).get(env.source))
    assert "credential_id" not in json.dumps(continuation)
    replay = await continue_after_approval(env.runtime, env.source, protocol)
    assert replay == first
    assert len(env.ticket.calls) == 1


@run_async
@pytest.mark.parametrize("changes", [
    {"verification_ok": False}, {"credential_persisted": False}, {"account_created": False},
    {"user_id": "different"}, {"email": "different@itsm.test"}, {"operation": "enable_user"},
])
async def test_unverified_or_different_identity_never_generates_ticket(env, changes):
    env.store.claim_for_execution(env.source)
    env.store.mark_approved(env.source, account_result(**changes))
    response = await continue_after_approval(env.runtime, env.source, {
        "approval_id": env.source, "execution_status": "succeeded",
    })
    assert response["continuation"]["status"] == "outcome_unknown"
    assert env.ticket.calls == []
    assert env.provider.calls == []


@run_async
async def test_rejection_or_source_failure_does_not_start_ticket(env):
    for state in ("rejected", "failed", "unresolved"):
        protocol = {"approval_id": env.source, "execution_status": state}
        assert await continue_after_approval(env.runtime, env.source, protocol) == protocol
    assert env.ticket.calls == []


@run_async
async def test_restart_after_preparing_claim_never_regenerates(env):
    protocol = await approve_source(env)
    store = WorkflowStore(env.store.path)
    assert store.claim(env.source) is True
    response = await continue_after_approval(env.runtime, env.source, protocol)
    assert response["continuation"]["status"] == "outcome_unknown"
    assert env.ticket.calls == []
    assert store.get(env.source)["state"] == "preparing"


@run_async
async def test_concurrent_calls_generate_exactly_one_persisted_ticket_proposal(env):
    protocol = await approve_source(env)
    env.ticket.entered, env.ticket.release = asyncio.Event(), asyncio.Event()
    first = asyncio.create_task(continue_after_approval(env.runtime, env.source, protocol))
    await env.ticket.entered.wait()
    second = asyncio.create_task(continue_after_approval(env.runtime, env.source, protocol))
    await asyncio.sleep(0)
    assert not second.done()
    env.ticket.release.set()
    completed, concurrent = await asyncio.gather(first, second)
    assert concurrent == completed
    replay = await continue_after_approval(env.runtime, env.source, protocol)
    assert replay == completed
    assert len(env.ticket.calls) == 1
    assert len(env.provider.calls) == 1


@run_async
async def test_unlinked_ticket_is_rejected_without_executing_jira(env):
    env.ticket.summary = "Some unrelated task"
    response = await continue_after_approval(env.runtime, env.source, await approve_source(env))
    assert response["continuation"]["status"] == "failed"
    with env.store._connect() as db:
        target = db.execute("SELECT status FROM approvals WHERE tool='ticket_create'").fetchone()
    assert target["status"] == "rejected"
    assert [name for name, _ in env.provider.calls] == ["account_create"]


@run_async
async def test_ticket_cannot_move_outside_saved_user_project(env):
    env.ticket.project_key = "OTHER"
    response = await continue_after_approval(env.runtime, env.source, await approve_source(env))
    assert response["continuation"]["status"] == "outcome_unknown"
    assert len(env.provider.calls) == 1
    with env.store._connect() as db:
        row = db.execute("SELECT status FROM approvals WHERE tool='ticket_create'").fetchone()
    assert row["status"] == "rejected"


@pytest.mark.parametrize("change", ["same_id", "wrong_result_tool", "wrong_result_arguments", "no_proposal"])
@run_async
async def test_continuation_schema_requires_exact_distinct_saved_proposal(env, change):
    response = await continue_after_approval(env.runtime, env.source, await approve_source(env))
    continuation = response["continuation"]
    if change == "same_id":
        continuation["proposed_tool"]["approval_id"] = env.source
    elif change == "wrong_result_tool":
        continuation["result"]["results"][0]["proposed_tool"] = "account_create"
    elif change == "wrong_result_arguments":
        continuation["result"]["results"][0]["proposed_arguments"] = {}
    else:
        continuation["proposed_tool"] = None
    with pytest.raises(ValidationError):
        WorkflowContinuation.model_validate(continuation)


def test_owner_or_job_cannot_rebind_existing_approval(env):
    store = WorkflowStore(env.store.path)
    entry = store.get(env.source)
    for name, value in (("owner_id", "other-owner"), ("job_id", "other-job"), ("attempt", 2)):
        data = {key: entry[key] for key in ("source_approval_id", "owner_id", "job_id", "attempt", "plan")}
        data[name] = value
        with pytest.raises(ValueError, match="rebound"):
            store.put(**data)


@run_async
async def test_deleted_pinned_connection_blocks_instead_of_using_another_model(env):
    protocol = await approve_source(env)
    env.extensions.delete("owner-1", env.connection["id"])
    response = await continue_after_approval(env.runtime, env.source, protocol)
    assert response["continuation"]["status"] == "outcome_unknown"
    assert env.ticket.calls == []
