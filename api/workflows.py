"""Integrate a bounded onboarding continuation into the existing job protocol.

Phoenix remains the authenticated owner of jobs, approvals and mail delivery.
These helpers run only inside its authenticated AI calls. There is no second
public execution API and no automatic approval of the follow-up Jira action.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any
import asyncio
import uuid

from agent.onboarding import WorkflowContinuation, WorkflowProposal, decode_ticket, verified_account
from agent.workflow_store import WorkflowStore
from subagents.core.definitions.types import AgentTask, HubResult


def workflow_store(runtime) -> WorkflowStore:
    store = WorkflowStore(runtime.approvals.store.path)
    store.initialize()
    return store


def persist_workflow_continuation(runtime, payload, hub_result) -> dict[str, Any] | None:
    """Call inside owner_execution, before persisting the initial completion."""
    plan = getattr(hub_result, "_pending_onboarding", None)
    if plan is None:
        return None
    if (hub_result.status != "approval_required" or len(hub_result.results) != 1
            or hub_result.results[0].proposed_tool != "account_create"):
        raise ValueError("Onboarding has no unambiguous account proposal.")
    from services.extensions.context import execution_selection

    selection = execution_selection.get()
    if selection is None:
        raise ValueError("Onboarding must bind the authenticated execution selection.")
    decode_ticket(plan)
    approval_id = hub_result.results[0].approval_id
    workflow_store(runtime).put(
        source_approval_id=approval_id, owner_id=payload.user_id,
        job_id=payload.job_id, attempt=payload.attempt,
        plan={"delegation": plan, "user_request": payload.message,
              "context": [turn.model_dump() for turn in payload.context],
              "selection": dict(selection)},
    )
    return {
        "kind": "account_ticket", "state": "awaiting_account_approval",
        "current_step": 1, "steps_total": 2, "source_approval_id": approval_id,
    }


def _response(approval_id: str, *, status: str, error: str,
              result: dict | None = None) -> dict[str, Any]:
    return WorkflowContinuation(
        source_approval_id=approval_id, status=status, proposed_tool=None,
        result=result, error=error,
    ).model_dump()


async def continue_after_approval(runtime, approval_id: str,
                                  protocol_payload: dict[str, Any]) -> dict[str, Any]:
    # A second live request in this process waits for the first generation and
    # replays its stored proposal, rather than prematurely marking a healthy
    # in-flight continuation unresolved. SQLite remains the crash/process CAS.
    locks = getattr(runtime, "workflow_locks", None)
    if locks is None:
        locks = runtime.workflow_locks = {}
    lock = locks.setdefault(approval_id, asyncio.Lock())
    async with lock:
        return await _continue_after_approval_locked(runtime, approval_id, protocol_payload)


async def _continue_after_approval_locked(runtime, approval_id: str,
                                         protocol_payload: dict[str, Any]) -> dict[str, Any]:
    """Advance only after this exact account approval has succeeded durably.

    A preparing claim is committed before inference. If the caller disconnects
    or the process stops, no later request reruns generation or a mutation.
    A ready continuation is replayed verbatim, including its one approval ID.
    """
    store = workflow_store(runtime)
    entry = store.get(approval_id)
    if entry is None or protocol_payload.get("execution_status") != "succeeded":
        return protocol_payload
    if protocol_payload.get("approval_id") != approval_id:
        raise ValueError("Onboarding source approval identity mismatch.")
    if entry["state"] in {"ready", "blocked"}:
        saved = WorkflowContinuation.model_validate(entry["response"]).model_dump()
        return {**protocol_payload, "continuation": saved}
    if not store.claim(approval_id):
        current = store.get(approval_id)
        if current is not None and current["state"] in {"ready", "blocked"}:
            saved = WorkflowContinuation.model_validate(current["response"]).model_dump()
            return {**protocol_payload, "continuation": saved}
        return {**protocol_payload, "continuation": _response(
            approval_id, status="outcome_unknown",
            error="The saved continuation is already preparing or was interrupted. Do not retry generation; trusted reconciliation is required.",
        )}

    try:
        source = runtime.approvals.store.get(approval_id)
        if source is None or source.tool != "account_create" or source.status != "approved":
            raise ValueError("Onboarding source is not an approved account creation.")
        identity = verified_account(source.result, source.arguments)
        plan = entry["plan"]
        ticket = decode_ticket(plan["delegation"])
        task = AgentTask(
            task_id=uuid.uuid4().hex, agent_name=ticket.agent_name,
            user_request=plan["user_request"], instructions=ticket.instructions,
            semantic_intent=ticket.semantic_intent,
            context={"conversation_context": plan["context"], "verified_account": identity},
        )
        from services.extensions.context import owner_execution
        from services.job_progress import job_progress

        async with job_progress(runtime.settings, entry["job_id"], entry["attempt"]):
            with owner_execution(entry["owner_id"], plan["user_request"], selection=plan["selection"]):
                result = await runtime.hub.runtime.run(task)

        if (result.status != "approval_required" or result.proposed_tool != "ticket_create"
                or not result.approval_id or result.approval_id == approval_id):
            response = _response(
                approval_id, status="failed",
                error="The account was created, but the requested Jira step did not produce a valid governed approval. No Jira action was executed.",
            )
        else:
            target = runtime.approvals.store.get(result.approval_id)
            if target is None or target.status != "pending" or target.tool != "ticket_create":
                raise ValueError("Onboarding ticket has no durable pending approval.")
            if target.arguments.get("project_key") not in ticket.semantic_intent.allowed_arguments.get("project_key", []):
                await runtime.approvals.reject_approval(target.approval_id)
                raise ValueError("Onboarding ticket changed the saved project scope.")
            summary = target.arguments.get("summary", "")
            if not isinstance(summary, str) or not any(
                value in summary for value in (identity.user_id, identity.email)
            ):
                # A malformed proposal is never shown as the next executable
                # step. Rejection only closes a pending proposal; no provider
                # operation or mutation retry occurs here.
                await runtime.approvals.reject_approval(target.approval_id)
                response = _response(
                    approval_id, status="failed",
                    error="The account was created, but the Jira proposal did not identify the verified account. No Jira action was executed.",
                )
            else:
                result = replace(result, proposed_arguments=target.arguments,
                                 task_instructions=task.instructions, semantic_intent=task.semantic_intent)
                hub_result = asdict(HubResult(
                    status="approval_required", user_request=plan["user_request"],
                    routes=[result.agent_name], results=[result], answer=result.answer,
                ))
                hub_result["workflow"] = {
                    "kind": "account_ticket", "state": "awaiting_ticket_approval",
                    "current_step": 2, "steps_total": 2, "source_approval_id": approval_id,
                    "account": asdict(identity),
                }
                response = WorkflowContinuation(
                    source_approval_id=approval_id, status="waiting_approval",
                    proposed_tool=WorkflowProposal(
                        tool="ticket_create", arguments=target.arguments, approval_id=target.approval_id,
                    ),
                    result=hub_result, error=None,
                ).model_dump()
        store.finish(approval_id, response, blocked=response["status"] != "waiting_approval")
    except Exception:
        # Do not serialize exception/provider text: it can contain request data
        # or secrets. The durable source approval retains the actual outcome.
        response = _response(
            approval_id, status="outcome_unknown",
            error="The account action finished, but its saved Jira continuation could not be confirmed. Do not repeat account creation or Jira generation; trusted reconciliation is required.",
        )
        store.finish(approval_id, response, blocked=True)
    return {**protocol_payload, "continuation": response}
