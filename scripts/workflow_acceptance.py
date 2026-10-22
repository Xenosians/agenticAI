"""Pure evidence checks for the real account -> Jira -> completion-mail flow.

These helpers neither submit requests nor grant approvals. A live driver must
bind its own source, owner, selected inference profile and configured providers,
persist intent before dispatch, and reconcile interrupted effects. A validated
job or outbox record alone never proves inbox delivery.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


class WorkflowEvidenceError(ValueError):
    """The observed product evidence does not establish the requested step."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise WorkflowEvidenceError(reason)


def _mapping(value: Any, reason: str) -> dict:
    _require(isinstance(value, dict), reason)
    return value


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def account_proposal(job: dict) -> dict:
    """Require a bounded workflow and return the exact first approval proposal."""
    _require(job.get("status") == "waiting_approval", "Account approval is not waiting")
    proposal = _mapping(job.get("proposed_tool"), "Account proposal is missing")
    result = _mapping(job.get("result"), "Workflow result is missing")
    workflow = _mapping(result.get("workflow"), "Bounded workflow is missing")
    _require(
        proposal.get("tool") == "account_create"
        and _nonempty(proposal.get("approval_id"))
        and isinstance(proposal.get("arguments"), dict)
        and workflow.get("kind") == "account_ticket"
        and workflow.get("state") == "awaiting_account_approval"
        and workflow.get("source_approval_id") == proposal["approval_id"],
        "Account proposal is not bound to the requested workflow",
    )
    return proposal


def ticket_proposal(job: dict, initial: dict, project: str) -> dict:
    """Check the actual account receipt before exposing the second approval."""
    source = account_proposal(initial)
    _require(job.get("job_id") == initial.get("job_id"), "Workflow changed job identity")
    _require(job.get("status") == "waiting_approval", "Ticket approval is not waiting")
    result = _mapping(job.get("result"), "Ticket workflow result is missing")
    workflow = _mapping(result.get("workflow"), "Ticket workflow is missing")
    proposal = _mapping(job.get("proposed_tool"), "Ticket proposal is missing")
    execution = _mapping(result.get("approval_execution"), "Account execution is missing")
    account = _mapping(execution.get("result"), "Account receipt is missing")
    arguments = _mapping(proposal.get("arguments"), "Ticket arguments are missing")
    _require(
        execution.get("approval_id") == source["approval_id"]
        and execution.get("execution_status") == "succeeded"
        and account.get("ok") is True
        and account.get("status") in {"success", "executed"}
        and account.get("operation") == "account_create"
        and account.get("account_created") is True
        and account.get("verification_ok") is True
        and account.get("credential_persisted") is True,
        "The source account has no verified persisted receipt",
    )
    identity = {name: account.get(name) for name in ("user_id", "email")}
    _require(all(_nonempty(value) for value in identity.values()), "Account identity is missing")
    for name, expected in (("user_id", "expected_username"), ("email", "expected_email")):
        _require(
            identity[name] == source["arguments"].get(expected),
            "Account identity differs from its exact approval",
        )
    _require(
        workflow.get("kind") == "account_ticket"
        and workflow.get("state") == "awaiting_ticket_approval"
        and workflow.get("source_approval_id") == source["approval_id"]
        and workflow.get("account") == identity
        and proposal.get("tool") == "ticket_create"
        and _nonempty(proposal.get("approval_id"))
        and proposal["approval_id"] != source["approval_id"],
        "Ticket proposal does not preserve the verified account and distinct approval",
    )
    summary = arguments.get("summary")
    _require(
        arguments.get("project_key") == project
        and isinstance(summary, str)
        and any(value in summary for value in identity.values()),
        "Ticket proposal changed the project or omitted the actual account",
    )
    return proposal


@dataclass(frozen=True)
class CompletedWorkflow:
    job_id: str
    attempt: int
    account_id: str
    account_email: str
    ticket_key: str


def completed_workflow(job: dict, initial: dict, ticket_waiting: dict, project: str) -> CompletedWorkflow:
    """Require both exact approvals and the terminal verified Jira read-back."""
    proposal = ticket_proposal(ticket_waiting, initial, project)
    _require(job.get("job_id") == initial.get("job_id") and _nonempty(job.get("job_id")),
             "Terminal workflow changed job identity")
    _require(job.get("status") == "completed", "Workflow is not completed")
    result = _mapping(job.get("result"), "Terminal result is missing")
    workflow = _mapping(result.get("workflow"), "Terminal workflow is missing")
    previous_workflow = ticket_waiting["result"]["workflow"]
    _require(
        workflow == {**previous_workflow, "state": "completed"},
        "Terminal workflow changed the verified account or workflow binding",
    )
    execution = _mapping(result.get("approval_execution"), "Jira execution is missing")
    receipt = _mapping(execution.get("result"), "Jira receipt is missing")
    key = receipt.get("ticket_key")
    _require(
        execution.get("approval_id") == proposal["approval_id"]
        and execution.get("execution_status") == "succeeded"
        and receipt.get("ok") is True
        and receipt.get("status") in {"success", "executed"}
        and receipt.get("provider") == "jira"
        and receipt.get("operation") == "create_ticket"
        and receipt.get("verification_ok") is True
        and receipt.get("mutation_performed") is True
        and isinstance(key, str)
        and re.fullmatch(re.escape(project.upper()) + r"-[0-9]+", key) is not None
        and receipt.get("new_value") == proposal["arguments"]["summary"],
        "Jira execution has no matching verified provider receipt",
    )
    history = result.get("approval_history")
    expected = (initial["proposed_tool"], proposal)
    _require(isinstance(history, list) and len(history) == 2, "Both approvals are not preserved")
    for entry, approved in zip(history, expected):
        entry = _mapping(entry, "Invalid approval history")
        recorded = _mapping(entry.get("execution"), "Approval history execution is missing")
        _require(entry.get("proposal") == approved
                 and recorded.get("approval_id") == approved["approval_id"]
                 and recorded.get("execution_status") == "succeeded",
                 "Approval history differs from the exact approved actions")
    attempt = job.get("attempts")
    _require(type(attempt) is int and attempt >= 1, "Terminal attempt identity is invalid")
    identity = workflow["account"]
    return CompletedWorkflow(job["job_id"], attempt, identity["user_id"], identity["email"], key)


def completion_mail_state(record: dict, completed: CompletedWorkflow) -> str:
    """Return observed outbox state; even `sent` establishes acceptance only."""
    _require(
        record.get("kind") == "job_notification"
        and record.get("dedupe_key") == f"job:{completed.job_id}:{completed.attempt}:completed",
        "Mail is not the durable completion event for this workflow",
    )
    body = record.get("text_body")
    _require(isinstance(body, str)
             and all(value in body for value in (completed.job_id, completed.account_id, completed.ticket_key)),
             "Completion mail does not identify this job, account and Jira ticket")
    state = record.get("status")
    _require(state in {"pending", "processing", "sent", "dead", "outcome_unknown"},
             "Completion mail state is invalid")
    return state


def confirm_inbox_message(message_id: str, body: str, completed: CompletedWorkflow) -> None:
    """Require independently fetched inbox evidence correlated to this workflow."""
    _require(_nonempty(message_id) and isinstance(body, str)
             and all(value in body for value in (completed.job_id, completed.account_id, completed.ticket_key)),
             "Inbox evidence is missing or belongs to another workflow")
