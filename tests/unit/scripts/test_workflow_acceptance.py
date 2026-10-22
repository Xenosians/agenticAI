from copy import deepcopy

import pytest

from scripts.workflow_acceptance import (
    CompletedWorkflow,
    WorkflowEvidenceError,
    account_proposal,
    completed_workflow,
    completion_mail_state,
    confirm_inbox_message,
    ticket_proposal,
)


@pytest.fixture
def flow():
    account = {
        "ok": True, "status": "executed", "operation": "account_create",
        "account_created": True, "verification_ok": True, "credential_persisted": True,
        "user_id": "fixture.person", "email": "fixture.person@example.test",
    }
    source = {
        "tool": "account_create", "approval_id": "fixture-account-approval",
        "arguments": {"expected_username": account["user_id"], "expected_email": account["email"]},
    }
    workflow = {
        "kind": "account_ticket", "state": "awaiting_account_approval",
        "source_approval_id": source["approval_id"],
    }
    initial = {
        "job_id": "fixture-job", "status": "waiting_approval", "proposed_tool": source,
        "result": {"workflow": workflow},
    }
    proposal = {
        "tool": "ticket_create", "approval_id": "fixture-ticket-approval",
        "arguments": {"project_key": "LAB", "summary": f"Onboard {account['user_id']}"},
    }
    account_execution = {
        "approval_id": source["approval_id"], "execution_status": "succeeded", "result": account,
    }
    waiting = {
        "job_id": initial["job_id"], "status": "waiting_approval", "proposed_tool": proposal,
        "result": {
            "workflow": {**workflow, "state": "awaiting_ticket_approval",
                         "account": {name: account[name] for name in ("user_id", "email")}},
            "approval_execution": account_execution,
        },
    }
    execution = {
        "approval_id": proposal["approval_id"], "execution_status": "succeeded",
        "result": {"ok": True, "status": "success", "provider": "jira", "operation": "create_ticket",
                   "verification_ok": True, "mutation_performed": True, "ticket_key": "LAB-7",
                   "new_value": proposal["arguments"]["summary"]},
    }
    completed = {
        "job_id": initial["job_id"], "attempts": 1, "status": "completed",
        "result": {
            "workflow": {**waiting["result"]["workflow"], "state": "completed"},
            "approval_execution": execution,
            "approval_history": [{"proposal": source, "execution": account_execution},
                                 {"proposal": proposal, "execution": execution}],
        },
    }
    return initial, waiting, completed


def test_exact_account_ticket_and_both_approvals_establish_completed_workflow(flow):
    initial, waiting, final = flow
    assert account_proposal(initial) == initial["proposed_tool"]
    assert ticket_proposal(waiting, initial, "LAB") == waiting["proposed_tool"]
    assert completed_workflow(final, initial, waiting, "LAB") == CompletedWorkflow(
        "fixture-job", 1, "fixture.person", "fixture.person@example.test", "LAB-7",
    )


@pytest.mark.parametrize("field,value", [
    ("account_created", False), ("verification_ok", False), ("credential_persisted", False),
    ("user_id", "someone.else"), ("email", "someone.else@example.test"),
])
def test_unverified_or_substituted_account_cannot_advance(flow, field, value):
    initial, waiting, _ = deepcopy(flow)
    waiting["result"]["approval_execution"]["result"][field] = value
    with pytest.raises(WorkflowEvidenceError):
        ticket_proposal(waiting, initial, "LAB")


@pytest.mark.parametrize("change", ["same-approval", "other-project", "missing-identity", "other-job"])
def test_ticket_proposal_must_preserve_exact_scope(flow, change):
    initial, waiting, _ = deepcopy(flow)
    if change == "same-approval":
        waiting["proposed_tool"]["approval_id"] = initial["proposed_tool"]["approval_id"]
    elif change == "other-project":
        waiting["proposed_tool"]["arguments"]["project_key"] = "OTHER"
    elif change == "missing-identity":
        waiting["proposed_tool"]["arguments"]["summary"] = "An unrelated account"
    else:
        waiting["job_id"] = "other-job"
    with pytest.raises(WorkflowEvidenceError):
        ticket_proposal(waiting, initial, "LAB")


@pytest.mark.parametrize("field,value", [
    ("ok", False), ("verification_ok", False), ("mutation_performed", False),
    ("provider", "mock"), ("operation", "list_tickets"), ("ticket_key", "OTHER-7"),
    ("new_value", "A different summary"),
])
def test_http_completed_or_claimed_success_cannot_replace_jira_readback(flow, field, value):
    initial, waiting, final = deepcopy(flow)
    final["result"]["approval_execution"]["result"][field] = value
    with pytest.raises(WorkflowEvidenceError):
        completed_workflow(final, initial, waiting, "LAB")


def test_missing_history_or_rebound_terminal_identity_is_rejected(flow):
    initial, waiting, final = deepcopy(flow)
    final["result"]["approval_history"] = final["result"]["approval_history"][:1]
    with pytest.raises(WorkflowEvidenceError):
        completed_workflow(final, initial, waiting, "LAB")

    final = deepcopy(flow[2])
    final["result"]["workflow"]["account"]["user_id"] = "other"
    with pytest.raises(WorkflowEvidenceError):
        completed_workflow(final, initial, waiting, "LAB")


@pytest.mark.parametrize("state", ["pending", "processing", "sent", "dead", "outcome_unknown"])
def test_outbox_state_is_preserved_without_claiming_inbox_delivery(flow, state):
    completed = completed_workflow(flow[2], flow[0], flow[1], "LAB")
    record = {
        "kind": "job_notification", "dedupe_key": "job:fixture-job:1:completed", "status": state,
        "text_body": "fixture-job\nCreated account: fixture.person\nCreated Jira ticket: LAB-7",
    }
    assert completion_mail_state(record, completed) == state
    with pytest.raises(WorkflowEvidenceError):
        confirm_inbox_message("", record["text_body"], completed)
    confirm_inbox_message("<actual-inbox-message@example.test>", record["text_body"], completed)


@pytest.mark.parametrize("change", ["wrong-event", "missing-account", "missing-ticket", "missing-job"])
def test_another_mail_or_missing_correlation_is_not_completion_evidence(flow, change):
    completed = completed_workflow(flow[2], flow[0], flow[1], "LAB")
    record = {
        "kind": "job_notification", "dedupe_key": "job:fixture-job:1:completed", "status": "sent",
        "text_body": "fixture-job fixture.person LAB-7",
    }
    if change == "wrong-event":
        record["dedupe_key"] = "job:another-job:1:completed"
    else:
        removed = {"missing-account": "fixture.person", "missing-ticket": "LAB-7", "missing-job": "fixture-job"}[change]
        record["text_body"] = record["text_body"].replace(removed, "")
    with pytest.raises(WorkflowEvidenceError):
        completion_mail_state(record, completed)
