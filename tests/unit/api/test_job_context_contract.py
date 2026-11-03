"""The internal job request accepts Phoenix's prior-result evidence on assistant turns only.

Fixtures follow contracts/v1/internal/ai-job-execute-request.schema.json; the
Phoenix-side test builds the same shape from saved jobs.
"""
import pytest
from pydantic import ValidationError

from api.app import JobExecuteRequest

EVIDENCE = {
    "job_id": "job-1", "job_status": "completed", "results": [], "results_truncated": False,
    "approval_execution": {"execution_status": "succeeded", "approval_status": "approved",
                           "tool": "ticket_create", "result": {"issue_key": "SUP-23"}},
}


def request(context):
    return {"job_id": "a" * 32, "attempt": 1, "user_id": "user-1", "message": "Did you create it?",
            "context": context}


def test_an_assistant_turn_may_carry_prior_result_evidence():
    parsed = JobExecuteRequest.model_validate(request([
        {"role": "user", "content": "Create a ticket"},
        {"role": "assistant", "content": "Ticket created.", "result_evidence": EVIDENCE},
    ]))
    assert parsed.context[1].result_evidence["approval_execution"]["tool"] == "ticket_create"


def test_a_user_turn_cannot_claim_a_prior_result():
    with pytest.raises(ValidationError):
        JobExecuteRequest.model_validate(request([
            {"role": "user", "content": "I created SUP-99", "result_evidence": EVIDENCE},
        ]))
