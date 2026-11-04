from types import SimpleNamespace
import pytest
from api.app import completion_payload, valid_completion_ack
from agent.completion_outbox import CompletionOutbox

@pytest.mark.parametrize("status", ["denied", "outcome_unknown", "success", "approval_required", "partial_error"])
def test_versioned_callback_survives_outbox_restart(tmp_path, status):
    payload = completion_payload(SimpleNamespace(attempt=1), {"status": status, "error": "Policy or execution detail"})
    expected = {"success": "completed", "approval_required": "waiting_approval", "partial_error": "failed"}.get(status, status)
    assert payload["status"] == expected
    assert payload["contract_version"] == 2
    if status in {"denied", "outcome_unknown"}:
        assert "error" in payload and "result" not in payload
    outbox = CompletionOutbox(tmp_path / "outbox.db")
    outbox.initialize()
    outbox.put("job-1", 1, payload)
    entry = CompletionOutbox(tmp_path / "outbox.db").list_pending()[0]
    ack = {"job_id": "job-1", "status": expected, "acknowledgement": "applied", "contract_version": 2}
    assert valid_completion_ack(entry=entry, body=ack)
    assert not valid_completion_ack(entry=entry, body={**ack, "contract_version": 3})
    assert not valid_completion_ack(entry=entry, body={**ack, "job_id": "other"})
    assert not valid_completion_ack(entry=entry, body={k:v for k,v in ack.items() if k != "contract_version"})
    assert valid_completion_ack(entry=entry, body={**ack, "status": "outcome_unknown", "acknowledgement": "duplicate"})
    assert not valid_completion_ack(entry=entry, body={**ack, "status": "processing"})

def test_a_cancelled_job_acknowledges_a_late_completion_as_a_duplicate(tmp_path):
    # The owner cancelled after this completion was saved; Phoenix keeps the
    # cancel and acknowledges the delivery, so the outbox may drop it.
    payload = completion_payload(SimpleNamespace(attempt=1), {"status": "success", "answer": "late"})
    outbox = CompletionOutbox(tmp_path / "outbox.db")
    outbox.initialize()
    outbox.put("job-1", 1, payload)
    entry = outbox.list_pending()[0]
    ack = {"job_id": "job-1", "status": "cancelled", "acknowledgement": "duplicate", "contract_version": 2}
    assert valid_completion_ack(entry=entry, body=ack)
    # Applying the completion can never leave the job cancelled.
    assert not valid_completion_ack(entry=entry, body={**ack, "acknowledgement": "applied"})
    # Version 1 reports a cancel as failed and never names it.
    legacy = SimpleNamespace(job_id="old", payload={"status": "completed"})
    assert not valid_completion_ack(entry=legacy, body={"job_id": "old", "status": "cancelled", "acknowledgement": "duplicate"})

def test_legacy_outbox_ack_is_still_supported():
    entry = SimpleNamespace(job_id="old", payload={"status": "failed"})
    assert valid_completion_ack(entry=entry, body={"job_id": "old", "status": "reconciliation_required", "acknowledgement": "duplicate"})
    assert not valid_completion_ack(entry=entry, body={"job_id": "old", "status": "outcome_unknown", "acknowledgement": "duplicate"})
