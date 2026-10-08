from types import SimpleNamespace

import pytest

from api.app import (
    build_approval_protocol_payload,
    valid_completion_ack,
)


def test_success_is_protocol_success_and_execution_success():
    payload = build_approval_protocol_payload(
        "approval-1",
        {
            "ok": True,
            "approval": {"status": "approved"},
            "result": {"ok": True, "status": "success"},
            "replayed": False,
        },
    )

    assert payload["protocol_status"] == "processed"
    assert payload["execution_status"] == "succeeded"
    assert payload["approval_status"] == "approved"
    assert payload["error"] is None


def test_failed_execution_is_still_a_processed_protocol_response():
    payload = build_approval_protocol_payload(
        "approval-2",
        {
            "ok": False,
            "approval": {"status": "failed"},
            "result": {
                "ok": False,
                "status": "error",
                "error": "tests failed",
            },
            "replayed": False,
        },
    )

    assert payload["protocol_status"] == "processed"
    assert payload["execution_status"] == "failed"
    assert payload["error"] == "tests failed"


def test_unresolved_execution_is_distinct_from_failure():
    payload = build_approval_protocol_payload(
        "approval-3",
        {
            "ok": False,
            "approval": {"status": "executing"},
            "result": {"ok": False, "status": "outcome_unknown"},
            "error": "provider outcome could not be confirmed",
            "replayed": True,
        },
    )

    assert payload["execution_status"] == "unresolved"
    assert payload["approval_status"] == "executing"
    assert payload["replayed"] is True


def test_missing_durable_authority_state_is_protocol_failure():
    with pytest.raises(ValueError):
        build_approval_protocol_payload(
            "approval-4",
            {"ok": False, "error": "not found"},
        )


def test_completion_ack_accepts_reconciliation_required_duplicate():
    entry = SimpleNamespace(
        job_id="job-1",
        payload={"status": "waiting_approval"},
    )

    assert valid_completion_ack(
        entry=entry,
        body={
            "job_id": "job-1",
            "acknowledgement": "duplicate",
            "status": "reconciliation_required",
        },
    ) is True
