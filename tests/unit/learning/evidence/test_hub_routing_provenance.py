import json

from pathlib import Path

from learning.evidence.hub_routing import (
    accept_current_hub_routing_attempt,
    capture_hub_routing_generation,
    consume_hub_routing_trace,
    current_hub_routing_trace,
    record_current_hub_routing_ledger,
    reject_current_hub_routing_attempt,
    reset_hub_routing_trace,
    set_current_hub_routing_parsed_output,
)


def _capture(
    *,
    response: str,
    repair_mode: bool = False,
):
    return capture_hub_routing_generation(
        model_key="hub-main",
        model_profile_resolver=None,
        specialists=[
            {
                "name": "jira-specialist",
                "capabilities": [
                    {
                        "name": "ticket_create",
                    }
                ],
            }
        ],
        messages=[
            {
                "role": "system",
                "content": "router prompt",
            },
            {
                "role": "user",
                "content": "Create a task.",
            },
        ],
        context_messages=[],
        user_request="Create a task.",
        repair_mode=repair_mode,
        include_workflow_protocol=(not repair_mode),
        repair_error=("bad plan" if repair_mode else None),
        max_new_tokens=512,
        response=response,
    )


def test_accepted_attempt_is_hash_pinned_and_not_training_eligible():
    reset_hub_routing_trace()

    _capture(response='{"delegations":[]}')

    set_current_hub_routing_parsed_output(
        {
            "delegations": [],
        }
    )

    accept_current_hub_routing_attempt(
        validated_delegation_count=0,
    )

    trace = current_hub_routing_trace()

    assert len(trace) == 1

    attempt = trace[0]

    assert attempt.validation_status == "accepted"
    assert attempt.validated_delegation_count == 0
    assert attempt.training_eligible is False
    assert len(attempt.system_prompt_sha256) == 64
    assert len(attempt.capability_catalog_sha256) == 64
    assert len(attempt.messages_sha256) == 64
    assert len(attempt.raw_response_sha256) == 64


def test_rejected_then_repair_attempt_are_both_preserved():
    reset_hub_routing_trace()

    _capture(response='{"delegations":[{"bad":true}]}')
    reject_current_hub_routing_attempt("invalid semantic contract")

    _capture(
        response='{"delegations":[]}',
        repair_mode=True,
    )
    set_current_hub_routing_parsed_output(
        {
            "delegations": [],
        }
    )
    accept_current_hub_routing_attempt(
        validated_delegation_count=0,
    )

    trace = current_hub_routing_trace()

    assert [item.mode for item in trace] == [
        "normal",
        "repair",
    ]

    assert [item.validation_status for item in trace] == [
        "rejected",
        "accepted",
    ]


def test_missing_model_resolver_fails_closed_for_training_identity():
    reset_hub_routing_trace()

    attempt = _capture(response='{"delegations":[]}')

    assert attempt.model_identity.complete is False
    assert (
        attempt.model_identity.incomplete_reason
        == "model-profile-resolver-unavailable"
    )


def test_parsed_output_is_sanitized_before_persistence():
    reset_hub_routing_trace()

    _capture(response='{"delegations":[]}')

    set_current_hub_routing_parsed_output(
        {
            "password": "hunter2",
            "delegations": [],
        }
    )

    accept_current_hub_routing_attempt(
        validated_delegation_count=0,
    )

    attempt = current_hub_routing_trace()[0]

    assert attempt.parsed_output["password"] == "<redacted>"


def test_consuming_trace_clears_current_task_state():
    reset_hub_routing_trace()

    _capture(response='{"delegations":[]}')
    accept_current_hub_routing_attempt(
        validated_delegation_count=0,
    )

    trace = consume_hub_routing_trace()

    assert len(trace) == 1
    assert current_hub_routing_trace() == []


def test_ledger_links_attempt_to_trajectory_and_job(tmp_path: Path):
    reset_hub_routing_trace()

    _capture(response='{"delegations":[]}')
    accept_current_hub_routing_attempt(
        validated_delegation_count=0,
    )

    path = tmp_path / "hub-routing-attempts.jsonl"

    count = record_current_hub_routing_ledger(
        trajectory_id="trajectory-1",
        job_id="job-1",
        job_attempt=1,
        path=path,
    )

    assert count == 1

    stored = json.loads(
        path.read_text(encoding="utf-8").splitlines()[0]
    )

    assert stored["schema"] == "hub-routing-ledger-record.v1"
    assert stored["trajectory_id"] == "trajectory-1"
    assert stored["attempt"]["training_eligible"] is False
