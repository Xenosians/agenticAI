from types import SimpleNamespace

from api.app import (
    completion_payload,
    completion_status,
)


def test_governed_denial_is_not_completed():

    assert (
        completion_status(
            {
                "status":
                    "denied",
            }
        )
        == "denied"
    )


def test_actual_hub_error_remains_failed():

    assert (
        completion_status(
            {
                "status":
                    "partial_error",
            }
        )
        == "failed"
    )


def test_approval_remains_waiting():

    assert (
        completion_status(
            {
                "status":
                    "approval_required",
            }
        )
        == "waiting_approval"
    )


def test_a_denied_request_reports_the_reason_of_the_step_that_stopped_it():
    reason = "The tool 'user_groups' is turned off in your MCP Tool Store. Turn it on there to use it."

    payload = completion_payload(
        SimpleNamespace(attempt=1),
        {"status": "denied", "error": None, "answer": reason},
    )

    assert payload["status"] == "denied"
    assert payload["error"] == reason
