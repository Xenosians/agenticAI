from api.app import (
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
        == "failed"
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
