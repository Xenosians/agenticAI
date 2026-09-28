from learning.continuous.events import (
    learning_event_from_trajectory,
)


def test_trajectory_becomes_non_authoritative_event():
    event = learning_event_from_trajectory(
        {
            "trajectory_id":
                "trajectory-1",

            "job_id":
                "job-1",

            "attempt":
                1,

            "hub_model":
                "hub-main",

            "hub_status":
                "partial_error",

            "signals": {
                "overall_success":
                    False,

                "had_error":
                    True,

                "waiting_approval":
                    False,
            },

            "execution_reward": {
                "total":
                    -0.5,
            },

            "quality": {
                "failure_types": [
                    "grounding_failed",
                ],
            },

            "steps": [
                {
                    "outcome_code":
                        "grounding_failed",
                }
            ],
        }
    )

    assert event.had_error
    assert (
        event.execution_reward
        == -0.5
    )
    assert (
        "grounding_failed"
        in event.outcome_codes
    )

    # Runtime observation never self-promotes into training truth.
    assert not event.training_eligible
