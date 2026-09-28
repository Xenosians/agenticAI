from learning.continual.automation.canary import (
    CanaryState,
    model_attributable_failure,
    observe_canary_event,
)


def test_canary_requests_rollback_after_failure_threshold():
    state = CanaryState(
        checkpoint_id="candidate",
        previous_checkpoint_id="previous",
        min_events_before_decision=4,
        max_events=10,
        max_failure_fraction=0.25,
    )

    for failed in [
        True,
        False,
        True,
        False,
    ]:
        state = observe_canary_event(
            state=state,
            had_error=failed,
        )

    assert state.rollback_requested
    assert state.complete


def test_first_checkpoint_canary_cannot_auto_rollback_to_unregistered_base():
    state = CanaryState(
        checkpoint_id="first",
        previous_checkpoint_id=None,
        min_events_before_decision=2,
        max_events=2,
        max_failure_fraction=0.0,
    )

    state = observe_canary_event(
        state=state,
        had_error=True,
    )

    state = observe_canary_event(
        state=state,
        had_error=True,
    )

    assert not state.rollback_requested
    assert state.complete



def test_provider_execution_error_is_not_a_model_canary_failure():
    assert not model_attributable_failure(
        outcome_codes=[
            "tool_execution_error",
        ],
        failure_types=[],
    )


def test_semantic_grounding_error_is_a_model_canary_failure():
    assert model_attributable_failure(
        outcome_codes=[
            "semantic_argument_unbound",
        ],
        failure_types=[],
    )
