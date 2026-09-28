from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CanaryState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint_id: str
    previous_checkpoint_id: str | None = None

    observed_events: int = 0
    failed_events: int = 0

    min_events_before_decision: int = Field(
        default=5,
        ge=1,
    )

    max_events: int = Field(
        default=20,
        ge=1,
    )

    max_failure_fraction: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
    )

    rollback_requested: bool = False
    complete: bool = False


MODEL_ATTRIBUTABLE_FAILURES = {
    "agent_tool_not_allowed",
    "grounding_failed",
    "invalid_tool_call_count",
    "semantic_argument_not_allowed",
    "semantic_argument_unbound",
    "semantic_grounded_argument_invalid",
    "semantic_tool_not_allowed",
    "semantic_tool_not_available",
    "tool_parse_error",
}


def model_attributable_failure(
    *,
    outcome_codes: list[str],
    failure_types: list[str],
) -> bool:
    """
    Canary rollback should respond to model/semantic regressions, not a
    provider outage or unrelated infrastructure failure.
    """
    observed = {
        str(value).strip()
        for value in [
            *outcome_codes,
            *failure_types,
        ]
        if str(value).strip()
    }

    return bool(
        observed
        & MODEL_ATTRIBUTABLE_FAILURES
    )


def observe_canary_event(
    *,
    state: CanaryState,
    had_error: bool,
) -> CanaryState:
    if state.complete:
        return state

    observed = (
        state.observed_events
        + 1
    )

    failed = (
        state.failed_events
        + (
            1
            if had_error
            else 0
        )
    )

    fraction = (
        failed
        / observed
    )

    rollback_requested = (
        observed
        >= state.min_events_before_decision
        and fraction
        > state.max_failure_fraction
        and state.previous_checkpoint_id
        is not None
    )

    complete = (
        rollback_requested
        or observed
        >= state.max_events
    )

    return state.model_copy(
        update={
            "observed_events":
                observed,

            "failed_events":
                failed,

            "rollback_requested":
                rollback_requested,

            "complete":
                complete,
        }
    )
