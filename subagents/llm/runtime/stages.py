"""Trusted inference-stage policy, independent of model or provider names."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class InferenceStage(StrEnum):
    CONTROL_ROUTING = "CONTROL_ROUTING"
    CANDIDATE_NARROWING = "CANDIDATE_NARROWING"
    SEMANTIC_ROUTING = "SEMANTIC_ROUTING"
    SPECIALIST_REASONING = "SPECIALIST_REASONING"
    PRIMARY_RESPONSE = "PRIMARY_RESPONSE"
    SYNTHESIS = "SYNTHESIS"
    REPAIR = "REPAIR"
    NETWORKING = "NETWORKING"


@dataclass(frozen=True)
class StagePolicy:
    allow_selected: bool
    allow_retrieval: bool
    request_token_ceiling: int | None = None
    output_token_ceiling: int | None = None


_LOCAL_POLICY = StagePolicy(
    allow_selected=False,
    allow_retrieval=False,
)

_SELECTED_REASONING_POLICY = StagePolicy(
    allow_selected=True,
    allow_retrieval=True,
    request_token_ceiling=6000,
    output_token_ceiling=1024,
)

_STAGE_POLICIES = {
    InferenceStage.CONTROL_ROUTING: _LOCAL_POLICY,
    InferenceStage.CANDIDATE_NARROWING: _LOCAL_POLICY,
    InferenceStage.SEMANTIC_ROUTING: StagePolicy(
        allow_selected=True,
        allow_retrieval=False,
        request_token_ceiling=5000,
        output_token_ceiling=768,
    ),
    InferenceStage.SPECIALIST_REASONING: _SELECTED_REASONING_POLICY,
    InferenceStage.PRIMARY_RESPONSE: _SELECTED_REASONING_POLICY,
    InferenceStage.SYNTHESIS: _SELECTED_REASONING_POLICY,
    InferenceStage.NETWORKING: _LOCAL_POLICY,
}


def resolve_stage_policy(
    stage: InferenceStage | str,
    repair_of: InferenceStage | str | None = None,
) -> StagePolicy:
    """Resolve explicit caller intent; invalid or ambiguous intent fails closed.

    Repair preserves its explicitly named source stage's adapter, retrieval,
    and budget restrictions. Ceilings bound external requests in addition to
    the provider's existing limits; they never raise a provider budget.
    """

    try:
        resolved = InferenceStage(stage)
    except (TypeError, ValueError):
        raise ValueError("Unknown inference stage.") from None

    if resolved is not InferenceStage.REPAIR:
        if repair_of is not None:
            raise ValueError("repair_of is only valid for the REPAIR stage.")
        return _STAGE_POLICIES[resolved]

    if repair_of is None:
        raise ValueError("REPAIR requires an explicit repair_of stage.")

    try:
        origin = InferenceStage(repair_of)
    except (TypeError, ValueError):
        raise ValueError("Unknown repair_of stage.") from None

    if origin is InferenceStage.REPAIR:
        raise ValueError("REPAIR cannot repair another REPAIR stage.")

    return _STAGE_POLICIES[origin]
