from dataclasses import FrozenInstanceError

import pytest

from subagents.llm.runtime.stages import (
    InferenceStage,
    StagePolicy,
    resolve_stage_policy,
)


@pytest.mark.parametrize(
    "stage",
    [
        InferenceStage.CONTROL_ROUTING,
        InferenceStage.CANDIDATE_NARROWING,
        InferenceStage.NETWORKING,
    ],
)
def test_local_stages_cannot_select_an_adapter_or_receive_retrieval(stage):
    policy = resolve_stage_policy(stage)

    assert policy.allow_selected is False
    assert policy.allow_retrieval is False
    assert policy.request_token_ceiling is None
    assert policy.output_token_ceiling is None


def test_semantic_routing_uses_selected_adapter_with_compact_budget_only():
    policy = resolve_stage_policy(InferenceStage.SEMANTIC_ROUTING)

    assert policy.allow_selected is True
    assert policy.allow_retrieval is False
    assert policy.request_token_ceiling == 5000
    assert policy.output_token_ceiling == 768


def test_explicit_selected_networking_stage_excludes_retrieval_and_bounds_remote_requests():
    policy = resolve_stage_policy(InferenceStage.NETWORKING_SELECTED)
    assert policy.allow_selected is True
    assert policy.allow_retrieval is False
    assert policy.request_token_ceiling == 6000
    assert policy.output_token_ceiling == 1024


@pytest.mark.parametrize(
    "stage",
    [
        InferenceStage.SPECIALIST_REASONING,
        InferenceStage.PRIMARY_RESPONSE,
        InferenceStage.SYNTHESIS,
    ],
)
def test_reasoning_stages_use_selected_adapter_and_bounded_retrieval(stage):
    policy = resolve_stage_policy(stage)

    assert policy.allow_selected is True
    assert policy.allow_retrieval is True
    assert policy.request_token_ceiling == 6000
    assert policy.output_token_ceiling == 1024


@pytest.mark.parametrize(
    "origin",
    [stage for stage in InferenceStage if stage is not InferenceStage.REPAIR],
)
def test_repair_preserves_every_source_stage_restriction(origin):
    assert resolve_stage_policy(
        InferenceStage.REPAIR,
        repair_of=origin,
    ) == resolve_stage_policy(origin)


@pytest.mark.parametrize("stage", list(InferenceStage))
def test_stable_stage_strings_resolve_to_the_same_policy_as_enum_members(stage):
    origin = (
        InferenceStage.SEMANTIC_ROUTING
        if stage is InferenceStage.REPAIR
        else None
    )

    assert str(stage) == stage.value
    assert resolve_stage_policy(
        stage.value,
        repair_of=origin.value if origin is not None else None,
    ) == resolve_stage_policy(stage, repair_of=origin)


@pytest.mark.parametrize("stage", ["", "primary_response", "UNKNOWN", None, 1, []])
def test_unknown_stages_fail_closed(stage):
    with pytest.raises(ValueError, match="Unknown inference stage"):
        resolve_stage_policy(stage)


def test_repair_requires_an_explicit_source_stage():
    with pytest.raises(ValueError, match="requires an explicit repair_of"):
        resolve_stage_policy(InferenceStage.REPAIR)


@pytest.mark.parametrize("origin", ["", "UNKNOWN", 1, []])
def test_repair_with_unknown_source_fails_closed(origin):
    with pytest.raises(ValueError, match="Unknown repair_of stage"):
        resolve_stage_policy(InferenceStage.REPAIR, repair_of=origin)


@pytest.mark.parametrize("origin", [InferenceStage.REPAIR, "REPAIR"])
def test_nested_repair_fails_closed(origin):
    with pytest.raises(ValueError, match="cannot repair another REPAIR"):
        resolve_stage_policy(InferenceStage.REPAIR, repair_of=origin)


@pytest.mark.parametrize(
    "stage",
    [stage for stage in InferenceStage if stage is not InferenceStage.REPAIR],
)
def test_non_repair_stage_rejects_repair_context(stage):
    with pytest.raises(ValueError, match="only valid for the REPAIR"):
        resolve_stage_policy(stage, repair_of=InferenceStage.CONTROL_ROUTING)


def test_policy_cannot_be_mutated_between_requests():
    policy = resolve_stage_policy(InferenceStage.CONTROL_ROUTING)

    assert isinstance(policy, StagePolicy)
    with pytest.raises(FrozenInstanceError):
        policy.allow_selected = True

    assert resolve_stage_policy(InferenceStage.CONTROL_ROUTING).allow_selected is False
