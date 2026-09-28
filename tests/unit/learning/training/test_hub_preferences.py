import hashlib
import json
from pathlib import Path

import pytest

from learning.continual.storage import (
    canonical_json,
    immutable_write_jsonl,
    load_jsonl_models,
)
from learning.curation.reviews import ReviewDecisionRecorder
from learning.evidence.hub_routing import (
    HubModelIdentity,
    HubRoutingAttemptEvidence,
    HubRoutingLedgerRecord,
)
from learning.evidence.types import (
    ExecutionReward,
    LearningTrajectory,
    TrajectorySignals,
    TrajectoryStep,
)
from learning.training.hub_preferences import (
    HubCorrectionRecorder,
    HubCorrectionReviewRecorder,
    HubPreferenceDatasetBuilder,
    HubPreferenceRecord,
)


def _sha(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def _trajectory() -> LearningTrajectory:
    return LearningTrajectory(
        trajectory_id="trajectory-1",
        observed_at="2026-09-25T00:00:00+00:00",
        job_id="job-1",
        attempt=1,
        user_request=(
            'Create a Task in Jira project KAN with summary "Checkpoint demo ticket".'
        ),
        hub_model="hub-main",
        hub_status="partial_error",
        routes=[
            "jira-specialist",
        ],
        steps=[
            TrajectoryStep(
                task_id="task-1",
                agent="jira-specialist",
                status="error",
                outcome_code=(
                    "semantic_grounded_argument_invalid"
                ),
                semantic_guard_decision={
                    "decision_code": (
                        "semantic_grounded_argument_invalid"
                    ),
                },
                proposed_tool="ticket_create",
                proposed_arguments={
                    "project_key": "KAN",
                    "summary": "Checkpoint demo ticket",
                    "ticket_type": None,
                },
            )
        ],
        final_answer="Request failed safely.",
        signals=TrajectorySignals(
            delegated=True,
            route_count=1,
            specialist_count=1,
            specialist_error_count=1,
            overall_success=False,
            had_error=True,
        ),
        execution_reward=ExecutionReward(),
        dataset_eligible=False,
    )


def _routing_record(
    trajectory: LearningTrajectory,
) -> HubRoutingLedgerRecord:
    messages = [
        {
            "role": "system",
            "content": "router prompt",
        },
        {
            "role": "user",
            "content": trajectory.user_request,
        },
    ]

    rejected_output = {
        "delegations": [
            {
                "agent": "jira-specialist",
                "instructions": "Create the Jira task.",
                "intent": {
                    "summary": "Create the Jira task",
                    "effect": "mutation",
                    "allowed_tools": [
                        "ticket_create",
                    ],
                    "allowed_arguments": {
                        "project_key": [
                            "KAN",
                        ],
                        "summary": [
                            "Checkpoint demo ticket",
                        ],
                    },
                    "forbidden_arguments": {},
                    "max_tool_calls": 1,
                    "clarification_required": False,
                },
            }
        ]
    }

    rejected = json.dumps(
        rejected_output,
        sort_keys=True,
    )

    message_sha = _sha(
        canonical_json(
            messages
        )
    )

    return HubRoutingLedgerRecord(
        trajectory_id=trajectory.trajectory_id,
        job_id=trajectory.job_id,
        job_attempt=1,
        attempt=HubRoutingAttemptEvidence(
            attempt_id="hub-route-1",
            created_at="2026-09-25T00:00:00+00:00",
            mode="normal",
            include_workflow_protocol=True,
            model_identity=HubModelIdentity(
                complete=True,
                model_key="hub-main",
                backend="hf-causal",
                quantization="4bit",
                compute_dtype="bfloat16",
                model_profile_sha256="1" * 64,
                model_artifact_sha256="2" * 64,
                model_weights_sha256="3" * 64,
                tokenizer_artifact_sha256="4" * 64,
                model_config_sha256="5" * 64,
                prompt_renderer_sha256="6" * 64,
            ),
            max_new_tokens=512,
            system_prompt_sha256=_sha(
                "router prompt"
            ),
            capability_catalog_sha256="7" * 64,
            messages_sha256=message_sha,
            conversation_context_sha256=_sha(
                canonical_json([])
            ),
            user_request_sha256=_sha(
                trajectory.user_request
            ),
            messages=messages,
            sanitized_messages_sha256=message_sha,
            messages_exact=True,
            raw_response=rejected,
            raw_response_sha256=_sha(
                rejected
            ),
            sanitized_response_sha256=_sha(
                rejected
            ),
            raw_response_exact=True,
            parsed_output=rejected_output,
            validation_status="accepted",
            validated_delegation_count=1,
            training_eligible=False,
        ),
        training_eligible=False,
    )


def _chosen_response() -> str:
    return json.dumps(
        {
            "delegations": [
                {
                    "agent": "jira-specialist",
                    "instructions": "Create the Jira task.",
                    "intent": {
                        "summary": "Create the Jira task",
                        "effect": "mutation",
                        "allowed_tools": [
                            "ticket_create",
                        ],
                        "allowed_arguments": {
                            "project_key": [
                                "KAN",
                            ],
                            "summary": [
                                "Checkpoint demo ticket",
                            ],
                            "ticket_type": [
                                "Task",
                            ],
                        },
                        "forbidden_arguments": {},
                        "max_tool_calls": 1,
                        "clarification_required": False,
                    },
                }
            ]
        },
        sort_keys=True,
    )


def _fixture(tmp_path: Path):
    trajectory = _trajectory()
    routing = _routing_record(
        trajectory
    )

    trajectories_path = (
        tmp_path
        / "trajectories.jsonl"
    )
    routing_path = (
        tmp_path
        / "hub-routing.jsonl"
    )
    corrections_path = (
        tmp_path
        / "hub-corrections.jsonl"
    )
    correction_reviews_path = (
        tmp_path
        / "hub-correction-reviews.jsonl"
    )
    trajectory_reviews_path = (
        tmp_path
        / "reviews.jsonl"
    )

    immutable_write_jsonl(
        trajectories_path,
        [
            trajectory,
        ],
    )
    immutable_write_jsonl(
        routing_path,
        [
            routing,
        ],
    )

    correction_recorder = HubCorrectionRecorder(
        path=corrections_path,
        enabled=True,
    )

    correction = correction_recorder.record(
        trajectory_id=trajectory.trajectory_id,
        attempt=routing,
        chosen_response=_chosen_response(),
        note="Preserve the explicit grounded ticket type.",
    )

    assert correction is not None

    return {
        "trajectory": trajectory,
        "routing": routing,
        "correction": correction,
        "trajectories_path": trajectories_path,
        "routing_path": routing_path,
        "corrections_path": corrections_path,
        "correction_reviews_path": correction_reviews_path,
        "trajectory_reviews_path": trajectory_reviews_path,
        "dataset_root": tmp_path / "datasets" / "hub-preference",
    }


def test_hub_preference_requires_both_review_boundaries(
    tmp_path: Path,
):
    fixture = _fixture(
        tmp_path
    )

    HubCorrectionReviewRecorder(
        path=fixture[
            "correction_reviews_path"
        ],
        enabled=True,
    ).record(
        correction_id=fixture[
            "correction"
        ].correction_id,
        decision="approve",
        source="trusted_review",
        reason="Reviewed chosen Hub contract.",
    )

    builder = HubPreferenceDatasetBuilder(
        trajectory_path=fixture[
            "trajectories_path"
        ],
        routing_path=fixture[
            "routing_path"
        ],
        correction_path=fixture[
            "corrections_path"
        ],
        correction_review_path=fixture[
            "correction_reviews_path"
        ],
        trajectory_review_path=fixture[
            "trajectory_reviews_path"
        ],
        dataset_root=fixture[
            "dataset_root"
        ],
    )

    with pytest.raises(
        ValueError,
        match="trajectory_review_not_approved",
    ):
        builder.build(
            eval_paths=[],
            promotion_reason="test",
        )


def test_reviewed_hub_correction_promotes_exact_preference_record(
    tmp_path: Path,
):
    fixture = _fixture(
        tmp_path
    )

    ReviewDecisionRecorder(
        path=fixture[
            "trajectory_reviews_path"
        ],
        enabled=True,
    ).record(
        subject_type="trajectory",
        subject_id=fixture[
            "trajectory"
        ].trajectory_id,
        decision="approve",
        source="trusted_review",
        reason="Trajectory is suitable for Hub correction evidence.",
    )

    HubCorrectionReviewRecorder(
        path=fixture[
            "correction_reviews_path"
        ],
        enabled=True,
    ).record(
        correction_id=fixture[
            "correction"
        ].correction_id,
        decision="approve",
        source="trusted_review",
        reason="Chosen Hub contract preserves the explicit user value.",
    )

    builder = HubPreferenceDatasetBuilder(
        trajectory_path=fixture[
            "trajectories_path"
        ],
        routing_path=fixture[
            "routing_path"
        ],
        correction_path=fixture[
            "corrections_path"
        ],
        correction_review_path=fixture[
            "correction_reviews_path"
        ],
        trajectory_review_path=fixture[
            "trajectory_reviews_path"
        ],
        dataset_root=fixture[
            "dataset_root"
        ],
    )

    result = builder.build(
        eval_paths=[],
        promotion_reason="test reviewed Hub preference",
    )

    assert result.manifest.record_count == 1
    assert result.manifest.training_authorized is False

    records = load_jsonl_models(
        Path(
            result.output_directory
        )
        / "records.jsonl",
        HubPreferenceRecord,
    )

    assert len(records) == 1

    record = records[0]

    assert record.target_component == "hub"
    assert record.source_attempt_id == "hub-route-1"
    assert "semantic_grounded_argument_invalid" in record.failure_codes
    assert record.prompt_messages[0]["content"] == "router prompt"
    assert '"ticket_type": ["Task"]' in record.chosen
    assert record.training_authorized is False
