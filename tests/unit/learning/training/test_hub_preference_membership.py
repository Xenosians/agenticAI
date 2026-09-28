from collections import Counter

from learning.curriculum.catalog import get_curriculum
from learning.curriculum.mastery import CurriculumState, initial_curriculum_state
from learning.training.hub_preferences import HubPreferenceRecord
from learning.training.membership import _hub_preference_candidates


def _hub_state_at(chapter_id: str) -> CurriculumState:
    curriculum = get_curriculum("hub")
    initial = initial_curriculum_state(curriculum)

    return CurriculumState(
        curriculum_id=initial.curriculum_id,
        curriculum_version=initial.curriculum_version,
        target_component=initial.target_component,
        updated_at=initial.updated_at,
        current_chapter_id=chapter_id,
        mastered_chapter_ids=[],
        chapter_attempt_counts={},
        best_mastery_scores={},
        last_evaluation_id=None,
        status="learning",
    )


def _record() -> HubPreferenceRecord:
    return HubPreferenceRecord(
        record_id="hub-pref-1",
        created_at="2026-09-25T00:00:00+00:00",
        trajectory_id="trajectory-1",
        correction_id="hub-correction-1",
        source_attempt_id="hub-route-1",
        target_model_key="hub-main",
        user_request="Create a Task.",
        prompt_messages=[
            {
                "role": "system",
                "content": "router prompt",
            },
            {
                "role": "user",
                "content": "Create a Task.",
            },
        ],
        rejected='{"delegations":[]}',
        chosen='{"delegations":[{}]}',
        failure_codes=[
            "semantic_grounded_argument_invalid",
        ],
        source_attempt_validation_status="accepted",
        source_model_artifact_sha256="1" * 64,
        source_model_weights_sha256="2" * 64,
        source_tokenizer_artifact_sha256="3" * 64,
        source_model_profile_sha256="4" * 64,
        source_system_prompt_sha256="5" * 64,
        source_capability_catalog_sha256="6" * 64,
        source_messages_sha256="7" * 64,
        source_attempt_sha256="8" * 64,
        source_trajectory_sha256="9" * 64,
        source_correction_sha256="a" * 64,
        trajectory_review_id="review-trajectory",
        correction_review_id="review-correction",
        dataset_eligible=True,
        training_authorized=False,
    )


def test_hub_preference_waits_for_matching_curriculum_chapter():
    curriculum = get_curriculum("hub")
    record = _record()
    promoted = [
        (
            record,
            ".runtime/learning/datasets/hub-preference/v000001/records.jsonl",
            "f" * 64,
        )
    ]

    exclusions = Counter()

    chapter_one = _hub_preference_candidates(
        curriculum=curriculum,
        state=_hub_state_at(
            "hub-01-structured-instructions"
        ),
        promoted_records=promoted,
        exclusions=exclusions,
    )

    assert chapter_one == []
    assert exclusions[
        "hub_preference_chapter_unmatched"
    ] == 1

    chapter_five = _hub_preference_candidates(
        curriculum=curriculum,
        state=_hub_state_at(
            "hub-05-grounded-bindings"
        ),
        promoted_records=promoted,
        exclusions=Counter(),
    )

    assert len(chapter_five) == 1
    assert chapter_five[0].role == "current_behavior"
    assert chapter_five[0].target_component == "hub"
    assert chapter_five[0].trajectory_id == "trajectory-1"
