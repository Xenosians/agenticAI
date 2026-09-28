from learning.curriculum.catalog import get_curriculum
from learning.curriculum.mastery import CurriculumState, initial_curriculum_state
from learning.curriculum.lessons import CurriculumLessonDatasetRecord
from learning.training.membership import classify_curriculum_lesson_role


def _state(*, current: str, mastered=None):
    curriculum = get_curriculum("hub")
    base = initial_curriculum_state(curriculum)
    return CurriculumState(
        curriculum_id=base.curriculum_id,
        curriculum_version=base.curriculum_version,
        target_component=base.target_component,
        updated_at=base.updated_at,
        current_chapter_id=current,
        mastered_chapter_ids=mastered or [],
        chapter_attempt_counts={},
        best_mastery_scores={},
        last_evaluation_id=None,
        status="learning",
    )


def _record(chapter_id: str):
    return CurriculumLessonDatasetRecord(
        record_id="record-1",
        created_at="2026-01-01T00:00:00+00:00",
        lesson_id="lesson-1",
        lesson_review_id="review-1",
        curriculum_id="hub-general-language-code-routing-v1",
        curriculum_version="1.0.0",
        target_component="hub",
        chapter_id=chapter_id,
        user_request="Hello.",
        chosen_response='{"delegations":[]}',
        rejected_response="Hello.",
        source_lesson_sha256="a" * 64,
        dataset_eligible=True,
        training_authorized=False,
    )


def test_current_chapter_lesson_is_current_lesson():
    assert classify_curriculum_lesson_role(
        target_component="hub",
        state=_state(current="hub-01-structured-instructions"),
        record=_record("hub-01-structured-instructions"),
    ) == "current_lesson"


def test_mastered_chapter_lesson_is_replay_lesson():
    assert classify_curriculum_lesson_role(
        target_component="hub",
        state=_state(current="hub-02-resource-operation", mastered=["hub-01-structured-instructions"]),
        record=_record("hub-01-structured-instructions"),
    ) == "replay_lesson"


def test_future_chapter_lesson_is_not_visible():
    assert classify_curriculum_lesson_role(
        target_component="hub",
        state=_state(current="hub-01-structured-instructions"),
        record=_record("hub-05-grounded-bindings"),
    ) is None
