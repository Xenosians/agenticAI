from pathlib import Path

from learning.curriculum.lessons import (
    CurriculumLessonDatasetBuilder,
    CurriculumLessonRecorder,
    CurriculumLessonReviewRecorder,
    build_lesson,
)


def test_hub_lesson_requires_valid_routing_object():
    lesson = build_lesson(
        curriculum_name="hub",
        chapter_id="hub-01-structured-instructions",
        user_request="Hello.",
        chosen_response='{"delegations":[]}',
        rejected_response="Hello.",
    )
    assert lesson.target_component == "hub"
    assert lesson.dataset_eligible is False


def test_lesson_identity_is_deterministic():
    kwargs = dict(
        curriculum_name="hub",
        chapter_id="hub-01-structured-instructions",
        user_request="Hello.",
        chosen_response='{"delegations":[]}',
        rejected_response="Hello.",
    )
    one = build_lesson(**kwargs)
    two = build_lesson(**kwargs)
    assert one.lesson_id == two.lesson_id
    assert one.content_sha256 == two.content_sha256


def test_approved_lesson_promotes_without_training_authority(tmp_path: Path):
    lesson_path = tmp_path / "lessons.jsonl"
    review_path = tmp_path / "reviews.jsonl"
    dataset_root = tmp_path / "datasets"
    lesson = build_lesson(
        curriculum_name="hub",
        chapter_id="hub-01-structured-instructions",
        user_request="Hello.",
        chosen_response='{"delegations":[]}',
        rejected_response="Hello.",
    )
    CurriculumLessonRecorder(path=lesson_path).record(lesson=lesson)
    CurriculumLessonReviewRecorder(path=review_path).record(
        lesson_id=lesson.lesson_id,
        decision="approve",
        source="trusted_review",
        reason="Reviewed foundation lesson.",
    )
    result = CurriculumLessonDatasetBuilder(
        lesson_path=lesson_path,
        review_path=review_path,
        dataset_root=dataset_root,
    ).build(
        curriculum_name="hub",
        promoted_by="trusted_review",
        promotion_reason="Unit test.",
    )
    assert result.manifest.record_count == 1
    assert result.manifest.training_authorized is False
    assert result.manifest.promotion_authorized is False


def test_unapproved_lesson_cannot_promote(tmp_path: Path):
    lesson_path = tmp_path / "lessons.jsonl"
    review_path = tmp_path / "reviews.jsonl"
    dataset_root = tmp_path / "datasets"
    lesson = build_lesson(
        curriculum_name="hub",
        chapter_id="hub-01-structured-instructions",
        user_request="Hello.",
        chosen_response='{"delegations":[]}',
    )
    CurriculumLessonRecorder(path=lesson_path).record(lesson=lesson)
    try:
        CurriculumLessonDatasetBuilder(
            lesson_path=lesson_path,
            review_path=review_path,
            dataset_root=dataset_root,
        ).build(
            curriculum_name="hub",
            promoted_by="trusted_review",
            promotion_reason="Unit test.",
        )
    except ValueError as exc:
        assert "No approved curriculum lessons" in str(exc)
    else:
        raise AssertionError("Unapproved lesson was promoted.")
