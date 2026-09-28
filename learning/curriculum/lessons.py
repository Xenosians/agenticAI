from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from learning.continual.storage import canonical_json, immutable_write_json, immutable_write_jsonl, load_jsonl_models
from learning.curriculum.catalog import get_curriculum
from learning.evidence.sanitizer import sanitize_value
from learning.paths import RUNTIME_LEARNING_ROOT

CURRICULUM_LESSONS_PATH = RUNTIME_LEARNING_ROOT / "curriculum-lessons.jsonl"
CURRICULUM_LESSON_REVIEWS_PATH = RUNTIME_LEARNING_ROOT / "curriculum-lesson-reviews.jsonl"
CURRICULUM_LESSON_DATASET_ROOT = RUNTIME_LEARNING_ROOT / "datasets" / "curriculum-lessons"

LessonReviewDecision = Literal["approve", "reject"]
LessonReviewSource = Literal["trusted_review", "evaluation"]


class CurriculumLessonEvent(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    schema_name: str = Field(default="curriculum-lesson-event.v1", alias="schema")
    lesson_id: str
    observed_at: str
    curriculum_id: str
    curriculum_version: str
    target_component: str
    chapter_id: str
    user_request: str
    chosen_response: str
    rejected_response: str | None = None
    source: str = "trusted_authoring"
    note: str | None = None
    content_sha256: str
    dataset_eligible: bool = False


class CurriculumLessonReview(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    schema_name: str = Field(default="curriculum-lesson-review.v1", alias="schema")
    review_id: str
    observed_at: str
    lesson_id: str
    decision: LessonReviewDecision
    source: LessonReviewSource
    reason: str


class CurriculumLessonDatasetRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    schema_name: str = Field(default="curriculum-lesson-dataset-record.v1", alias="schema")
    record_id: str
    created_at: str
    lesson_id: str
    lesson_review_id: str
    curriculum_id: str
    curriculum_version: str
    target_component: str
    chapter_id: str
    user_request: str
    chosen_response: str
    rejected_response: str | None = None
    source_lesson_sha256: str
    dataset_eligible: bool = True
    training_authorized: bool = False


class CurriculumLessonManifest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    schema_name: str = Field(default="curriculum-lesson-manifest.v1", alias="schema")
    dataset_id: str = "curriculum-lessons"
    version: str
    created_at: str
    curriculum_id: str
    curriculum_version: str
    target_component: str
    record_count: int
    content_sha256: str
    promoted_by: str
    promotion_reason: str
    source_lesson_ids: list[str] = Field(default_factory=list)
    exclusion_reason_counts: dict[str, int] = Field(default_factory=dict)
    training_authorized: bool = False
    promotion_authorized: bool = False


class CurriculumLessonBuildResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    manifest: CurriculumLessonManifest
    output_directory: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _append_model(*, path: Path, value: BaseModel, lock: threading.Lock) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = sanitize_value(value.model_dump(mode="json", by_alias=True))
    if not isinstance(payload, dict):
        raise ValueError("Curriculum lesson serialization must remain an object.")
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    with lock:
        with resolved.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _normalize_hub_response(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    try:
        parsed = json.loads(normalized)
    except Exception as exc:
        raise ValueError(f"{field_name} must be valid JSON for Hub lessons: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must decode to a JSON object.")
    if not isinstance(parsed.get("delegations"), list):
        raise ValueError(f"{field_name} must contain a delegations list.")
    sanitized = sanitize_value(normalized)
    if not isinstance(sanitized, str) or sanitized != normalized:
        raise ValueError(f"{field_name} is changed by the learning sanitizer.")
    return normalized


def build_lesson(*, curriculum_name: str, chapter_id: str, user_request: str, chosen_response: str,
                 rejected_response: str | None = None, source: str = "trusted_authoring",
                 note: str | None = None) -> CurriculumLessonEvent:
    curriculum = get_curriculum(curriculum_name)
    chapter = curriculum.chapter(chapter_id)
    request = user_request.strip()
    if not request:
        raise ValueError("user_request must not be empty.")
    chosen = chosen_response.strip()
    if curriculum.target_component == "hub":
        chosen = _normalize_hub_response(chosen, field_name="chosen_response")
    elif not chosen:
        raise ValueError("chosen_response must not be empty.")
    rejected = rejected_response.strip() if isinstance(rejected_response, str) and rejected_response.strip() else None
    if rejected is not None and rejected == chosen:
        raise ValueError("rejected_response and chosen_response must differ.")
    source = source.strip().lower()
    if source not in {"trusted_authoring", "evaluation"}:
        raise ValueError("source must be trusted_authoring or evaluation.")
    identity = {
        "curriculum_id": curriculum.curriculum_id,
        "curriculum_version": curriculum.version,
        "target_component": curriculum.target_component,
        "chapter_id": chapter.chapter_id,
        "user_request": request,
        "chosen_response": chosen,
        "rejected_response": rejected,
    }
    content_sha = _sha256_text(canonical_json(identity))
    return CurriculumLessonEvent(
        lesson_id="lesson-" + content_sha[:24],
        observed_at=_utc_now(),
        curriculum_id=curriculum.curriculum_id,
        curriculum_version=curriculum.version,
        target_component=curriculum.target_component,
        chapter_id=chapter.chapter_id,
        user_request=request,
        chosen_response=chosen,
        rejected_response=rejected,
        source=source,
        note=note.strip() if isinstance(note, str) and note.strip() else None,
        content_sha256=content_sha,
        dataset_eligible=False,
    )


def load_curriculum_lessons(path: Path = CURRICULUM_LESSONS_PATH) -> list[CurriculumLessonEvent]:
    return load_jsonl_models(path, CurriculumLessonEvent)


def load_curriculum_lesson_reviews(path: Path = CURRICULUM_LESSON_REVIEWS_PATH) -> list[CurriculumLessonReview]:
    return load_jsonl_models(path, CurriculumLessonReview)


def latest_curriculum_lesson_review_index(reviews: list[CurriculumLessonReview]) -> dict[str, CurriculumLessonReview]:
    result: dict[str, CurriculumLessonReview] = {}
    for review in reviews:
        result[review.lesson_id] = review
    return result


class CurriculumLessonRecorder:
    def __init__(self, *, path: Path = CURRICULUM_LESSONS_PATH) -> None:
        self.path = path.expanduser().resolve()
        self._lock = threading.Lock()

    def record(self, *, lesson: CurriculumLessonEvent) -> CurriculumLessonEvent:
        existing = {item.lesson_id: item for item in load_curriculum_lessons(self.path)}
        prior = existing.get(lesson.lesson_id)
        if prior is not None:
            prior_payload = prior.model_dump(exclude={"observed_at"})
            new_payload = lesson.model_dump(exclude={"observed_at"})
            if prior_payload == new_payload:
                return prior
            raise ValueError("lesson_id collision with different lesson content.")
        _append_model(path=self.path, value=lesson, lock=self._lock)
        return lesson


class CurriculumLessonReviewRecorder:
    def __init__(self, *, path: Path = CURRICULUM_LESSON_REVIEWS_PATH) -> None:
        self.path = path.expanduser().resolve()
        self._lock = threading.Lock()

    def record(self, *, lesson_id: str, decision: str, source: str, reason: str) -> CurriculumLessonReview:
        lesson_id = lesson_id.strip()
        decision = decision.strip().lower()
        source = source.strip().lower()
        reason = reason.strip()
        if not lesson_id:
            raise ValueError("lesson_id must not be empty.")
        if decision not in {"approve", "reject"}:
            raise ValueError("decision must be approve or reject.")
        if source not in {"trusted_review", "evaluation"}:
            raise ValueError("source must be trusted_review or evaluation.")
        if not reason:
            raise ValueError("reason must not be empty.")
        review = CurriculumLessonReview(
            review_id="lesson-review-" + uuid.uuid4().hex,
            observed_at=_utc_now(),
            lesson_id=lesson_id,
            decision=decision,
            source=source,
            reason=reason,
        )
        _append_model(path=self.path, value=review, lock=self._lock)
        return review


def _next_version(root: Path) -> str:
    if not root.exists():
        return "v000001"
    highest = 0
    for child in root.iterdir():
        if child.is_dir() and len(child.name) == 7 and child.name.startswith("v") and child.name[1:].isdigit():
            highest = max(highest, int(child.name[1:]))
    return f"v{highest + 1:06d}"


class CurriculumLessonDatasetBuilder:
    def __init__(self, *, lesson_path: Path = CURRICULUM_LESSONS_PATH,
                 review_path: Path = CURRICULUM_LESSON_REVIEWS_PATH,
                 dataset_root: Path = CURRICULUM_LESSON_DATASET_ROOT) -> None:
        self.lesson_path = lesson_path.expanduser().resolve()
        self.review_path = review_path.expanduser().resolve()
        self.dataset_root = dataset_root.expanduser().resolve()

    def build(self, *, curriculum_name: str, promoted_by: str,
              promotion_reason: str) -> CurriculumLessonBuildResult:
        curriculum = get_curriculum(curriculum_name)
        promoted_by = promoted_by.strip().lower()
        promotion_reason = promotion_reason.strip()
        if promoted_by not in {"trusted_review", "evaluation"}:
            raise ValueError("promoted_by must be trusted_review or evaluation.")
        if not promotion_reason:
            raise ValueError("promotion_reason must not be empty.")
        lessons = [
            item for item in load_curriculum_lessons(self.lesson_path)
            if item.curriculum_id == curriculum.curriculum_id
            and item.curriculum_version == curriculum.version
            and item.target_component == curriculum.target_component
        ]
        review_index = latest_curriculum_lesson_review_index(load_curriculum_lesson_reviews(self.review_path))
        exclusions: Counter[str] = Counter()
        records: list[CurriculumLessonDatasetRecord] = []
        for lesson in lessons:
            review = review_index.get(lesson.lesson_id)
            if review is None or review.decision != "approve":
                exclusions["lesson_review_not_approved"] += 1
                continue
            identity = {"lesson_id": lesson.lesson_id, "review_id": review.review_id, "content_sha256": lesson.content_sha256}
            records.append(CurriculumLessonDatasetRecord(
                record_id="lesson-dataset-" + _sha256_text(canonical_json(identity))[:24],
                created_at=_utc_now(),
                lesson_id=lesson.lesson_id,
                lesson_review_id=review.review_id,
                curriculum_id=lesson.curriculum_id,
                curriculum_version=lesson.curriculum_version,
                target_component=lesson.target_component,
                chapter_id=lesson.chapter_id,
                user_request=lesson.user_request,
                chosen_response=lesson.chosen_response,
                rejected_response=lesson.rejected_response,
                source_lesson_sha256=lesson.content_sha256,
                dataset_eligible=True,
                training_authorized=False,
            ))
        if not records:
            detail = ", ".join(f"{key}={value}" for key, value in sorted(exclusions.items()))
            raise ValueError("No approved curriculum lessons are available." + (f" Exclusions: {detail}" if detail else ""))
        target_root = self.dataset_root / curriculum.curriculum_id
        version = _next_version(target_root)
        target = target_root / version
        target.mkdir(parents=True, exist_ok=False)
        try:
            ordered = sorted(records, key=lambda item: item.record_id)
            content_sha256 = immutable_write_jsonl(target / "records.jsonl", ordered)
            manifest = CurriculumLessonManifest(
                version=version,
                created_at=_utc_now(),
                curriculum_id=curriculum.curriculum_id,
                curriculum_version=curriculum.version,
                target_component=curriculum.target_component,
                record_count=len(ordered),
                content_sha256=content_sha256,
                promoted_by=promoted_by,
                promotion_reason=promotion_reason,
                source_lesson_ids=sorted(item.lesson_id for item in ordered),
                exclusion_reason_counts=dict(sorted(exclusions.items())),
                training_authorized=False,
                promotion_authorized=False,
            )
            immutable_write_json(target / "manifest.json", manifest)
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            raise
        return CurriculumLessonBuildResult(manifest=manifest, output_directory=str(target))
