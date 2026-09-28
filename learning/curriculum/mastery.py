from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from learning.curriculum.chapters import (
    CurriculumChapter,
    CurriculumDefinition,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CurriculumState(BaseModel):
    """
    Durable curriculum progression state.

    This state grants no training or runtime authority.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    schema_name: str = Field(
        default="curriculum-state.v1",
        alias="schema",
    )

    curriculum_id: str
    curriculum_version: str
    target_component: str

    updated_at: str

    current_chapter_id: str | None
    mastered_chapter_ids: list[str] = Field(default_factory=list)

    chapter_attempt_counts: dict[str, int] = Field(default_factory=dict)
    best_mastery_scores: dict[str, float] = Field(default_factory=dict)

    last_evaluation_id: str | None = None
    status: str = "learning"


class ChapterMasteryEvaluation(BaseModel):
    """
    Immutable mastery-gate result for one curriculum chapter.

    Mastery requires:
        every required metric threshold
        enough reviewed examples
        the curriculum safety threshold
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    schema_name: str = Field(
        default="chapter-mastery-evaluation.v1",
        alias="schema",
    )

    evaluation_id: str
    created_at: str

    curriculum_id: str
    curriculum_version: str
    target_component: str

    chapter_id: str
    chapter_ordinal: int

    source_evaluation_id: str | None = None

    metrics: dict[str, float] = Field(default_factory=dict)
    reviewed_example_count: int = 0
    safety_pass_rate: float = 0.0

    checks: dict[str, bool] = Field(default_factory=dict)
    mastery_score: float = 0.0
    mastered: bool = False

    block_reasons: list[str] = Field(default_factory=list)


def initial_curriculum_state(
    curriculum: CurriculumDefinition,
) -> CurriculumState:
    first = curriculum.chapters[0]

    return CurriculumState(
        curriculum_id=curriculum.curriculum_id,
        curriculum_version=curriculum.version,
        target_component=curriculum.target_component,
        updated_at=_utc_now(),
        current_chapter_id=first.chapter_id,
        mastered_chapter_ids=[],
        chapter_attempt_counts={},
        best_mastery_scores={},
        last_evaluation_id=None,
        status="learning",
    )


def validate_state_against_curriculum(
    *,
    curriculum: CurriculumDefinition,
    state: CurriculumState,
) -> None:
    if state.curriculum_id != curriculum.curriculum_id:
        raise ValueError("Curriculum state belongs to a different curriculum")

    if state.curriculum_version != curriculum.version:
        raise ValueError(
            "Curriculum state version does not match the current curriculum"
        )

    if state.target_component != curriculum.target_component:
        raise ValueError("Curriculum state target component mismatch")

    known_ids = {item.chapter_id for item in curriculum.chapters}

    if not set(state.mastered_chapter_ids).issubset(known_ids):
        raise ValueError("Curriculum state contains unknown mastered chapters")

    if (
        state.current_chapter_id is not None
        and state.current_chapter_id not in known_ids
    ):
        raise ValueError("Curriculum state contains an unknown current chapter")


def _metric_checks(
    *,
    chapter: CurriculumChapter,
    metrics: dict[str, float],
) -> tuple[dict[str, bool], list[str], float]:
    checks: dict[str, bool] = {}
    reasons: list[str] = []
    normalized_scores: list[float] = []

    for name, threshold in chapter.mastery_thresholds.items():
        observed = metrics.get(name)

        passed = (
            observed is not None
            and float(observed) >= float(threshold)
        )

        checks[f"metric:{name}"] = passed

        if not passed:
            reasons.append(
                f"metric-below-threshold:{name}"
            )

        if observed is None:
            normalized_scores.append(0.0)
        elif threshold <= 0.0:
            normalized_scores.append(1.0)
        else:
            normalized_scores.append(
                min(
                    1.0,
                    max(
                        0.0,
                        float(observed) / float(threshold),
                    ),
                )
            )

    score = (
        sum(normalized_scores) / len(normalized_scores)
        if normalized_scores
        else 0.0
    )

    return checks, reasons, score


def evaluate_current_chapter(
    *,
    curriculum: CurriculumDefinition,
    state: CurriculumState,
    metrics: dict[str, float],
    reviewed_example_count: int,
    safety_pass_rate: float,
    source_evaluation_id: str | None = None,
) -> ChapterMasteryEvaluation:
    validate_state_against_curriculum(
        curriculum=curriculum,
        state=state,
    )

    if state.current_chapter_id is None:
        raise ValueError("Curriculum is already complete")

    if reviewed_example_count < 0:
        raise ValueError("reviewed_example_count must be non-negative")

    if not 0.0 <= float(safety_pass_rate) <= 1.0:
        raise ValueError("safety_pass_rate must be in [0, 1]")

    chapter = curriculum.chapter(
        state.current_chapter_id
    )

    checks, reasons, score = _metric_checks(
        chapter=chapter,
        metrics=metrics,
    )

    examples_ok = (
        reviewed_example_count
        >= chapter.min_reviewed_examples
    )
    safety_ok = (
        float(safety_pass_rate)
        >= curriculum.required_safety_pass_rate
    )

    checks["minimum_reviewed_examples"] = examples_ok
    checks["safety_gate"] = safety_ok

    if not examples_ok:
        reasons.append("insufficient-reviewed-examples")

    if not safety_ok:
        reasons.append("safety-regression")

    mastered = all(checks.values())

    return ChapterMasteryEvaluation(
        evaluation_id=f"mastery-{uuid.uuid4().hex}",
        created_at=_utc_now(),
        curriculum_id=curriculum.curriculum_id,
        curriculum_version=curriculum.version,
        target_component=curriculum.target_component,
        chapter_id=chapter.chapter_id,
        chapter_ordinal=chapter.ordinal,
        source_evaluation_id=source_evaluation_id,
        metrics={
            str(name): float(value)
            for name, value in metrics.items()
        },
        reviewed_example_count=reviewed_example_count,
        safety_pass_rate=float(safety_pass_rate),
        checks=checks,
        mastery_score=score,
        mastered=mastered,
        block_reasons=list(dict.fromkeys(reasons)),
    )


def apply_mastery_evaluation(
    *,
    curriculum: CurriculumDefinition,
    state: CurriculumState,
    evaluation: ChapterMasteryEvaluation,
) -> CurriculumState:
    validate_state_against_curriculum(
        curriculum=curriculum,
        state=state,
    )

    if state.current_chapter_id is None:
        raise ValueError("Curriculum is already complete")

    if evaluation.curriculum_id != curriculum.curriculum_id:
        raise ValueError("Mastery evaluation belongs to another curriculum")

    if evaluation.curriculum_version != curriculum.version:
        raise ValueError("Mastery evaluation curriculum version mismatch")

    if evaluation.chapter_id != state.current_chapter_id:
        raise ValueError(
            "Mastery evaluation does not target the current chapter"
        )

    attempts = dict(state.chapter_attempt_counts)
    attempts[evaluation.chapter_id] = (
        attempts.get(evaluation.chapter_id, 0) + 1
    )

    best_scores = dict(state.best_mastery_scores)
    best_scores[evaluation.chapter_id] = max(
        best_scores.get(evaluation.chapter_id, 0.0),
        evaluation.mastery_score,
    )

    mastered_ids = list(state.mastered_chapter_ids)
    current_id: str | None = state.current_chapter_id
    status = state.status

    if evaluation.mastered:
        if evaluation.chapter_id not in mastered_ids:
            mastered_ids.append(evaluation.chapter_id)

        next_chapter = curriculum.next_chapter(
            evaluation.chapter_id
        )

        if next_chapter is None:
            current_id = None
            status = "completed"
        else:
            current_id = next_chapter.chapter_id
            status = "learning"

    return CurriculumState(
        curriculum_id=state.curriculum_id,
        curriculum_version=state.curriculum_version,
        target_component=state.target_component,
        updated_at=_utc_now(),
        current_chapter_id=current_id,
        mastered_chapter_ids=mastered_ids,
        chapter_attempt_counts=attempts,
        best_mastery_scores=best_scores,
        last_evaluation_id=evaluation.evaluation_id,
        status=status,
    )
