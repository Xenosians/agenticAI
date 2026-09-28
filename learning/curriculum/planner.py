from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from learning.continual.storage import canonical_json
from learning.curriculum.chapters import CurriculumDefinition
from learning.curriculum.mastery import (
    CurriculumState,
    validate_state_against_curriculum,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CurriculumMixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_chapter_fraction: float
    mastered_replay_fraction: float
    hard_case_fraction: float


class CurriculumTrainingPlan(BaseModel):
    """
    Curriculum planning artifact only.

    It never invokes a trainer and never activates an adapter.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    schema_name: str = Field(
        default="curriculum-training-plan.v1",
        alias="schema",
    )

    plan_id: str
    created_at: str

    curriculum_id: str
    curriculum_version: str
    target_component: str

    current_chapter_id: str | None
    current_chapter_title: str | None
    focus_tags: list[str] = Field(default_factory=list)

    mastered_replay_chapter_ids: list[str] = Field(default_factory=list)

    matching_failure_count: int = 0
    failure_code_counts: dict[str, int] = Field(default_factory=dict)
    matching_candidate_ids: list[str] = Field(default_factory=list)

    mixture: CurriculumMixture

    training_authorized: bool = False
    promotion_authorized: bool = False

    recommended_action: str


def load_failure_candidates(
    path: Path | None,
) -> list[dict[str, Any]]:
    if path is None:
        return []

    resolved = path.expanduser().resolve()

    if not resolved.exists():
        return []

    if not resolved.is_file():
        raise ValueError(
            f"Failure candidate path is not a file: {resolved}"
        )

    result: list[dict[str, Any]] = []

    for line_number, line in enumerate(
        resolved.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            value = json.loads(line)
        except Exception as exc:
            raise ValueError(
                f"Invalid failure candidate JSON at "
                f"{resolved}:{line_number}: {exc}"
            ) from exc

        if not isinstance(value, dict):
            raise ValueError(
                f"Failure candidate at {resolved}:{line_number} "
                "must be an object"
            )

        result.append(value)

    return result


def build_curriculum_training_plan(
    *,
    curriculum: CurriculumDefinition,
    state: CurriculumState,
    failure_candidates: list[dict[str, Any]] = (),
) -> CurriculumTrainingPlan:
    validate_state_against_curriculum(
        curriculum=curriculum,
        state=state,
    )

    if state.current_chapter_id is None:
        mixture = CurriculumMixture(
            current_chapter_fraction=0.0,
            mastered_replay_fraction=1.0,
            hard_case_fraction=0.0,
        )

        identity = {
            "curriculum_id": curriculum.curriculum_id,
            "version": curriculum.version,
            "state": state.model_dump(
                mode="json",
                by_alias=True,
            ),
            "matching_candidate_ids": [],
            "mixture": mixture.model_dump(mode="json"),
        }

        digest = hashlib.sha256(
            canonical_json(identity).encode("utf-8")
        ).hexdigest()

        return CurriculumTrainingPlan(
            plan_id=f"curriculum-plan-{digest[:24]}",
            created_at=_utc_now(),
            curriculum_id=curriculum.curriculum_id,
            curriculum_version=curriculum.version,
            target_component=curriculum.target_component,
            current_chapter_id=None,
            current_chapter_title=None,
            focus_tags=[],
            mastered_replay_chapter_ids=list(
                state.mastered_chapter_ids
            ),
            matching_failure_count=0,
            failure_code_counts={},
            matching_candidate_ids=[],
            mixture=mixture,
            training_authorized=False,
            promotion_authorized=False,
            recommended_action=(
                "Curriculum complete. Preserve mastered replay and "
                "continue held-out regression evaluation."
            ),
        )

    chapter = curriculum.chapter(
        state.current_chapter_id
    )

    allowed_failure_codes = set(
        chapter.failure_codes
    )

    matching: list[dict[str, Any]] = []

    for candidate in failure_candidates:
        failure_code = candidate.get(
            "failure_code"
        )

        if (
            isinstance(failure_code, str)
            and failure_code in allowed_failure_codes
        ):
            matching.append(candidate)

    failure_counts = Counter(
        str(item.get("failure_code"))
        for item in matching
    )

    matching_ids = [
        str(item.get("candidate_id"))
        for item in matching
        if item.get("candidate_id")
    ]

    replay_fraction = (
        0.30
        if state.mastered_chapter_ids
        else 0.0
    )

    hard_fraction = (
        0.10
        if matching
        else 0.0
    )

    current_fraction = (
        1.0
        - replay_fraction
        - hard_fraction
    )

    mixture = CurriculumMixture(
        current_chapter_fraction=current_fraction,
        mastered_replay_fraction=replay_fraction,
        hard_case_fraction=hard_fraction,
    )

    identity = {
        "curriculum_id": curriculum.curriculum_id,
        "version": curriculum.version,
        "state": state.model_dump(
            mode="json",
            by_alias=True,
        ),
        "matching_candidate_ids": sorted(matching_ids),
        "mixture": mixture.model_dump(mode="json"),
    }

    digest = hashlib.sha256(
        canonical_json(identity).encode("utf-8")
    ).hexdigest()

    if matching:
        next_action = (
            "Study the current chapter using reviewed chapter examples, "
            "mastered replay, and the matching hard-case queue. "
            "Training still requires the separate trusted dataset and "
            "trainer authorization gates."
        )
    else:
        next_action = (
            "Study the current chapter using reviewed examples and mastered "
            "replay. Collect or review more hard cases before increasing "
            "failure-focused sampling."
        )

    return CurriculumTrainingPlan(
        plan_id=f"curriculum-plan-{digest[:24]}",
        created_at=_utc_now(),
        curriculum_id=curriculum.curriculum_id,
        curriculum_version=curriculum.version,
        target_component=curriculum.target_component,
        current_chapter_id=chapter.chapter_id,
        current_chapter_title=chapter.title,
        focus_tags=list(chapter.focus_tags),
        mastered_replay_chapter_ids=list(
            state.mastered_chapter_ids
        ),
        matching_failure_count=len(matching),
        failure_code_counts=dict(failure_counts),
        matching_candidate_ids=matching_ids,
        mixture=mixture,
        training_authorized=False,
        promotion_authorized=False,
        recommended_action=next_action,
    )
