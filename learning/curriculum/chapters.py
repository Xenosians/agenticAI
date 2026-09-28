from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


CurriculumTarget = Literal[
    "hub",
    "developer-specialist",
    "jira-specialist",
    "account-specialist",
    "access-specialist",
    "asset-specialist",
    "knowledge-specialist",
    "atlassian-specialist",
    "ticket-specialist",
]


class CurriculumChapter(BaseModel):
    """
    One ordered curriculum chapter.

    A chapter is a semantic learning unit, not a raw token range.

    Examples:
        structured output
        scope vs primary resource
        grounded argument binding
        Git read-operation identity
        single-file patching

    failure_codes connect live runtime evidence to the chapter that
    should receive additional attention. They do not make evidence
    training-eligible.
    """

    model_config = ConfigDict(extra="forbid")

    chapter_id: str
    ordinal: int = Field(ge=1)
    title: str
    description: str

    focus_tags: list[str] = Field(default_factory=list)
    failure_codes: list[str] = Field(default_factory=list)

    mastery_thresholds: dict[str, float] = Field(default_factory=dict)
    min_reviewed_examples: int = Field(default=1, ge=1)


class CurriculumDefinition(BaseModel):
    """
    Ordered curriculum for one model/component.

    The curriculum is intentionally independent from any trainer.
    It decides what may be studied next; it never mutates weights.
    """

    model_config = ConfigDict(extra="forbid")

    curriculum_id: str
    version: str
    target_component: CurriculumTarget

    required_safety_pass_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )

    chapters: list[CurriculumChapter]

    @model_validator(mode="after")
    def validate_curriculum(self):
        if not self.curriculum_id.strip():
            raise ValueError("curriculum_id must not be empty")

        if not self.version.strip():
            raise ValueError("version must not be empty")

        if not self.chapters:
            raise ValueError("curriculum must contain at least one chapter")

        expected = list(range(1, len(self.chapters) + 1))
        observed = [item.ordinal for item in self.chapters]

        if observed != expected:
            raise ValueError(
                "Curriculum chapter ordinals must be contiguous and ordered "
                f"from 1. expected={expected!r}, observed={observed!r}"
            )

        ids = [item.chapter_id for item in self.chapters]
        if len(ids) != len(set(ids)):
            raise ValueError("Curriculum chapter_id values must be unique")

        for chapter in self.chapters:
            if not chapter.chapter_id.strip():
                raise ValueError("chapter_id must not be empty")

            if not chapter.title.strip():
                raise ValueError(
                    f"Curriculum chapter '{chapter.chapter_id}' has no title"
                )

            if not chapter.mastery_thresholds:
                raise ValueError(
                    f"Curriculum chapter '{chapter.chapter_id}' "
                    "must define mastery thresholds"
                )

            for name, threshold in chapter.mastery_thresholds.items():
                if not name.strip():
                    raise ValueError("mastery metric name must not be empty")
                if not 0.0 <= float(threshold) <= 1.0:
                    raise ValueError(
                        f"Mastery threshold '{name}' must be in [0, 1]"
                    )

        return self

    def chapter(self, chapter_id: str) -> CurriculumChapter:
        for chapter in self.chapters:
            if chapter.chapter_id == chapter_id:
                return chapter

        raise ValueError(
            f"Unknown chapter_id for curriculum '{self.curriculum_id}': "
            f"{chapter_id}"
        )

    def next_chapter(
        self,
        chapter_id: str,
    ) -> CurriculumChapter | None:
        current = self.chapter(chapter_id)

        next_ordinal = current.ordinal + 1

        for chapter in self.chapters:
            if chapter.ordinal == next_ordinal:
                return chapter

        return None
