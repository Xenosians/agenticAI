from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from learning.curriculum.catalog import get_curriculum
from learning.curriculum.lessons import (
    CurriculumLessonDatasetBuilder,
    CurriculumLessonRecorder,
    CurriculumLessonReviewRecorder,
    build_lesson,
    latest_curriculum_lesson_review_index,
    load_curriculum_lesson_reviews,
    load_curriculum_lessons,
)


def _import(args) -> int:
    path = args.file.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Lesson candidate file does not exist: {path}")
    recorder = CurriculumLessonRecorder()
    recorded = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"Lesson candidate at {path}:{line_number} must be an object.")
        lesson = build_lesson(
            curriculum_name=args.curriculum,
            chapter_id=args.chapter,
            user_request=str(raw.get("user_request", "")),
            chosen_response=str(raw.get("chosen_response", "")),
            rejected_response=raw.get("rejected_response"),
            source=args.source,
            note=raw.get("note"),
        )
        recorded.append(recorder.record(lesson=lesson))
    print("Curriculum Lessons Imported")
    print("===========================")
    print(f"Count:       {len(recorded)}")
    for item in recorded:
        print(f"{item.lesson_id} chapter={item.chapter_id}")
    print("Review:      REQUIRED")
    print("Training:    DISABLED")
    return 0


def _review(args) -> int:
    curriculum = get_curriculum(args.curriculum)
    chapter = curriculum.chapter(args.chapter)
    lessons = [item for item in load_curriculum_lessons()
               if item.curriculum_id == curriculum.curriculum_id
               and item.curriculum_version == curriculum.version
               and item.chapter_id == chapter.chapter_id]
    if len(lessons) != args.expected_count:
        raise ValueError(
            "Lesson count changed. "
            f"expected={args.expected_count} observed={len(lessons)}. "
            "Inspect the lesson ledger before reviewing."
        )
    recorder = CurriculumLessonReviewRecorder()
    reviews = [recorder.record(
        lesson_id=item.lesson_id,
        decision=args.decision,
        source=args.source,
        reason=args.reason,
    ) for item in lessons]
    print("Curriculum Lesson Reviews")
    print("=========================")
    print(f"Chapter:     {chapter.chapter_id}")
    print(f"Decision:    {args.decision}")
    print(f"Count:       {len(reviews)}")
    print("Training:    DISABLED")
    return 0


def _promote(args) -> int:
    result = CurriculumLessonDatasetBuilder().build(
        curriculum_name=args.curriculum,
        promoted_by=args.source,
        promotion_reason=args.reason,
    )
    manifest = result.manifest
    print("Curriculum Lesson Dataset")
    print("=========================")
    print(f"Version:       {manifest.version}")
    print(f"Curriculum:    {manifest.curriculum_id}")
    print(f"Records:       {manifest.record_count}")
    print(f"SHA-256:       {manifest.content_sha256}")
    print(f"Output:        {result.output_directory}")
    print("Training:      DISABLED")
    print("Activation:    DISABLED")
    return 0


def _status(args) -> int:
    curriculum = get_curriculum(args.curriculum)
    lessons = [item for item in load_curriculum_lessons() if item.curriculum_id == curriculum.curriculum_id]
    review_index = latest_curriculum_lesson_review_index(load_curriculum_lesson_reviews())
    chapter_counts = Counter(item.chapter_id for item in lessons)
    approved_counts = Counter(
        item.chapter_id for item in lessons
        if review_index.get(item.lesson_id) is not None
        and review_index[item.lesson_id].decision == "approve"
    )
    print("Curriculum Lesson Status")
    print("========================")
    print(f"Curriculum: {curriculum.curriculum_id}")
    for chapter in curriculum.chapters:
        print(
            f"{chapter.chapter_id}: lessons={chapter_counts[chapter.chapter_id]} "
            f"approved={approved_counts[chapter.chapter_id]} required={chapter.min_reviewed_examples}"
        )
    print("Training:   DISABLED")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Governed curriculum foundation lessons.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("import")
    p.add_argument("--curriculum", required=True)
    p.add_argument("--chapter", required=True)
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--source", choices=["trusted_authoring", "evaluation"], default="trusted_authoring")
    p.set_defaults(func=_import)

    p = sub.add_parser("review")
    p.add_argument("--curriculum", required=True)
    p.add_argument("--chapter", required=True)
    p.add_argument("decision", choices=["approve", "reject"])
    p.add_argument("--reason", required=True)
    p.add_argument("--source", choices=["trusted_review", "evaluation"], default="trusted_review")
    p.add_argument("--expected-count", type=int, required=True)
    p.set_defaults(func=_review)

    p = sub.add_parser("promote")
    p.add_argument("--curriculum", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--source", choices=["trusted_review", "evaluation"], default="trusted_review")
    p.set_defaults(func=_promote)

    p = sub.add_parser("status")
    p.add_argument("--curriculum", required=True)
    p.set_defaults(func=_status)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
