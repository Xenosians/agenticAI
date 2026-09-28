from __future__ import annotations

import argparse
import json
from pathlib import Path

from learning.training.membership import (
    DEFAULT_MEMBERSHIP_ROOT,
    build_training_membership_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an immutable curriculum-aware training membership plan. "
            "The command selects exact trusted member IDs but never trains "
            "or activates a model."
        )
    )

    parser.add_argument(
        "--curriculum",
        default="hub",
        help=(
            "Built-in curriculum name/id. "
            "Current built-ins: hub, developer."
        ),
    )

    parser.add_argument(
        "--state",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            DEFAULT_MEMBERSHIP_ROOT
        ),
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--seed",
        default="training-membership-v1",
    )

    parser.add_argument(
        "--max-code-members",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--max-repository-fraction",
        type=float,
        default=0.60,
    )

    parser.add_argument(
        "--json",
        action="store_true",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    result = build_training_membership_plan(
        curriculum_name=(
            args.curriculum
        ),
        state_path=(
            args.state
        ),
        output_root=(
            args.output_root
        ),
        validation_fraction=(
            args.validation_fraction
        ),
        seed=(
            args.seed
        ),
        max_code_members=(
            args.max_code_members
        ),
        max_repository_fraction=(
            args.max_repository_fraction
        ),
    )

    manifest = (
        result.manifest
    )

    if args.json:
        print(
            json.dumps(
                manifest.model_dump(
                    mode="json",
                    by_alias=True,
                ),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )

        return 0

    print(
        "Curriculum Training Membership"
    )
    print(
        "=============================="
    )
    print(
        f"Plan:         {manifest.plan_id}"
    )
    print(
        f"Target:       {manifest.target_component}"
    )
    print(
        f"Curriculum:   {manifest.curriculum_id}"
    )
    print(
        f"Chapter:      {manifest.current_chapter_id}"
    )
    print(
        f"Title:        {manifest.current_chapter_title}"
    )
    print(
        f"Members:      {manifest.member_count}"
    )
    print(
        f"Train:        {manifest.train_member_count}"
    )
    print(
        f"Validation:   {manifest.validation_member_count}"
    )
    print(
        "Roles:        "
        + json.dumps(
            manifest.role_counts,
            sort_keys=True,
        )
    )
    print(
        "Kinds:        "
        + json.dumps(
            manifest.member_kind_counts,
            sort_keys=True,
        )
    )
    print(
        "Reviewed:     "
        f"{manifest.reviewed_current_training_example_count}/"
        f"{manifest.required_reviewed_current_behavior_count}"
    )
    print(
        "  behavior:   "
        f"{manifest.reviewed_current_behavior_count}"
    )
    print(
        "  lessons:    "
        f"{manifest.reviewed_current_lesson_count}"
    )
    print(
        f"Hard cases:   {manifest.hard_case_count}"
    )
    print(
        "Hard+prov:    "
        f"{manifest.exact_provenance_hard_case_count}"
    )
    print(
        f"Ready:        {manifest.readiness_ready}"
    )

    if manifest.blocked_reasons:
        print(
            "Blocked by:   "
            + ", ".join(
                manifest.blocked_reasons
            )
        )

    print(
        f"Artifact:     {result.output_directory}"
    )
    print(
        "Training:     DISABLED"
    )
    print(
        "Promotion:    DISABLED"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
