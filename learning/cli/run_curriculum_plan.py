from __future__ import annotations

import argparse
import json
from pathlib import Path

from learning.continual.storage import (
    atomic_write_json,
    immutable_write_json,
    read_json_model,
)
from learning.curriculum.catalog import get_curriculum
from learning.curriculum.mastery import (
    CurriculumState,
    initial_curriculum_state,
)
from learning.curriculum.planner import (
    build_curriculum_training_plan,
    load_failure_candidates,
)
from learning.paths import RUNTIME_LEARNING_ROOT


DEFAULT_CURRICULUM_ROOT = (
    RUNTIME_LEARNING_ROOT
    / "continual"
    / "curriculum"
)


def _latest_failure_candidates() -> Path | None:
    root = (
        RUNTIME_LEARNING_ROOT
        / "continual"
        / "failure-mining"
    )

    if not root.is_dir():
        return None

    candidates = list(
        root.glob(
            "*/candidates.jsonl"
        )
    )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item: item.stat().st_mtime,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build one chapter-based continual-learning plan. "
            "This command does NOT train or activate a model."
        )
    )

    parser.add_argument(
        "--curriculum",
        default="hub",
        help=(
            "Built-in curriculum name or curriculum id. "
            "Examples: hub, developer."
        ),
    )

    parser.add_argument(
        "--state",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--failure-candidates",
        type=Path,
        default=None,
        help=(
            "Failure-mining candidates.jsonl. When omitted, the latest "
            "runtime failure-mining artifact is used if one exists."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            DEFAULT_CURRICULUM_ROOT
            / "plans"
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    curriculum = get_curriculum(
        args.curriculum
    )

    state_path = (
        args.state
        if args.state is not None
        else (
            DEFAULT_CURRICULUM_ROOT
            / curriculum.curriculum_id
            / "state.json"
        )
    ).expanduser().resolve()

    if state_path.is_file():
        state = read_json_model(
            state_path,
            CurriculumState,
        )
    else:
        state = initial_curriculum_state(
            curriculum
        )
        atomic_write_json(
            state_path,
            state,
        )

    candidate_path = (
        args.failure_candidates
        if args.failure_candidates is not None
        else _latest_failure_candidates()
    )

    failure_candidates = (
        load_failure_candidates(
            candidate_path
        )
    )

    plan = build_curriculum_training_plan(
        curriculum=curriculum,
        state=state,
        failure_candidates=failure_candidates,
    )

    output_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    plan_path = (
        output_root
        / plan.plan_id
        / "plan.json"
    )

    if not plan_path.exists():
        immutable_write_json(
            plan_path,
            plan,
        )

    if args.json:
        print(
            json.dumps(
                plan.model_dump(
                    mode="json",
                    by_alias=True,
                ),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        return 0

    print("Chapter-Based Continual Learning Plan")
    print("====================================")
    print(f"Curriculum:   {plan.curriculum_id}")
    print(f"Target:       {plan.target_component}")
    print(f"Chapter:      {plan.current_chapter_id}")
    print(f"Title:        {plan.current_chapter_title}")
    print(
        "Mastered:     "
        + (
            ", ".join(
                plan.mastered_replay_chapter_ids
            )
            if plan.mastered_replay_chapter_ids
            else "(none)"
        )
    )
    print(
        f"Hard cases:   {plan.matching_failure_count}"
    )
    print(
        "Mixture:      "
        f"current={plan.mixture.current_chapter_fraction:.2f} "
        f"replay={plan.mixture.mastered_replay_fraction:.2f} "
        f"hard={plan.mixture.hard_case_fraction:.2f}"
    )
    print(f"Plan:         {plan_path}")
    print("Training:     DISABLED")
    print("Promotion:    DISABLED")
    print(f"Next:         {plan.recommended_action}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
