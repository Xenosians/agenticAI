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
    apply_mastery_evaluation,
    evaluate_current_chapter,
    initial_curriculum_state,
)
from learning.paths import RUNTIME_LEARNING_ROOT


DEFAULT_CURRICULUM_ROOT = (
    RUNTIME_LEARNING_ROOT
    / "continual"
    / "curriculum"
)


def _parse_metric(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Metric must use NAME=VALUE"
        )

    name, raw = value.split(
        "=",
        1,
    )

    name = name.strip()

    if not name:
        raise argparse.ArgumentTypeError(
            "Metric name must not be empty"
        )

    try:
        score = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Metric value must be numeric: {value}"
        ) from exc

    if not 0.0 <= score <= 1.0:
        raise argparse.ArgumentTypeError(
            "Metric values must be in [0, 1]"
        )

    return name, score


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate mastery of the current curriculum chapter and "
            "advance only when every gate passes. No model training occurs."
        )
    )

    parser.add_argument(
        "--curriculum",
        default="hub",
    )

    parser.add_argument(
        "--state",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        type=_parse_metric,
        help=(
            "Observed held-out metric as NAME=VALUE. "
            "Repeat for every required metric."
        ),
    )

    parser.add_argument(
        "--reviewed-examples",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--safety-pass-rate",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--source-evaluation-id",
        default=None,
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

    metrics = dict(
        args.metric
    )

    evaluation = evaluate_current_chapter(
        curriculum=curriculum,
        state=state,
        metrics=metrics,
        reviewed_example_count=(
            args.reviewed_examples
        ),
        safety_pass_rate=(
            args.safety_pass_rate
        ),
        source_evaluation_id=(
            args.source_evaluation_id
        ),
    )

    evaluation_path = (
        DEFAULT_CURRICULUM_ROOT
        / curriculum.curriculum_id
        / "evaluations"
        / f"{evaluation.evaluation_id}.json"
    )

    immutable_write_json(
        evaluation_path,
        evaluation,
    )

    next_state = apply_mastery_evaluation(
        curriculum=curriculum,
        state=state,
        evaluation=evaluation,
    )

    atomic_write_json(
        state_path,
        next_state,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "evaluation": evaluation.model_dump(
                        mode="json",
                        by_alias=True,
                    ),
                    "state": next_state.model_dump(
                        mode="json",
                        by_alias=True,
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        return 0

    print("Curriculum Mastery Gate")
    print("=======================")
    print(f"Curriculum: {curriculum.curriculum_id}")
    print(f"Chapter:    {evaluation.chapter_id}")
    print(f"Score:      {evaluation.mastery_score:.4f}")
    print(f"Mastered:   {evaluation.mastered}")
    print(f"Safety:     {evaluation.safety_pass_rate:.4f}")

    if evaluation.block_reasons:
        print(
            "Blocked by: "
            + ", ".join(
                evaluation.block_reasons
            )
        )

    print(f"Artifact:   {evaluation_path}")
    print(f"State:      {state_path}")
    print(
        "Next:       "
        + (
            next_state.current_chapter_id
            if next_state.current_chapter_id
            else "curriculum-complete"
        )
    )
    print("Training:   NOT PERFORMED")
    print("Promotion:  NOT PERFORMED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
