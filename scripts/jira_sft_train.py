from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from learning.training.jira_sft import (  # noqa: E402
    train_jira_sft,
)


DEFAULT_CORPUS = (
    PROJECT_ROOT
    / ".runtime"
    / "learning"
    / "jira-sft"
    / "v1"
)

DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/c/project/agenticaiPersonal/Models/BLOOMZ-560M-Jira-SFT"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train a bounded Jira BLOOMZ QLoRA adapter and merge it "
            "into a standalone candidate checkpoint."
        )
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--worker-prompt-profile",
        choices=("standard", "compact", "compact_grounded"),
        default=None,
        help=(
            "Bind an explicitly prepared candidate corpus prompt without changing "
            "serving configuration. Must match the corpus manifest."
        ),
    )

    parser.add_argument(
        "--allow-training",
        action="store_true",
        help=(
            "Required explicit authorization for real optimizer steps."
        ),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=160,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=1280,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--checkpoint-interval", type=int, default=10,
        help="Optimizer steps between validation checkpoints.",
    )

    parser.add_argument(
        "--early-stopping-patience", type=int, default=3,
        help="Stop after this many checkpoints without a lower validation loss.",
    )

    parser.add_argument(
        "--resume-run", type=Path, default=None,
        help="Continue an interrupted run directory from its newest checkpoint.",
    )

    args = parser.parse_args()

    manifest = train_jira_sft(
        project_root=PROJECT_ROOT,
        corpus_directory=args.corpus,
        output_root=args.output_root,
        allow_training=args.allow_training,
        max_steps=args.max_steps,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        worker_prompt_profile=args.worker_prompt_profile,
        checkpoint_interval=args.checkpoint_interval,
        early_stopping_patience=args.early_stopping_patience,
        resume_run_directory=args.resume_run,
    )

    manifest_path = (
        Path(
            manifest.output_directory
        )
        / "training-manifest.json"
    )

    print()
    print(
        "JIRA SFT TRAINING: PASS"
    )
    print(
        "training manifest:",
        manifest_path,
    )
    print(
        "adapter:",
        manifest.adapter_directory,
    )
    print(
        "merged model:",
        manifest.merged_model_directory,
    )
    print(
        "adapter sha256:",
        manifest.adapter_sha256,
    )
    print(
        "merged model sha256:",
        manifest.merged_model_sha256,
    )
    print(
        "trainable ratio:",
        manifest.trainable_ratio,
    )
    print()
    print(
        "The production Jira specialist was NOT modified."
    )
    print(
        "Register the merged checkpoint as the jira-func-candidate profile, then "
        "run the isolated Jira model gate on the development and held-out case sets."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
