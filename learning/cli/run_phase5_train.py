from __future__ import annotations

import argparse
from pathlib import Path

from config import Settings

from learning.paths import REPOSITORY_ROOT
from learning.training.phase5_hybrid_qlora import (
    Phase5TrainingSettings,
    train_phase5_hub_adapter,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run curriculum-gated Hub SFT + DPO QLoRA training. "
            "The resulting adapter is registered but never activated."
        )
    )

    parser.add_argument(
        "--materialization-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--allow-training",
        action="store_true",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--max-optimizer-steps",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--compute-dtype",
        choices=[
            "bfloat16",
            "float16",
        ],
        default="bfloat16",
    )

    args = parser.parse_args()

    if not args.allow_training:
        print(
            "TRAINING REFUSED: pass --allow-training "
            "to execute real optimizer steps."
        )
        return 2

    settings = Settings()

    profile = settings.require_model_profile(
        settings.hub_model_key
    )

    recipe = Phase5TrainingSettings(
        compute_dtype=args.compute_dtype,
        learning_rate=args.learning_rate,
        beta=args.beta,
        epochs=args.epochs,
        gradient_accumulation_steps=(
            args.gradient_accumulation
        ),
        max_length=args.max_length,
        max_optimizer_steps=(
            args.max_optimizer_steps
        ),
        checkpoint_every_steps=(
            args.checkpoint_every
        ),
        early_stop_patience=args.patience,
    )

    manifest = train_phase5_hub_adapter(
        materialization_directory=(
            args.materialization_dir
        ),
        settings=recipe,
        allow_training=True,
        backend=profile.backend,
        agent_directory=(
            REPOSITORY_ROOT
            / "subagents"
            / "agents"
        ),
    )

    print("Phase-5 Hub Training")
    print("====================")
    print(f"Run:          {manifest.run_id}")
    print(f"Steps:        {manifest.optimizer_steps}")
    print(f"SFT steps:    {manifest.sft_optimizer_steps}")
    print(f"DPO steps:    {manifest.dpo_optimizer_steps}")
    print(f"Best score:   {manifest.best_score:.6f}")
    print(f"Guard:        {manifest.fit_diagnosis}")
    print(f"Adapter:      {manifest.adapter_directory}")
    print(
        f"Checkpoint:   {manifest.registered_checkpoint_id}"
    )
    print("Activation:    DISABLED")
    print()
    print(
        "Next: evaluate this checkpoint before promotion."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
