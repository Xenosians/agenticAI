from __future__ import annotations

import argparse
import json

from pathlib import Path

from config import Settings

from config.path_portability import (
    resolve_portable_path,
)

from learning.continual.checkpoints import (
    AdapterCheckpointStore,
)

from learning.training.phase5_hybrid_qlora import (
    Phase5TrainingSettings,
    train_phase5_adapter,
)

from learning.training.phase5_materializer import (
    Phase5MaterializationManifest,
)


def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Train a candidate developer QLoRA adapter from a "
            "governed developer corpus materialization. "
            "The candidate is registered but NEVER activated."
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
        default=2,
    )

    parser.add_argument(
        "--max-optimizer-steps",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
    )

    parser.add_argument(
        "--compute-dtype",
        choices=[
            "bfloat16",
            "float16",
        ],
        default="bfloat16",
    )

    parser.add_argument(
        "--no-seed-active",
        action="store_true",
        help=(
            "Start from the frozen base rather than the currently "
            "promoted compatible adapter."
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.allow_training:

        print(
            "TRAINING REFUSED: pass --allow-training "
            "to execute real optimizer steps."
        )

        return 2

    directory = (
        resolve_portable_path(
            args.materialization_dir,
        )
    )

    materialization = (
        Phase5MaterializationManifest
        .model_validate_json(
            (
                directory
                / "manifest.json"
            ).read_text(
                encoding="utf-8"
            )
        )
    )

    if (
        materialization.target_component
        != "developer-specialist"
        or materialization
        .evaluation_contract
        != "developer_sft_loss"
    ):
        raise ValueError(
            "This CLI accepts only developer-specialist "
            "developer_sft_loss materializations."
        )

    settings = Settings()

    profile = (
        settings
        .require_model_profile(
            materialization
            .target_model_key
        )
    )

    seed_adapter = None
    seed_checkpoint_id = None

    if not args.no_seed_active:

        store = (
            AdapterCheckpointStore()
        )

        active = store.active()

        if active is not None:

            candidate_seed = (
                store.verify_adapter(
                    active.checkpoint_id
                )
            )

            if (
                candidate_seed
                .target_model_key
                == materialization
                .target_model_key
            ):

                seed_adapter = (
                    resolve_portable_path(
                        candidate_seed
                        .adapter_directory
                    )
                )

                seed_checkpoint_id = (
                    candidate_seed
                    .checkpoint_id
                )

    recipe = (
        Phase5TrainingSettings(
            compute_dtype=(
                args.compute_dtype
            ),

            learning_rate=(
                args.learning_rate
            ),

            epochs=args.epochs,

            gradient_accumulation_steps=(
                args.gradient_accumulation
            ),

            max_length=(
                args.max_length
            ),

            max_optimizer_steps=(
                args.max_optimizer_steps
            ),

            checkpoint_every_steps=(
                args.checkpoint_every
            ),

            early_stop_patience=(
                args.patience
            ),
        )
    )

    result = train_phase5_adapter(
        materialization_directory=(
            directory
        ),

        settings=recipe,

        allow_training=True,

        backend=(
            profile.backend
        ),

        agent_directory=(
            settings.agents_dir
        ),

        seed_adapter_directory=(
            seed_adapter
        ),
    )

    if args.json:

        payload = (
            result.model_dump(
                mode="json"
            )
        )

        payload[
            "seed_checkpoint_id"
        ] = (
            seed_checkpoint_id
        )

        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )

        return 0

    print(
        "Developer Corpus QLoRA"
    )

    print(
        "======================"
    )

    print(
        "run="
        + result.run_id
    )

    print(
        "seed_checkpoint="
        + str(
            seed_checkpoint_id
        )
    )

    print(
        "target="
        + result.target_component
    )

    print(
        "evaluation="
        + result.evaluation_contract
    )

    print(
        "optimizer_steps="
        + str(
            result.optimizer_steps
        )
    )

    print(
        "sft_steps="
        + str(
            result.sft_optimizer_steps
        )
    )

    print(
        "best_score="
        + str(
            result.best_score
        )
    )

    print(
        "guard="
        + result.fit_diagnosis
    )

    print(
        "checkpoint="
        + result.registered_checkpoint_id
    )

    print(
        "adapter="
        + result.adapter_directory
    )

    print(
        "activation=DISABLED"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
