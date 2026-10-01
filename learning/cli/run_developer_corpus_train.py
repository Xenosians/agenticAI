from __future__ import annotations

import argparse
import json
import os

from pathlib import Path

from learning.cli.crashlog import (
    run_with_crashlog,
)

from config import Settings

from config.path_portability import (
    resolve_portable_path,
)

from learning.continual.checkpoints import (
    AdapterCheckpointStore,
)

from learning.cli.reporting import (
    render_developer_training_summary,
)

from learning.training.hub_hybrid_qlora import (
    HubTrainingSettings,
    train_phase5_adapter,
)

from learning.training.hub_training_materializer import (
    HubTrainingMaterializationManifest,
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
        help=(
            "Print the full machine-readable run manifest. "
            "For normal interactive use, omit this flag and "
            "use the concise developer summary."
        ),
    )

    parser.add_argument(
        "--cuda-telemetry",
        action="store_true",
        help=(
            "Print detailed CUDA memory snapshots around "
            "backward and optimizer steps. Intended for "
            "GPU debugging, not normal training output."
        ),
    )

    args = parser.parse_args()

    if args.cuda_telemetry:
        os.environ[
            "PHASE5_CUDA_TELEMETRY"
        ] = "1"

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
        HubTrainingMaterializationManifest
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
        HubTrainingSettings(
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
        render_developer_training_summary(
            result,
            seed_checkpoint_id=(
                seed_checkpoint_id
            ),
        )
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        run_with_crashlog(
            "developer-corpus-train",
            main,
        )
    )
