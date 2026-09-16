from __future__ import annotations

import argparse
import json
import tempfile

from pathlib import (
    Path,
)

from learning.dpo_materializer import (
    SpecialistDpoMaterializer,
)

from learning.dpo_qlora import (
    SpecialistDpoQloraDryRun,
    SpecialistDpoQloraSettings,
)

from learning.types import (
    DatasetPromotion,
    PreferenceDatasetRecord,
    PreferenceOption,
)


# ============================================================
# PROJECT PATHS
# ============================================================


PROJECT_ROOT = (
    Path(
        __file__
    )
    .resolve()
    .parents[
        1
    ]
)


DEFAULT_AGENT_PATH = (
    PROJECT_ROOT
    / "subagents"
    / "agents"
    / "account-specialist.md"
)


DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / ".runtime"
    / "learning"
    / "training-smoke"
)


DEFAULT_MODEL_PATH = (
    Path(
        "/mnt/c/project/"
        "agenticaiPersonal/"
        "Models/"
        "qwen2.5-0.5b-funccall"
    )
)


# Measured against the real account-specialist runtime prompt:
#
#   largest prompt:        1157 tokens
#   largest full sequence: 1178 tokens
#
# 1536 leaves 358 tokens of headroom for the current target.
#
# This is intentionally target-specific and must not be assumed
# safe for every future specialist.
DEFAULT_MAX_LENGTH = (
    1536
)


# ============================================================
# SYNTHETIC TRUSTED RECORDS
# ============================================================


def _promotion(
) -> DatasetPromotion:

    return (
        DatasetPromotion(
            promoted_at=(
                "2026-09-16T00:00:00+00:00"
            ),

            promoted_by=(
                "trusted_review"
            ),

            reason=(
                "Synthetic Phase 4A hardware smoke test."
            ),
        )
    )


def _record(
    *,
    record_id: str,
    user_request: str,
    task_instructions: str,
    rejected_user_id: str,
    chosen_user_id: str,
) -> PreferenceDatasetRecord:

    return (
        PreferenceDatasetRecord(
            record_id=(
                record_id
            ),

            source_example_id=(
                f"example-{record_id}"
            ),

            trajectory_id=(
                f"trajectory-{record_id}"
            ),

            correction_id=(
                f"correction-{record_id}"
            ),

            task_id=(
                f"task-{record_id}"
            ),

            task_instructions=(
                task_instructions
            ),

            source=(
                "trusted_review"
            ),

            correction_type=(
                "account_identifier"
            ),

            user_request=(
                user_request
            ),

            rejected=(
                PreferenceOption(
                    agent=(
                        "account-specialist"
                    ),

                    tool=(
                        "account_status"
                    ),

                    arguments={
                        "user_id":
                            rejected_user_id
                    },

                    answer=None,
                )
            ),

            chosen=(
                PreferenceOption(
                    agent=(
                        "account-specialist"
                    ),

                    tool=(
                        "account_status"
                    ),

                    arguments={
                        "user_id":
                            chosen_user_id
                    },

                    answer=None,
                )
            ),

            promotion=(
                _promotion()
            ),

            dataset_eligible=True,
        )
    )


def _training_records(
) -> list[
    PreferenceDatasetRecord
]:

    return [
        _record(
            record_id=(
                "smoke-train-1"
            ),

            user_request=(
                "Is jdoe locked?"
            ),

            task_instructions=(
                "Check the account state for jdoe."
            ),

            rejected_user_id=(
                "asmith"
            ),

            chosen_user_id=(
                "jdoe"
            ),
        ),

        _record(
            record_id=(
                "smoke-train-2"
            ),

            user_request=(
                "Check the account status for asmith."
            ),

            task_instructions=(
                "Inspect the requested directory account."
            ),

            rejected_user_id=(
                "jdoe"
            ),

            chosen_user_id=(
                "asmith"
            ),
        ),
    ]


def _validation_records(
) -> list[
    PreferenceDatasetRecord
]:

    return [
        _record(
            record_id=(
                "smoke-validation-1"
            ),

            user_request=(
                "Tell me the current account state "
                "for mbrown."
            ),

            task_instructions=(
                "Retrieve account status for mbrown."
            ),

            rejected_user_id=(
                "jdoe"
            ),

            chosen_user_id=(
                "mbrown"
            ),
        )
    ]


# ============================================================
# CLI
# ============================================================


def build_parser(
) -> argparse.ArgumentParser:

    parser = (
        argparse.ArgumentParser(
            description=(
                "Run the Phase 4A specialist DPO/QLoRA "
                "hardware smoke test. The real local model "
                "is loaded in 4-bit NF4, LoRA is attached, "
                "TRL DPOTrainer is constructed, datasets are "
                "processed, one DPO batch is collated, and "
                "the optimizer is constructed. "
                "trainer.train() is never called."
            )
        )
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=(
            DEFAULT_MODEL_PATH
        ),
    )

    parser.add_argument(
        "--agent-path",
        type=Path,
        default=(
            DEFAULT_AGENT_PATH
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            DEFAULT_OUTPUT_ROOT
        ),
    )

    parser.add_argument(
        "--expected-model-key",
        default=(
            "qwen2.5-0.5b-funccall"
        ),
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=(
            DEFAULT_MAX_LENGTH
        ),
        help=(
            "Maximum DPO sequence length. "
            "Default 1536 is validated specifically "
            "for the current account-specialist prompt."
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
    )

    return parser


# ============================================================
# RUN
# ============================================================


def main(
) -> int:

    args = (
        build_parser()
        .parse_args()
    )

    model_path = (
        args.model_path
        .expanduser()
        .resolve()
    )

    agent_path = (
        args.agent_path
        .expanduser()
        .resolve()
    )

    output_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    if not model_path.is_dir():

        print(
            "ERROR: model directory does not exist: "
            f"{model_path}"
        )

        return 1

    if not agent_path.is_file():

        print(
            "ERROR: agent definition does not exist: "
            f"{agent_path}"
        )

        return 1

    print(
        "Phase 4A DPO/QLoRA Hardware Smoke"
    )

    print(
        "================================="
    )

    print(
        f"Model: {model_path}"
    )

    print(
        f"Agent: {agent_path}"
    )

    print(
        f"Max length: {args.max_length}"
    )

    print(
        "Training execution: DISABLED"
    )

    print()

    try:

        with tempfile.TemporaryDirectory(
            prefix=(
                "agentic-dpo-smoke-"
            )
        ) as temporary_directory:

            temporary_root = (
                Path(
                    temporary_directory
                )
            )

            materialized_root = (
                temporary_root
                / "materialized"
            )

            split_id = (
                "synthetic-hardware-smoke"
            )

            # =================================================
            # REAL PHASE 4A.2 MATERIALIZATION
            # =================================================

            materializer = (
                SpecialistDpoMaterializer(
                    agent_definition_path=(
                        agent_path
                    ),

                    output_root=(
                        materialized_root
                    ),
                )
            )

            train_result = (
                materializer.build(
                    records=(
                        _training_records()
                    ),

                    source_split_id=(
                        split_id
                    ),

                    source_partition=(
                        "train"
                    ),

                    source_sha256=(
                        "synthetic-train-source"
                    ),
                )
            )

            validation_result = (
                materializer.build(
                    records=(
                        _validation_records()
                    ),

                    source_split_id=(
                        split_id
                    ),

                    source_partition=(
                        "validation"
                    ),

                    source_sha256=(
                        "synthetic-validation-source"
                    ),
                )
            )

            # =================================================
            # REAL PHASE 4A.3 TRAINER SETTINGS
            # =================================================

            settings = (
                SpecialistDpoQloraSettings(
                    base_model_path=(
                        model_path
                    ),

                    expected_model_key=(
                        args.expected_model_key
                    ),

                    output_root=(
                        output_root
                    ),

                    compute_dtype=(
                        "bfloat16"
                    ),

                    bnb_4bit_quant_type=(
                        "nf4"
                    ),

                    bnb_4bit_use_double_quant=True,

                    lora_r=16,

                    lora_alpha=32,

                    lora_dropout=0.05,

                    lora_target_modules=(
                        "all-linear"
                    ),

                    beta=0.1,

                    learning_rate=5e-6,

                    num_train_epochs=1.0,

                    per_device_train_batch_size=1,

                    per_device_eval_batch_size=1,

                    gradient_accumulation_steps=8,

                    max_length=(
                        args.max_length
                    ),

                    gradient_checkpointing=True,

                    optimizer=(
                        "paged_adamw_8bit"
                    ),

                    weight_decay=0.0,

                    warmup_ratio=0.03,

                    seed=42,
                )
            )

            dry_run = (
                SpecialistDpoQloraDryRun(
                    train_directory=(
                        train_result
                        .output_directory
                    ),

                    validation_directory=(
                        validation_result
                        .output_directory
                    ),

                    settings=(
                        settings
                    ),
                )
            )

            # =================================================
            # REAL MODEL + TRAINER CONSTRUCTION
            #
            # run() does NOT call trainer.train().
            # =================================================

            result = (
                dry_run.run()
            )

    except Exception as exc:

        print()

        print(
            "HARDWARE SMOKE: FAIL"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        return 1

    # ========================================================
    # RESULT
    # ========================================================

    if args.json:

        print(
            json.dumps(
                result.model_dump(
                    mode="json",
                    by_alias=True,
                ),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )

    else:

        print()

        print(
            "HARDWARE SMOKE: PASS"
        )

        print(
            "--------------------"
        )

        print(
            f"Dry-run ID: "
            f"{result.dry_run_id}"
        )

        print(
            f"Split: "
            f"{result.source_split_id}"
        )

        print(
            f"Agent: "
            f"{result.target_agent}"
        )

        print(
            f"Model key: "
            f"{result.target_model_key}"
        )

        print(
            f"Model class: "
            f"{result.model_class}"
        )

        print(
            f"Trainer: "
            f"{result.trainer_class}"
        )

        print(
            f"Optimizer: "
            f"{result.optimizer_class}"
        )

        print(
            f"4-bit loaded: "
            f"{result.quantized_4bit}"
        )

        print(
            f"PEFT enabled: "
            f"{result.peft_enabled}"
        )

        print(
            f"Train records: "
            f"{result.train_record_count}"
        )

        print(
            f"Validation records: "
            f"{result.validation_record_count}"
        )

        print(
            f"Processed train records: "
            f"{result.processed_train_record_count}"
        )

        print(
            f"Processed validation records: "
            f"{result.processed_validation_record_count}"
        )

        print(
            "Trainable parameters: "
            f"{result.trainable_parameter_count:,}"
        )

        print(
            "Total parameters: "
            f"{result.total_parameter_count:,}"
        )

        print(
            "Trainable percentage: "
            f"{result.trainable_parameter_percent:.4f}%"
        )

        print(
            "First DPO batch shape: "
            f"{result.first_batch_shape}"
        )

        print(
            "Peak CUDA allocated: "
            f"{result.peak_cuda_memory_gib:.3f} GiB"
        )

        print(
            "trainer.train() executed: "
            f"{result.training_executed}"
        )

    if (
        result.training_executed
        is not False
    ):

        print(
            "ERROR: smoke result unexpectedly "
            "reported training execution."
        )

        return 1

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )