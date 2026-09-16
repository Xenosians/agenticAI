from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import tempfile

from datetime import (
    datetime,
    timezone,
)

from pathlib import (
    Path,
)

from typing import (
    Any,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from learning.dpo_materializer import (
    SpecialistDpoMaterializer,
)

from learning.dpo_qlora import (
    SpecialistDpoQloraDryRun,
    SpecialistDpoQloraSettings,
    fingerprint_base_model,
)

from learning.run_dpo_qlora_smoke import (
    DEFAULT_AGENT_PATH,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_PATH,
    PROJECT_ROOT,
    _training_records,
    _validation_records,
)


# ============================================================
# OUTPUT
# ============================================================


DEFAULT_TRAINING_OUTPUT_ROOT = (
    PROJECT_ROOT
    / ".runtime"
    / "learning"
    / "training-step-smoke"
)


# ============================================================
# HELPERS
# ============================================================


def _utc_now(
) -> str:

    return (
        datetime
        .now(
            timezone.utc
        )
        .isoformat()
    )


def _run_timestamp(
) -> str:

    return (
        datetime
        .now(
            timezone.utc
        )
        .strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
    )


def _canonical_json(
    value: Any,
) -> str:

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )
    )


def _sha256_file(
    path: Path,
) -> str:

    digest = (
        hashlib.sha256()
    )

    with path.open(
        "rb"
    ) as handle:

        while True:

            chunk = (
                handle.read(
                    1024
                    * 1024
                )
            )

            if not chunk:

                break

            digest.update(
                chunk
            )

    return (
        digest.hexdigest()
    )


def _hash_trainable_parameters(
    model,
) -> str:
    """
    Deterministically fingerprint all trainable parameters.

    Phase 4B.1 uses this before and after trainer.train() to prove
    that the single optimizer step actually modified LoRA weights.
    """

    import torch

    digest = (
        hashlib.sha256()
    )

    trainable_parameter_count = (
        0
    )

    with torch.no_grad():

        for (
            name,
            parameter,
        ) in model.named_parameters():

            if not parameter.requires_grad:

                continue

            trainable_parameter_count += (
                parameter.numel()
            )

            digest.update(
                name.encode(
                    "utf-8"
                )
            )

            digest.update(
                str(
                    tuple(
                        parameter.shape
                    )
                ).encode(
                    "utf-8"
                )
            )

            value = (
                parameter
                .detach()
                .to(
                    device="cpu",
                    dtype=torch.float32,
                )
                .contiguous()
            )

            digest.update(
                value
                .numpy()
                .tobytes()
            )

    if (
        trainable_parameter_count
        <= 0
    ):

        raise ValueError(
            "Model contains no trainable parameters."
        )

    return (
        digest.hexdigest()
    )


# ============================================================
# HARD TRAINING AUTHORIZATION
# ============================================================


def authorize_one_step_training(
    *,
    allow_training: bool,
    max_steps: int | None,
) -> None:
    """
    Phase 4B.1 may execute training only when explicitly armed.

    Both conditions are mandatory:

        --allow-training
        --max-steps 1

    No other optimizer-step count is permitted.
    """

    if not allow_training:

        raise PermissionError(
            "Training authorization was not supplied. "
            "Phase 4B.1 requires --allow-training."
        )

    if (
        max_steps
        != 1
    ):

        raise ValueError(
            "Phase 4B.1 permits exactly one optimizer step. "
            "Pass --max-steps 1."
        )


# ============================================================
# ADAPTER ARTIFACT
# ============================================================


class TrainingArtifactFile(
    BaseModel
):
    path: str

    size_bytes: int

    sha256: str


class TrainingAdapterArtifact(
    BaseModel
):
    directory: Path

    content_sha256: str

    files: list[
        TrainingArtifactFile
    ] = Field(
        default_factory=list
    )


def fingerprint_adapter_directory(
    directory: Path,
) -> TrainingAdapterArtifact:

    resolved = (
        directory
        .expanduser()
        .resolve()
    )

    if not resolved.is_dir():

        raise ValueError(
            "Adapter output directory does not exist: "
            f"{resolved}"
        )

    files: list[
        TrainingArtifactFile
    ] = []

    for path in sorted(
        (
            candidate

            for candidate
            in resolved.rglob(
                "*"
            )

            if candidate.is_file()
        ),

        key=lambda item: (
            str(
                item.relative_to(
                    resolved
                )
            )
        ),
    ):

        files.append(
            TrainingArtifactFile(
                path=(
                    str(
                        path.relative_to(
                            resolved
                        )
                    )
                ),

                size_bytes=(
                    path.stat().st_size
                ),

                sha256=(
                    _sha256_file(
                        path
                    )
                ),
            )
        )

    if not files:

        raise ValueError(
            "Adapter output directory is empty."
        )

    names = {
        file.path

        for file
        in files
    }

    if (
        "adapter_config.json"
        not in names
    ):

        raise ValueError(
            "Saved PEFT artifact is missing "
            "adapter_config.json."
        )

    if not (
        "adapter_model.safetensors"
        in names
        or "adapter_model.bin"
        in names
    ):

        raise ValueError(
            "Saved PEFT artifact contains no adapter "
            "weight file."
        )

    aggregate = (
        hashlib.sha256()
    )

    for file in files:

        aggregate.update(
            _canonical_json(
                file.model_dump(
                    mode="json"
                )
            ).encode(
                "utf-8"
            )
        )

    return (
        TrainingAdapterArtifact(
            directory=(
                resolved
            ),

            content_sha256=(
                aggregate.hexdigest()
            ),

            files=(
                files
            ),
        )
    )


# ============================================================
# RESULT
# ============================================================


class OneStepTrainingSmokeManifest(
    BaseModel
):
    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    schema_name: str = Field(
        default=(
            "dpo-qlora-one-step-smoke.v1"
        ),
        alias="schema",
    )

    created_at: str

    run_id: str

    synthetic_only: bool = True

    training_executed: bool

    requested_max_steps: int

    observed_global_step: int

    source_split_id: str

    target_agent: str

    target_model_key: str

    max_length: int

    gradient_accumulation_steps: int

    train_record_count: int

    validation_record_count: int

    processed_train_record_count: int

    processed_validation_record_count: int

    optimizer_class: str

    optimizer_module: str

    optimizer_bits: int

    optimizer_is_paged: bool

    trainable_parameter_sha256_before: str

    trainable_parameter_sha256_after: str

    weight_update_observed: bool

    training_loss: float

    peak_cuda_allocated_gib: float

    peak_cuda_reserved_gib: float

    base_model_sha256_before: str

    base_model_sha256_after: str

    base_model_unchanged: bool

    adapter: TrainingAdapterArtifact

    package_versions: dict[
        str,
        str,
    ]


# ============================================================
# CLI
# ============================================================


def build_parser(
) -> argparse.ArgumentParser:

    parser = (
        argparse.ArgumentParser(
            description=(
                "Run the Phase 4B.1 synthetic DPO/QLoRA "
                "one-step training smoke. This performs "
                "a real forward pass, backward pass, and "
                "exactly one optimizer step. Real training "
                "data cannot be supplied to this executable."
            )
        )
    )

    parser.add_argument(
        "--allow-training",
        action="store_true",
        help=(
            "Explicitly authorize the real backward pass "
            "and optimizer step."
        ),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Must be exactly 1."
        ),
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
        "--max-length",
        type=int,
        default=(
            DEFAULT_MAX_LENGTH
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
    )

    return parser


# ============================================================
# TRAINING IMPLEMENTATION
# ============================================================


def _run_training(
    *,
    model_path: Path,
    agent_path: Path,
    max_length: int,
) -> OneStepTrainingSmokeManifest:

    import torch

    model_path = (
        model_path
        .expanduser()
        .resolve()
    )

    agent_path = (
        agent_path
        .expanduser()
        .resolve()
    )

    if not model_path.is_dir():

        raise ValueError(
            "Model directory does not exist: "
            f"{model_path}"
        )

    if not agent_path.is_file():

        raise ValueError(
            "Agent definition does not exist: "
            f"{agent_path}"
        )

    if (
        max_length
        < 128
    ):

        raise ValueError(
            "max_length must be at least 128."
        )

    # ========================================================
    # BASE MODEL IMMUTABILITY FINGERPRINT
    # ========================================================

    base_before = (
        fingerprint_base_model(
            model_path
        )
    )

    trainer = (
        None
    )

    run_root: (
        Path
        | None
    ) = None

    try:

        with tempfile.TemporaryDirectory(
            prefix=(
                "agentic-dpo-train-smoke-"
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

            trainer_work_root = (
                temporary_root
                / "trainer"
            )

            split_id = (
                "synthetic-one-step-training-smoke"
            )

            # =================================================
            # SYNTHETIC DATA ONLY
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
                        "synthetic-one-step-train"
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
                        "synthetic-one-step-validation"
                    ),
                )
            )

            # =================================================
            # ONE-STEP RECIPE
            #
            # This is deliberately not the eventual production
            # training recipe.
            #
            # gradient_accumulation_steps = 1:
            #     proves one actual forward/backward/update cycle.
            #
            # warmup_ratio = 0:
            #     avoids the sole optimizer step receiving zero LR.
            # =================================================

            settings = (
                SpecialistDpoQloraSettings(
                    base_model_path=(
                        model_path
                    ),

                    expected_model_key=(
                        "qwen2.5-0.5b-funccall"
                    ),

                    output_root=(
                        trainer_work_root
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

                    gradient_accumulation_steps=1,

                    max_length=(
                        max_length
                    ),

                    gradient_checkpointing=True,

                    optimizer=(
                        "paged_adamw_8bit"
                    ),

                    weight_decay=0.0,

                    warmup_ratio=0.0,

                    seed=42,
                )
            )

            builder = (
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
            # REUSE PHASE 4A'S VERIFIED CONSTRUCTION PATH
            # =================================================

            versions = (
                builder
                ._assert_versions()
            )

            (
                train,
                validation,
            ) = (
                builder
                ._load_inputs()
            )

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

                torch.cuda.reset_peak_memory_stats()

            trainer = (
                builder
                ._construct_trainer(
                    train=(
                        train
                    ),

                    validation=(
                        validation
                    ),
                )
            )

            # =================================================
            # DATASET SAFETY
            # =================================================

            processed_train_count = (
                len(
                    trainer.train_dataset
                )
            )

            processed_validation_count = (
                len(
                    trainer.eval_dataset
                )
            )

            if (
                processed_train_count
                != len(
                    train.records
                )
            ):

                raise ValueError(
                    "TRL preprocessing dropped one or more "
                    "training records."
                )

            if (
                processed_validation_count
                != len(
                    validation.records
                )
            ):

                raise ValueError(
                    "TRL preprocessing dropped one or more "
                    "validation records."
                )

            # =================================================
            # ABSOLUTE ONE-STEP LIMIT
            # =================================================

            trainer.args.max_steps = (
                1
            )

            trainer.args.gradient_accumulation_steps = (
                1
            )

            trainer.args.save_strategy = (
                "no"
            )

            trainer.args.eval_strategy = (
                "no"
            )

            trainer.args.logging_strategy = (
                "no"
            )

            # =================================================
            # BUILD AND VERIFY OPTIMIZER
            # =================================================

            trainer.create_optimizer_and_scheduler(
                num_training_steps=1
            )

            optimizer = (
                trainer.optimizer
            )

            if optimizer is None:

                raise ValueError(
                    "Trainer did not construct an optimizer."
                )

            optimizer_class = (
                optimizer
                .__class__
                .__name__
            )

            optimizer_module = (
                optimizer
                .__class__
                .__module__
            )

            optimizer_bits = (
                getattr(
                    getattr(
                        optimizer,
                        "args",
                        None,
                    ),
                    "optim_bits",
                    None,
                )
            )

            optimizer_is_paged = (
                bool(
                    getattr(
                        optimizer,
                        "is_paged",
                        False,
                    )
                )
            )

            if not (
                optimizer_module
                .startswith(
                    "bitsandbytes"
                )
            ):

                raise ValueError(
                    "One-step smoke resolved a non-"
                    "bitsandbytes optimizer: "
                    f"{optimizer_module}."
                    f"{optimizer_class}"
                )

            if (
                optimizer_bits
                != 8
            ):

                raise ValueError(
                    "One-step smoke optimizer is not "
                    "configured for 8-bit state."
                )

            if not optimizer_is_paged:

                raise ValueError(
                    "One-step smoke optimizer is not paged."
                )

            # =================================================
            # TRAINABLE PARAMETER STATE BEFORE UPDATE
            # =================================================

            parameter_hash_before = (
                _hash_trainable_parameters(
                    trainer.model
                )
            )

            # =================================================
            # THE ONLY REAL TRAINING CALL
            #
            # trainer.train() performs:
            #
            #     forward
            #     DPO loss
            #     backward
            #     optimizer.step()
            #
            # max_steps is hard-limited to exactly one.
            # =================================================

            train_output = (
                trainer.train()
            )

            observed_global_step = (
                int(
                    trainer
                    .state
                    .global_step
                )
            )

            if (
                observed_global_step
                != 1
            ):

                raise ValueError(
                    "Phase 4B.1 expected exactly one "
                    "optimizer step but observed "
                    f"{observed_global_step}."
                )

            if (
                int(
                    train_output.global_step
                )
                != 1
            ):

                raise ValueError(
                    "TrainOutput does not report exactly "
                    "one global step."
                )

            # =================================================
            # PROVE THE LORA PARAMETERS CHANGED
            # =================================================

            parameter_hash_after = (
                _hash_trainable_parameters(
                    trainer.model
                )
            )

            weight_update_observed = (
                parameter_hash_before
                != parameter_hash_after
            )

            if not weight_update_observed:

                raise ValueError(
                    "One optimizer step completed but no "
                    "trainable parameter change was observed."
                )

            # =================================================
            # MEMORY
            # =================================================

            peak_allocated_gib = (
                float(
                    torch.cuda
                    .max_memory_allocated()
                )
                / (
                    1024
                    ** 3
                )

                if torch.cuda.is_available()

                else 0.0
            )

            peak_reserved_gib = (
                float(
                    torch.cuda
                    .max_memory_reserved()
                )
                / (
                    1024
                    ** 3
                )

                if torch.cuda.is_available()

                else 0.0
            )

            # =================================================
            # SAVE ISOLATED ADAPTER ONLY
            # =================================================

            run_id = (
                "one-step-"
                + _run_timestamp()
            )

            run_root = (
                DEFAULT_TRAINING_OUTPUT_ROOT
                / run_id
            )

            if run_root.exists():

                raise ValueError(
                    "One-step smoke output already exists: "
                    f"{run_root}"
                )

            adapter_directory = (
                run_root
                / "adapter"
            )

            run_root.mkdir(
                parents=True,
                exist_ok=False,
            )

            try:

                trainer.model.save_pretrained(
                    adapter_directory,
                    safe_serialization=True,
                )

                processing_class = (
                    getattr(
                        trainer,
                        "processing_class",
                        None,
                    )
                )

                if (
                    processing_class
                    is not None
                    and hasattr(
                        processing_class,
                        "save_pretrained",
                    )
                ):

                    processing_class.save_pretrained(
                        adapter_directory
                    )

                adapter_artifact = (
                    fingerprint_adapter_directory(
                        adapter_directory
                    )
                )

                # =============================================
                # BASE CHECKPOINT MUST BE IDENTICAL
                # =============================================

                base_after = (
                    fingerprint_base_model(
                        model_path
                    )
                )

                base_model_unchanged = (
                    base_before.content_sha256
                    == base_after.content_sha256
                )

                if not base_model_unchanged:

                    raise ValueError(
                        "Base-model checkpoint changed during "
                        "adapter training."
                    )

                # =============================================
                # RESULT MANIFEST
                # =============================================

                manifest = (
                    OneStepTrainingSmokeManifest(
                        created_at=(
                            _utc_now()
                        ),

                        run_id=(
                            run_id
                        ),

                        synthetic_only=True,

                        training_executed=True,

                        requested_max_steps=1,

                        observed_global_step=(
                            observed_global_step
                        ),

                        source_split_id=(
                            train
                            .manifest
                            .source_split_id
                        ),

                        target_agent=(
                            train
                            .manifest
                            .target_agent
                        ),

                        target_model_key=(
                            train
                            .manifest
                            .target_model_key
                        ),

                        max_length=(
                            max_length
                        ),

                        gradient_accumulation_steps=1,

                        train_record_count=(
                            len(
                                train.records
                            )
                        ),

                        validation_record_count=(
                            len(
                                validation.records
                            )
                        ),

                        processed_train_record_count=(
                            processed_train_count
                        ),

                        processed_validation_record_count=(
                            processed_validation_count
                        ),

                        optimizer_class=(
                            optimizer_class
                        ),

                        optimizer_module=(
                            optimizer_module
                        ),

                        optimizer_bits=(
                            int(
                                optimizer_bits
                            )
                        ),

                        optimizer_is_paged=(
                            optimizer_is_paged
                        ),

                        trainable_parameter_sha256_before=(
                            parameter_hash_before
                        ),

                        trainable_parameter_sha256_after=(
                            parameter_hash_after
                        ),

                        weight_update_observed=(
                            weight_update_observed
                        ),

                        training_loss=(
                            float(
                                train_output
                                .training_loss
                            )
                        ),

                        peak_cuda_allocated_gib=(
                            peak_allocated_gib
                        ),

                        peak_cuda_reserved_gib=(
                            peak_reserved_gib
                        ),

                        base_model_sha256_before=(
                            base_before.content_sha256
                        ),

                        base_model_sha256_after=(
                            base_after.content_sha256
                        ),

                        base_model_unchanged=(
                            base_model_unchanged
                        ),

                        adapter=(
                            adapter_artifact
                        ),

                        package_versions=(
                            versions
                        ),
                    )
                )

                (
                    run_root
                    / "manifest.json"
                ).write_text(
                    json.dumps(
                        manifest.model_dump(
                            mode="json",
                            by_alias=True,
                        ),
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )

            except Exception:

                shutil.rmtree(
                    run_root,
                    ignore_errors=True,
                )

                run_root = (
                    None
                )

                raise

            return (
                manifest
            )

    finally:

        if trainer is not None:

            del trainer

        gc.collect()

        try:

            import torch

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

        except Exception:

            pass


# ============================================================
# MAIN
# ============================================================


def main(
) -> int:

    args = (
        build_parser()
        .parse_args()
    )

    # Authorization happens before model loading, CUDA setup,
    # materialization, or any training-capable operation.
    try:

        authorize_one_step_training(
            allow_training=(
                args.allow_training
            ),

            max_steps=(
                args.max_steps
            ),
        )

    except Exception as exc:

        print(
            "TRAINING SMOKE: REFUSED"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        return 2

    print(
        "Phase 4B.1 DPO/QLoRA One-Step Training Smoke"
    )

    print(
        "============================================="
    )

    print(
        "Model: "
        f"{args.model_path.expanduser().resolve()}"
    )

    print(
        "Agent: "
        f"{args.agent_path.expanduser().resolve()}"
    )

    print(
        f"Max length: {args.max_length}"
    )

    print(
        "Evidence: SYNTHETIC ONLY"
    )

    print(
        "Optimizer steps permitted: EXACTLY 1"
    )

    print(
        "Real backward pass: ENABLED"
    )

    print()

    try:

        result = (
            _run_training(
                model_path=(
                    args.model_path
                ),

                agent_path=(
                    args.agent_path
                ),

                max_length=(
                    args.max_length
                ),
            )
        )

    except Exception as exc:

        print()

        print(
            "TRAINING SMOKE: FAIL"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        return 1

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
            "TRAINING SMOKE: PASS"
        )

        print(
            "--------------------"
        )

        print(
            f"Run ID: "
            f"{result.run_id}"
        )

        print(
            f"Global step: "
            f"{result.observed_global_step}"
        )

        print(
            f"Training loss: "
            f"{result.training_loss:.6f}"
        )

        print(
            f"Optimizer: "
            f"{result.optimizer_module}."
            f"{result.optimizer_class}"
        )

        print(
            f"Optimizer bits: "
            f"{result.optimizer_bits}"
        )

        print(
            f"Optimizer paged: "
            f"{result.optimizer_is_paged}"
        )

        print(
            f"Weight update observed: "
            f"{result.weight_update_observed}"
        )

        print(
            f"Base model unchanged: "
            f"{result.base_model_unchanged}"
        )

        print(
            "Peak CUDA allocated: "
            f"{result.peak_cuda_allocated_gib:.3f} GiB"
        )

        print(
            "Peak CUDA reserved: "
            f"{result.peak_cuda_reserved_gib:.3f} GiB"
        )

        print(
            f"Adapter: "
            f"{result.adapter.directory}"
        )

        print(
            f"Adapter SHA-256: "
            f"{result.adapter.content_sha256}"
        )

        print(
            f"Synthetic only: "
            f"{result.synthetic_only}"
        )

        print(
            f"Training executed: "
            f"{result.training_executed}"
        )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )