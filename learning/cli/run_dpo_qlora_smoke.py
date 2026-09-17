from __future__ import annotations

import argparse
import json
import tempfile

from pathlib import Path
from typing import Any

from config import ModelProfileSettings

from learning.evidence.execution_provenance import (
    build_specialist_execution_provenance,
)

from learning.evidence.types import (
    DatasetPromotion,
    PreferenceDatasetRecord,
    PreferenceOption,
)

from learning.training.dpo_materializer import (
    SpecialistDpoMaterializer,
)

from learning.training.dpo_qlora import (
    SpecialistDpoQloraDryRun,
    SpecialistDpoQloraSettings,
)

from learning.training.provenance_gate import (
    SPECIALIST_TRAINING_MAX_NEW_TOKENS,
)

from subagents.core.definitions.types import (
    AgentDefinition,
)

from subagents.core.tooling.capabilities import (
    build_agent_capability_catalog,
)

from subagents.core.tooling.prompt import (
    build_worker_system_prompt,
)


# ============================================================
# PROJECT PATHS
# ============================================================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
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


DEFAULT_MODEL_PATH = Path(
    "/mnt/c/project/"
    "agenticaiPersonal/"
    "Models/"
    "qwen2.5-0.5b-funccall"
)


# Measured against the real account-specialist runtime prompt:
#
#   largest prompt:        1157 tokens
#   largest full sequence: 1178 tokens
#
# 1536 leaves headroom for the current target.
DEFAULT_MAX_LENGTH = 1536


# ============================================================
# SYNTHETIC RUNTIME PROFILE
# ============================================================


def _build_model_profile(
    model_path: Path,
) -> ModelProfileSettings:
    """
    Reproduce the current account-specialist runtime profile.

    The synthetic smoke path must use the same profile identity
    that its generated execution provenance claims.
    """

    return ModelProfileSettings(
        backend="qwen-funccall",
        model_path=(
            model_path
            .expanduser()
            .resolve()
        ),
        enabled=True,
        quantization="bnb4",
        compute_dtype="bfloat16",
        device_map="auto",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


# ============================================================
# SYNTHETIC TRUSTED RECORDS
# ============================================================


def _promotion() -> DatasetPromotion:

    return DatasetPromotion(
        promoted_at="2026-09-16T00:00:00+00:00",
        promoted_by="trusted_review",
        reason=(
            "Synthetic Phase 4A hardware smoke test."
        ),
    )


def _runtime_messages(
    *,
    system_prompt: str,
    user_request: str,
    task_instructions: str,
) -> list[dict[str, str]]:
    """
    Reproduce AgentRuntime specialist message construction exactly.
    """

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_request,
        },
    ]

    normalized_instructions = (
        task_instructions.strip()
    )

    if normalized_instructions:

        messages.append(
            {
                "role": "user",
                "content": (
                    "Additional task context "
                    "from the routing stage:\n"
                    f"{normalized_instructions}"
                ),
            }
        )

    return messages


def _record(
    *,
    record_id: str,
    user_request: str,
    task_instructions: str,
    rejected_user_id: str,
    chosen_user_id: str,
    agent: AgentDefinition,
    model_profile: ModelProfileSettings,
    capability_catalog: list[dict[str, Any]],
    system_prompt: str,
) -> PreferenceDatasetRecord:
    """
    Construct synthetic preference evidence with a complete,
    cryptographically verifiable execution identity.

    This remains synthetic evidence. Provenance proves what
    environment the synthetic behavior represents; it does not
    turn synthetic evidence into real user evidence.
    """

    normalized_instructions = (
        task_instructions.strip()
    )

    messages = _runtime_messages(
        system_prompt=system_prompt,
        user_request=user_request,
        task_instructions=normalized_instructions,
    )

    execution_provenance = (
        build_specialist_execution_provenance(
            agent=agent,
            model_profile=model_profile,
            capability_catalog=capability_catalog,
            messages=messages,
            user_request=user_request,
            task_instructions=normalized_instructions,
            max_new_tokens=(
                SPECIALIST_TRAINING_MAX_NEW_TOKENS
            ),
        )
    )

    return PreferenceDatasetRecord(
        record_id=record_id,
        source_example_id=f"example-{record_id}",
        trajectory_id=f"trajectory-{record_id}",
        correction_id=f"correction-{record_id}",
        task_id=f"task-{record_id}",
        task_instructions=normalized_instructions,
        execution_provenance=execution_provenance,
        source="trusted_review",
        correction_type="account_identifier",
        user_request=user_request,
        rejected=PreferenceOption(
            agent=agent.name,
            tool="account_status",
            arguments={
                "user_id": rejected_user_id,
            },
            answer=None,
        ),
        chosen=PreferenceOption(
            agent=agent.name,
            tool="account_status",
            arguments={
                "user_id": chosen_user_id,
            },
            answer=None,
        ),
        promotion=_promotion(),
        dataset_eligible=True,
    )


def _training_records(
    *,
    agent: AgentDefinition,
    model_profile: ModelProfileSettings,
    capability_catalog: list[dict[str, Any]],
    system_prompt: str,
) -> list[PreferenceDatasetRecord]:

    return [
        _record(
            record_id="smoke-train-1",
            user_request="Is jdoe locked?",
            task_instructions=(
                "Check the account state for jdoe."
            ),
            rejected_user_id="asmith",
            chosen_user_id="jdoe",
            agent=agent,
            model_profile=model_profile,
            capability_catalog=capability_catalog,
            system_prompt=system_prompt,
        ),
        _record(
            record_id="smoke-train-2",
            user_request=(
                "Check the account status for asmith."
            ),
            task_instructions=(
                "Inspect the requested directory account."
            ),
            rejected_user_id="jdoe",
            chosen_user_id="asmith",
            agent=agent,
            model_profile=model_profile,
            capability_catalog=capability_catalog,
            system_prompt=system_prompt,
        ),
    ]


def _validation_records(
    *,
    agent: AgentDefinition,
    model_profile: ModelProfileSettings,
    capability_catalog: list[dict[str, Any]],
    system_prompt: str,
) -> list[PreferenceDatasetRecord]:

    return [
        _record(
            record_id="smoke-validation-1",
            user_request=(
                "Tell me the current account state "
                "for mbrown."
            ),
            task_instructions=(
                "Retrieve account status for mbrown."
            ),
            rejected_user_id="jdoe",
            chosen_user_id="mbrown",
            agent=agent,
            model_profile=model_profile,
            capability_catalog=capability_catalog,
            system_prompt=system_prompt,
        )
    ]


def _synthetic_preference_records(
    *,
    agent: AgentDefinition,
    model_profile: ModelProfileSettings,
) -> tuple[
    list[PreferenceDatasetRecord],
    list[PreferenceDatasetRecord],
]:
    """
    Build train + validation synthetic evidence against one exact
    specialist execution environment.
    """

    capability_catalog = (
        build_agent_capability_catalog(
            agent,
            include_arguments=True,
        )
    )

    system_prompt = (
        build_worker_system_prompt(
            agent,
            capability_catalog=capability_catalog,
        )
    )

    training = _training_records(
        agent=agent,
        model_profile=model_profile,
        capability_catalog=capability_catalog,
        system_prompt=system_prompt,
    )

    validation = _validation_records(
        agent=agent,
        model_profile=model_profile,
        capability_catalog=capability_catalog,
        system_prompt=system_prompt,
    )

    return (
        training,
        validation,
    )


# ============================================================
# CLI
# ============================================================


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
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

    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )

    parser.add_argument(
        "--agent-path",
        type=Path,
        default=DEFAULT_AGENT_PATH,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--expected-model-key",
        default="qwen2.5-0.5b-funccall",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
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


def main() -> int:

    args = build_parser().parse_args()

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
            prefix="agentic-dpo-smoke-",
        ) as temporary_directory:

            temporary_root = Path(
                temporary_directory
            )

            materialized_root = (
                temporary_root
                / "materialized"
            )

            split_id = (
                "synthetic-hardware-smoke"
            )

            model_profile = (
                _build_model_profile(
                    model_path
                )
            )

            materializer = (
                SpecialistDpoMaterializer(
                    agent_definition_path=agent_path,
                    model_profile=model_profile,
                    output_root=materialized_root,
                )
            )

            if (
                args.expected_model_key
                != materializer.agent.model
            ):

                raise ValueError(
                    "Requested expected model key does not "
                    "match the specialist definition: "
                    f"requested={args.expected_model_key!r}, "
                    f"agent={materializer.agent.model!r}."
                )

            (
                training_records,
                validation_records,
            ) = _synthetic_preference_records(
                agent=materializer.agent,
                model_profile=model_profile,
            )

            train_result = materializer.build(
                records=training_records,
                source_split_id=split_id,
                source_partition="train",
                source_sha256=(
                    "synthetic-train-source"
                ),
            )

            validation_result = materializer.build(
                records=validation_records,
                source_split_id=split_id,
                source_partition="validation",
                source_sha256=(
                    "synthetic-validation-source"
                ),
            )

            settings = SpecialistDpoQloraSettings(
                base_model_path=model_path,
                expected_model_key=(
                    args.expected_model_key
                ),
                output_root=output_root,
                compute_dtype="bfloat16",
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                lora_r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                lora_target_modules="all-linear",
                beta=0.1,
                learning_rate=5e-6,
                num_train_epochs=1.0,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=8,
                max_length=args.max_length,
                gradient_checkpointing=True,
                optimizer="paged_adamw_8bit",
                weight_decay=0.0,
                warmup_ratio=0.03,
                seed=42,
            )

            dry_run = SpecialistDpoQloraDryRun(
                train_directory=(
                    train_result.output_directory
                ),
                validation_directory=(
                    validation_result.output_directory
                ),
                settings=settings,
            )

            result = dry_run.run()

    except Exception as exc:

        print()
        print(
            "HARDWARE SMOKE: FAIL"
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
            "HARDWARE SMOKE: PASS"
        )
        print(
            "--------------------"
        )
        print(
            f"Dry-run ID: {result.dry_run_id}"
        )
        print(
            f"Split: {result.source_split_id}"
        )
        print(
            f"Agent: {result.target_agent}"
        )
        print(
            f"Model key: {result.target_model_key}"
        )
        print(
            f"Model class: {result.model_class}"
        )
        print(
            f"Trainer: {result.trainer_class}"
        )
        print(
            f"Optimizer: {result.optimizer_class}"
        )
        print(
            f"4-bit loaded: {result.quantized_4bit}"
        )
        print(
            f"PEFT enabled: {result.peft_enabled}"
        )
        print(
            f"Train records: {result.train_record_count}"
        )
        print(
            "Validation records: "
            f"{result.validation_record_count}"
        )
        print(
            "Processed train records: "
            f"{result.processed_train_record_count}"
        )
        print(
            "Processed validation records: "
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

    if result.training_executed is not False:

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
