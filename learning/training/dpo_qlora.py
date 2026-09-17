from __future__ import annotations

import gc
import hashlib
import json

from datetime import (
    datetime,
    timezone,
)

from importlib import (
    metadata,
)

from pathlib import (
    Path,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from learning.evidence.execution_provenance import (
    PROVENANCE_SCHEMA,
    RuntimeModelArtifactFingerprint,
    fingerprint_runtime_model_artifact,
)

from learning.training.dpo_materializer import (
    SpecialistDpoManifest,
    SpecialistDpoRecord,
)


# ============================================================
# TRAINING API LOCK
# ============================================================


EXPECTED_TRAINING_VERSIONS = {
    "transformers":
        "5.16.1",

    "accelerate":
        "1.14.0",

    "peft":
        "0.20.0",

    "trl":
        "1.12.0",

    "bitsandbytes":
        "0.50.2",

    "datasets":
        "5.0.1",
}


TASK_CONTEXT_PREFIX = (
    "Additional task context "
    "from the routing stage:\n"
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


def _canonical_json(
    value,
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


def _sha256_text(
    value: str,
) -> str:

    return (
        hashlib
        .sha256(
            value.encode(
                "utf-8"
            )
        )
        .hexdigest()
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


def _installed_versions(
) -> dict[
    str,
    str,
]:

    result: dict[
        str,
        str,
    ] = {}

    for package in (
        "torch",
        "transformers",
        "accelerate",
        "peft",
        "trl",
        "bitsandbytes",
        "datasets",
    ):

        try:

            result[
                package
            ] = (
                metadata.version(
                    package
                )
            )

        except (
            metadata.PackageNotFoundError
        ):

            result[
                package
            ] = (
                "missing"
            )

    return result


# ============================================================
# SETTINGS
# ============================================================


class SpecialistDpoQloraSettings(
    BaseModel
):
    """
    Specialist DPO + QLoRA configuration.

    warmup_ratio remains the stable recipe-level name.

    Transformers 5.16.1 accepts a float in [0, 1) through
    warmup_steps to represent a ratio, so translation happens only
    at the dependency boundary.
    """

    model_config = (
        ConfigDict(
            populate_by_name=True,
            extra="forbid",
        )
    )

    schema_name: str = Field(
        default=(
            "specialist-dpo-qlora-settings.v1"
        ),
        alias="schema",
    )

    base_model_path: Path

    expected_model_key: str

    output_root: Path

    compute_dtype: str = (
        "bfloat16"
    )

    bnb_4bit_quant_type: str = (
        "nf4"
    )

    bnb_4bit_use_double_quant: bool = (
        True
    )

    lora_r: int = Field(
        default=16,
        ge=1,
    )

    lora_alpha: int = Field(
        default=32,
        ge=1,
    )

    lora_dropout: float = Field(
        default=0.05,
        ge=0.0,
        lt=1.0,
    )

    lora_target_modules: str = (
        "all-linear"
    )

    beta: float = Field(
        default=0.1,
        gt=0.0,
    )

    learning_rate: float = Field(
        default=5e-6,
        gt=0.0,
    )

    num_train_epochs: float = Field(
        default=1.0,
        gt=0.0,
    )

    per_device_train_batch_size: int = Field(
        default=1,
        ge=1,
    )

    per_device_eval_batch_size: int = Field(
        default=1,
        ge=1,
    )

    gradient_accumulation_steps: int = Field(
        default=8,
        ge=1,
    )

    max_length: int = Field(
        default=1024,
        ge=128,
    )

    gradient_checkpointing: bool = (
        True
    )

    optimizer: str = (
        "paged_adamw_8bit"
    )

    weight_decay: float = Field(
        default=0.0,
        ge=0.0,
    )

    warmup_ratio: float = Field(
        default=0.03,
        ge=0.0,
        lt=1.0,
    )

    seed: int = (
        42
    )

    @field_validator(
        "expected_model_key"
    )
    @classmethod
    def validate_model_key(
        cls,
        value: str,
    ) -> str:

        normalized = (
            value.strip()
        )

        if not normalized:

            raise ValueError(
                "expected_model_key must not be empty."
            )

        return normalized

    @field_validator(
        "compute_dtype"
    )
    @classmethod
    def validate_compute_dtype(
        cls,
        value: str,
    ) -> str:

        normalized = (
            value
            .strip()
            .lower()
        )

        if normalized not in {
            "bfloat16",
            "float16",
        }:

            raise ValueError(
                "compute_dtype must be "
                "'bfloat16' or 'float16'."
            )

        return normalized

    @field_validator(
        "bnb_4bit_quant_type"
    )
    @classmethod
    def validate_quant_type(
        cls,
        value: str,
    ) -> str:

        normalized = (
            value
            .strip()
            .lower()
        )

        if (
            normalized
            != "nf4"
        ):

            raise ValueError(
                "Phase 4A QLoRA requires NF4."
            )

        return normalized

    @field_validator(
        "lora_target_modules"
    )
    @classmethod
    def validate_lora_targets(
        cls,
        value: str,
    ) -> str:

        normalized = (
            value.strip()
        )

        if (
            normalized
            != "all-linear"
        ):

            raise ValueError(
                "Phase 4A QLoRA currently requires "
                "lora_target_modules='all-linear'."
            )

        return normalized


# ============================================================
# VERIFIED MATERIALIZED PARTITION
# ============================================================


class VerifiedDpoPartition(
    BaseModel
):
    directory: Path

    manifest: SpecialistDpoManifest

    records: list[
        SpecialistDpoRecord
    ]


def _validate_tool_call(
    value: str,
) -> None:

    try:

        payload = (
            json.loads(
                value
            )
        )

    except json.JSONDecodeError as exc:

        raise ValueError(
            "Materialized DPO completion is not "
            "valid JSON."
        ) from exc

    if (
        not isinstance(
            payload,
            list,
        )
        or len(
            payload
        )
        != 1
    ):

        raise ValueError(
            "Materialized specialist completion must "
            "contain exactly one tool call."
        )

    call = (
        payload[
            0
        ]
    )

    if not isinstance(
        call,
        dict,
    ):

        raise ValueError(
            "Materialized specialist tool call "
            "must be an object."
        )

    tool_name = (
        call.get(
            "name"
        )
    )

    arguments = (
        call.get(
            "arguments"
        )
    )

    if (
        not isinstance(
            tool_name,
            str,
        )
        or not tool_name.strip()
    ):

        raise ValueError(
            "Materialized specialist tool call "
            "has no valid tool name."
        )

    if not isinstance(
        arguments,
        dict,
    ):

        raise ValueError(
            "Materialized specialist tool call "
            "arguments must be an object."
        )


def load_specialist_dpo_partition(
    *,
    directory: Path,
    expected_partition: str,
) -> VerifiedDpoPartition:
    """
    Parse and structurally verify a materialized DPO artifact.

    Important:

        this function intentionally remains backward compatible.

    Historical artifacts remain readable here.

    Production training eligibility is enforced separately by
    validate_specialist_dpo_training_contract().
    """

    resolved = (
        directory
        .expanduser()
        .resolve()
    )

    manifest_path = (
        resolved
        / "manifest.json"
    )

    records_path = (
        resolved
        / "records.jsonl"
    )

    if not manifest_path.is_file():

        raise ValueError(
            "DPO partition manifest does not exist: "
            f"{manifest_path}"
        )

    if not records_path.is_file():

        raise ValueError(
            "DPO partition records do not exist: "
            f"{records_path}"
        )

    try:

        manifest = (
            SpecialistDpoManifest
            .model_validate_json(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )
        )

    except Exception as exc:

        raise ValueError(
            "Invalid specialist DPO manifest: "
            f"{exc}"
        ) from exc

    normalized_expected = (
        expected_partition
        .strip()
        .lower()
    )

    if normalized_expected not in {
        "train",
        "validation",
    }:

        raise ValueError(
            "expected_partition must be "
            "'train' or 'validation'."
        )

    if (
        manifest.source_partition
        != normalized_expected
    ):

        raise ValueError(
            "DPO partition mismatch: expected "
            f"'{normalized_expected}' but manifest "
            f"contains '{manifest.source_partition}'."
        )

    observed_sha256 = (
        _sha256_file(
            records_path
        )
    )

    if (
        observed_sha256
        != manifest.content_sha256
    ):

        raise ValueError(
            "DPO partition verification failed: "
            "records.jsonl SHA-256 does not match "
            "manifest."
        )

    records: list[
        SpecialistDpoRecord
    ] = []

    with records_path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for (
            line_number,
            line,
        ) in enumerate(
            handle,
            start=1,
        ):

            if not line.strip():

                continue

            try:

                payload = (
                    json.loads(
                        line
                    )
                )

                record = (
                    SpecialistDpoRecord
                    .model_validate(
                        payload
                    )
                )

            except Exception as exc:

                raise ValueError(
                    "Invalid specialist DPO record at "
                    f"{records_path}:{line_number}: "
                    f"{exc}"
                ) from exc

            records.append(
                record
            )

    if (
        len(
            records
        )
        != manifest.record_count
    ):

        raise ValueError(
            "DPO partition record count does not "
            "match manifest."
        )

    source_record_ids = [
        record.source_record_id

        for record
        in records
    ]

    if (
        source_record_ids
        != manifest.source_record_ids
    ):

        raise ValueError(
            "DPO partition source record membership "
            "does not match manifest."
        )

    dpo_ids = [
        record.dpo_record_id

        for record
        in records
    ]

    if (
        len(
            dpo_ids
        )
        != len(
            set(
                dpo_ids
            )
        )
    ):

        raise ValueError(
            "DPO partition contains duplicate "
            "dpo_record_id values."
        )

    if (
        len(
            source_record_ids
        )
        != len(
            set(
                source_record_ids
            )
        )
    ):

        raise ValueError(
            "DPO partition contains duplicate "
            "source_record_id values."
        )

    for record in records:

        if (
            record.target_agent
            != manifest.target_agent
        ):

            raise ValueError(
                "DPO record target_agent does not "
                "match partition manifest."
            )

        if (
            record.target_model_key
            != manifest.target_model_key
        ):

            raise ValueError(
                "DPO record target_model_key does not "
                "match partition manifest."
            )

        if record.change_type not in {
            "tool",
            "arguments",
        }:

            raise ValueError(
                "Unsupported specialist DPO "
                f"change_type: {record.change_type}"
            )

        if not record.prompt_messages:

            raise ValueError(
                "Materialized DPO record has no prompt."
            )

        if (
            record
            .prompt_messages[
                0
            ]
            .get(
                "role"
            )
            != "system"
        ):

            raise ValueError(
                "Materialized specialist prompt must "
                "begin with the system message."
            )

        if not any(
            message.get(
                "role"
            )
            == "user"

            for message
            in record.prompt_messages
        ):

            raise ValueError(
                "Materialized specialist prompt must "
                "contain a user message."
            )

        if (
            record.chosen
            == record.rejected
        ):

            raise ValueError(
                "Chosen and rejected specialist "
                "completions are identical."
            )

        _validate_tool_call(
            record.chosen
        )

        _validate_tool_call(
            record.rejected
        )

    return (
        VerifiedDpoPartition(
            directory=(
                resolved
            ),

            manifest=(
                manifest
            ),

            records=(
                records
            ),
        )
    )


# ============================================================
# TRAIN / VALIDATION PAIR
# ============================================================


def validate_specialist_dpo_pair(
    *,
    train: VerifiedDpoPartition,
    validation: VerifiedDpoPartition,
    expected_model_key: str,
) -> None:

    if (
        train.manifest.source_split_id
        != validation.manifest.source_split_id
    ):

        raise ValueError(
            "Train and validation DPO artifacts "
            "come from different Phase-3 splits."
        )

    if (
        train.manifest.target_agent
        != validation.manifest.target_agent
    ):

        raise ValueError(
            "Train and validation DPO artifacts "
            "target different specialists."
        )

    if (
        train.manifest.target_model_key
        != validation.manifest.target_model_key
    ):

        raise ValueError(
            "Train and validation DPO artifacts "
            "target different model keys."
        )

    if (
        train.manifest.target_model_key
        != expected_model_key
    ):

        raise ValueError(
            "DPO artifact model key does not match "
            "the requested training model key: "
            f"artifact={train.manifest.target_model_key!r}, "
            f"requested={expected_model_key!r}."
        )

    train_source_ids = {
        record.source_record_id

        for record
        in train.records
    }

    validation_source_ids = {
        record.source_record_id

        for record
        in validation.records
    }

    source_overlap = (
        train_source_ids
        & validation_source_ids
    )

    if source_overlap:

        raise ValueError(
            "DPO train/validation source record "
            "overlap detected: "
            + ", ".join(
                sorted(
                    source_overlap
                )
            )
        )

    train_dpo_ids = {
        record.dpo_record_id

        for record
        in train.records
    }

    validation_dpo_ids = {
        record.dpo_record_id

        for record
        in validation.records
    }

    dpo_overlap = (
        train_dpo_ids
        & validation_dpo_ids
    )

    if dpo_overlap:

        raise ValueError(
            "DPO train/validation materialized "
            "record overlap detected: "
            + ", ".join(
                sorted(
                    dpo_overlap
                )
            )
        )


# ============================================================
# PHASE 4C.2 TRAINING PROVENANCE CONTRACT
# ============================================================


def _manifest_training_identity(
    manifest: SpecialistDpoManifest,
) -> tuple:

    return (
        manifest.target_agent,
        manifest.target_model_key,
        manifest.target_model_artifact_sha256,
        manifest.target_model_weights_sha256,
        manifest.target_tokenizer_artifact_sha256,
        manifest.target_model_profile_sha256,
        manifest.target_agent_definition_sha256,
        manifest.target_capability_catalog_sha256,
        manifest.target_system_prompt_sha256,
        manifest.specialist_max_new_tokens,
    )


def _execution_environment_identity(
    provenance,
) -> tuple:

    return (
        provenance.agent_name,
        provenance.model_key,

        provenance.backend,
        provenance.quantization,
        provenance.compute_dtype,
        provenance.device_map,
        provenance.bnb_4bit_quant_type,
        provenance.bnb_4bit_use_double_quant,
        provenance.model_profile_sha256,

        provenance.model_artifact_sha256,
        provenance.model_weights_sha256,
        provenance.model_config_sha256,
        provenance.generation_config_sha256,

        provenance.tokenizer_artifact_sha256,
        provenance.tokenizer_config_sha256,
        provenance.tokenizer_json_sha256,
        provenance.chat_template_sha256,

        provenance.agent_definition_sha256,
        provenance.capability_catalog_sha256,
        provenance.system_prompt_sha256,

        provenance.max_new_tokens,
    )


def _validate_record_prompt_provenance(
    record: SpecialistDpoRecord,
) -> None:

    provenance = (
        record.source_execution_provenance
    )

    if provenance is None:

        raise ValueError(
            "Training record is missing source "
            "execution provenance."
        )

    messages = (
        record.prompt_messages
    )

    if (
        len(
            messages
        )
        not in {
            2,
            3,
        }
    ):

        raise ValueError(
            "Training record prompt does not match "
            "the specialist runtime message shape."
        )

    if (
        messages[
            0
        ].get(
            "role"
        )
        != "system"
    ):

        raise ValueError(
            "Training record prompt does not begin "
            "with the specialist system message."
        )

    if (
        messages[
            1
        ].get(
            "role"
        )
        != "user"
    ):

        raise ValueError(
            "Training record prompt does not contain "
            "the original request in message position 2."
        )

    system_content = (
        messages[
            0
        ].get(
            "content"
        )
    )

    user_content = (
        messages[
            1
        ].get(
            "content"
        )
    )

    if not isinstance(
        system_content,
        str,
    ):

        raise ValueError(
            "Training record system prompt is invalid."
        )

    if not isinstance(
        user_content,
        str,
    ):

        raise ValueError(
            "Training record user request is invalid."
        )

    if (
        _sha256_text(
            system_content
        )
        != provenance.system_prompt_sha256
    ):

        raise ValueError(
            "Training record system prompt does not "
            "match source execution provenance."
        )

    if (
        _sha256_text(
            user_content
        )
        != provenance.user_request_sha256
    ):

        raise ValueError(
            "Training record user request does not "
            "match source execution provenance."
        )

    if (
        _sha256_text(
            _canonical_json(
                messages
            )
        )
        != provenance.messages_sha256
    ):

        raise ValueError(
            "Training record messages do not match "
            "source execution provenance."
        )

    if (
        provenance.task_instructions_sha256
        is None
    ):

        if (
            len(
                messages
            )
            != 2
        ):

            raise ValueError(
                "Training record contains routing context "
                "that was not present in source provenance."
            )

        return

    if (
        len(
            messages
        )
        != 3
    ):

        raise ValueError(
            "Training record is missing routing context "
            "recorded by source provenance."
        )

    context_message = (
        messages[
            2
        ]
    )

    if (
        context_message.get(
            "role"
        )
        != "user"
    ):

        raise ValueError(
            "Training record routing context message "
            "has an invalid role."
        )

    context_content = (
        context_message.get(
            "content"
        )
    )

    if (
        not isinstance(
            context_content,
            str,
        )
        or not context_content.startswith(
            TASK_CONTEXT_PREFIX
        )
    ):

        raise ValueError(
            "Training record routing context does not "
            "match the specialist runtime contract."
        )

    task_instructions = (
        context_content[
            len(
                TASK_CONTEXT_PREFIX
            ):
        ]
    )

    if (
        _sha256_text(
            task_instructions
        )
        != provenance.task_instructions_sha256
    ):

        raise ValueError(
            "Training record routing context does not "
            "match source execution provenance."
        )


def _validate_partition_training_provenance(
    partition: VerifiedDpoPartition,
) -> None:

    manifest = (
        partition.manifest
    )

    if (
        manifest.execution_provenance_enforced
        is not True
    ):

        raise ValueError(
            "DPO partition is not provenance-enforced "
            "and is not eligible for training."
        )

    required_manifest_values = {
        "target_model_artifact_sha256":
            manifest.target_model_artifact_sha256,

        "target_model_weights_sha256":
            manifest.target_model_weights_sha256,

        "target_tokenizer_artifact_sha256":
            manifest.target_tokenizer_artifact_sha256,

        "target_model_profile_sha256":
            manifest.target_model_profile_sha256,

        "target_agent_definition_sha256":
            manifest.target_agent_definition_sha256,

        "target_capability_catalog_sha256":
            manifest.target_capability_catalog_sha256,

        "target_system_prompt_sha256":
            manifest.target_system_prompt_sha256,

        "specialist_max_new_tokens":
            manifest.specialist_max_new_tokens,
    }

    missing = [
        name

        for (
            name,
            value,
        ) in required_manifest_values.items()

        if value is None
    ]

    if missing:

        raise ValueError(
            "Provenance-enforced DPO manifest is "
            "missing required training identity: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    if not partition.records:

        raise ValueError(
            "Provenance-enforced DPO partition is empty."
        )

    reference_environment = None

    for record in partition.records:

        provenance = (
            record.source_execution_provenance
        )

        if provenance is None:

            raise ValueError(
                "Training record is missing source "
                "execution provenance."
            )

        if (
            provenance.schema_name
            != PROVENANCE_SCHEMA
        ):

            raise ValueError(
                "Training record uses unsupported "
                "execution provenance schema: "
                f"{provenance.schema_name}"
            )

        if (
            provenance.provenance_complete
            is not True
        ):

            raise ValueError(
                "Training record contains incomplete "
                "execution provenance."
            )

        if (
            provenance.agent_name
            != manifest.target_agent
        ):

            raise ValueError(
                "Training record provenance agent does "
                "not match DPO manifest."
            )

        if (
            provenance.model_key
            != manifest.target_model_key
        ):

            raise ValueError(
                "Training record provenance model key "
                "does not match DPO manifest."
            )

        if (
            provenance.model_artifact_sha256
            != manifest.target_model_artifact_sha256
        ):

            raise ValueError(
                "Training record model artifact identity "
                "does not match DPO manifest."
            )

        if (
            provenance.model_weights_sha256
            != manifest.target_model_weights_sha256
        ):

            raise ValueError(
                "Training record model weights identity "
                "does not match DPO manifest."
            )

        if (
            provenance.tokenizer_artifact_sha256
            != manifest.target_tokenizer_artifact_sha256
        ):

            raise ValueError(
                "Training record tokenizer identity "
                "does not match DPO manifest."
            )

        if (
            provenance.model_profile_sha256
            != manifest.target_model_profile_sha256
        ):

            raise ValueError(
                "Training record model profile identity "
                "does not match DPO manifest."
            )

        if (
            provenance.agent_definition_sha256
            != manifest.target_agent_definition_sha256
        ):

            raise ValueError(
                "Training record agent definition identity "
                "does not match DPO manifest."
            )

        if (
            provenance.capability_catalog_sha256
            != manifest.target_capability_catalog_sha256
        ):

            raise ValueError(
                "Training record capability catalog identity "
                "does not match DPO manifest."
            )

        if (
            provenance.system_prompt_sha256
            != manifest.target_system_prompt_sha256
        ):

            raise ValueError(
                "Training record system prompt identity "
                "does not match DPO manifest."
            )

        if (
            provenance.max_new_tokens
            != manifest.specialist_max_new_tokens
        ):

            raise ValueError(
                "Training record generation contract "
                "does not match DPO manifest."
            )

        _validate_record_prompt_provenance(
            record
        )

        environment = (
            _execution_environment_identity(
                provenance
            )
        )

        if (
            reference_environment
            is None
        ):

            reference_environment = (
                environment
            )

        elif (
            environment
            != reference_environment
        ):

            raise ValueError(
                "DPO partition mixes records from "
                "different specialist execution environments."
            )


def validate_specialist_dpo_training_contract(
    *,
    train: VerifiedDpoPartition,
    validation: VerifiedDpoPartition,
    expected_model_key: str,
) -> None:
    """
    Fail-closed training eligibility gate.

    The normal partition loader remains backward compatible, but
    actual training requires both partitions to prove that Phase
    4C.2 materialization provenance enforcement occurred.
    """

    validate_specialist_dpo_pair(
        train=(
            train
        ),

        validation=(
            validation
        ),

        expected_model_key=(
            expected_model_key
        ),
    )

    _validate_partition_training_provenance(
        train
    )

    _validate_partition_training_provenance(
        validation
    )

    train_identity = (
        _manifest_training_identity(
            train.manifest
        )
    )

    validation_identity = (
        _manifest_training_identity(
            validation.manifest
        )
    )

    if (
        train_identity
        != validation_identity
    ):

        raise ValueError(
            "Train and validation DPO artifacts were "
            "materialized against different provenance "
            "environments."
        )

    train_environment = (
        _execution_environment_identity(
            train
            .records[
                0
            ]
            .source_execution_provenance
        )
    )

    validation_environment = (
        _execution_environment_identity(
            validation
            .records[
                0
            ]
            .source_execution_provenance
        )
    )

    if (
        train_environment
        != validation_environment
    ):

        raise ValueError(
            "Train and validation DPO records come from "
            "different specialist execution environments."
        )


# ============================================================
# TRAINING BASE MODEL IDENTITY
# ============================================================


def _assert_runtime_fingerprint_matches_provenance(
    *,
    fingerprint: RuntimeModelArtifactFingerprint,
    record: SpecialistDpoRecord,
) -> None:

    provenance = (
        record.source_execution_provenance
    )

    if provenance is None:

        raise ValueError(
            "Cannot validate training base model because "
            "source execution provenance is missing."
        )

    comparisons = (
        (
            "model artifact",
            fingerprint.content_sha256,
            provenance.model_artifact_sha256,
        ),

        (
            "model weights",
            fingerprint.weights_sha256,
            provenance.model_weights_sha256,
        ),

        (
            "model config",
            fingerprint.config_sha256,
            provenance.model_config_sha256,
        ),

        (
            "generation config",
            fingerprint.generation_config_sha256,
            provenance.generation_config_sha256,
        ),

        (
            "tokenizer artifact",
            fingerprint.tokenizer_sha256,
            provenance.tokenizer_artifact_sha256,
        ),

        (
            "tokenizer config",
            fingerprint.tokenizer_config_sha256,
            provenance.tokenizer_config_sha256,
        ),

        (
            "tokenizer JSON",
            fingerprint.tokenizer_json_sha256,
            provenance.tokenizer_json_sha256,
        ),

        (
            "chat template",
            fingerprint.chat_template_sha256,
            provenance.chat_template_sha256,
        ),
    )

    for (
        label,
        observed,
        expected,
    ) in comparisons:

        if (
            observed
            != expected
        ):

            raise ValueError(
                "Current training base model "
                f"{label} identity does not match "
                "the provenance-verified DPO artifact."
            )


def validate_training_base_model_identity(
    *,
    train: VerifiedDpoPartition,
    validation: VerifiedDpoPartition,
    base_model_path: Path,
) -> RuntimeModelArtifactFingerprint:
    """
    Inspect CURRENT checkpoint bytes from disk immediately before
    trainer construction.

    use_cache=False is deliberate.

    Runtime provenance caching represents the already-loaded model
    process snapshot, while training must verify the checkpoint it
    is about to load right now.
    """

    fingerprint = (
        fingerprint_runtime_model_artifact(
            base_model_path,
            use_cache=False,
        )
    )

    for partition in (
        train,
        validation,
    ):

        manifest = (
            partition.manifest
        )

        if (
            fingerprint.content_sha256
            != manifest.target_model_artifact_sha256
        ):

            raise ValueError(
                "Current training base model artifact "
                "does not match DPO materialization provenance."
            )

        if (
            fingerprint.weights_sha256
            != manifest.target_model_weights_sha256
        ):

            raise ValueError(
                "Current training base model weights "
                "do not match DPO materialization provenance."
            )

        if (
            fingerprint.tokenizer_sha256
            != manifest.target_tokenizer_artifact_sha256
        ):

            raise ValueError(
                "Current training tokenizer does not match "
                "DPO materialization provenance."
            )

        for record in partition.records:

            _assert_runtime_fingerprint_matches_provenance(
                fingerprint=(
                    fingerprint
                ),

                record=(
                    record
                ),
            )

    return fingerprint


# ============================================================
# TRL DATASET FORMAT
# ============================================================


def build_dpo_dataset_rows(
    records: list[
        SpecialistDpoRecord
    ],
) -> list[
    dict
]:

    return [
        {
            "prompt": [
                dict(
                    message
                )

                for message
                in record.prompt_messages
            ],

            "chosen": [
                {
                    "role":
                        "assistant",

                    "content":
                        record.chosen,
                }
            ],

            "rejected": [
                {
                    "role":
                        "assistant",

                    "content":
                        record.rejected,
                }
            ],
        }

        for record
        in records
    ]


# ============================================================
# LEGACY TRAINING MODEL FINGERPRINT
# ============================================================


class ModelArtifactFile(
    BaseModel
):
    path: str

    size_bytes: int

    sha256: str


class ModelArtifactFingerprint(
    BaseModel
):
    model_type: (
        str
        | None
    ) = None

    architectures: list[
        str
    ] = Field(
        default_factory=list
    )

    checkpoint_declares_quantization: bool = False

    content_sha256: str

    files: list[
        ModelArtifactFile
    ] = Field(
        default_factory=list
    )


def fingerprint_base_model(
    model_path: Path,
) -> ModelArtifactFingerprint:
    """
    Existing Phase-4 training artifact fingerprint.

    This deliberately retains its historical hashing semantics so
    prior Phase-4B synthetic manifests remain verifiable.

    Phase 4C.2 training authorization additionally uses the richer
    execution-provenance fingerprint above.
    """

    resolved = (
        model_path
        .expanduser()
        .resolve()
    )

    if not resolved.is_dir():

        raise ValueError(
            "Base model directory does not exist: "
            f"{resolved}"
        )

    config_path = (
        resolved
        / "config.json"
    )

    if not config_path.is_file():

        raise ValueError(
            "Base model is missing config.json: "
            f"{resolved}"
        )

    try:

        config_payload = (
            json.loads(
                config_path.read_text(
                    encoding="utf-8"
                )
            )
        )

    except Exception as exc:

        raise ValueError(
            "Base model config.json is invalid."
        ) from exc

    if (
        config_payload.get(
            "quantization_config"
        )
        is not None
    ):

        raise ValueError(
            "Base model checkpoint already declares "
            "quantization. Phase 4A expects a normal "
            "checkpoint and applies QLoRA 4-bit "
            "quantization at load time."
        )

    candidate_paths: set[
        Path
    ] = set()

    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ):

        path = (
            resolved
            / name
        )

        if path.is_file():

            candidate_paths.add(
                path
            )

    for pattern in (
        "*.safetensors",
        "*.bin",
        "*.index.json",
    ):

        for path in (
            resolved.glob(
                pattern
            )
        ):

            if path.is_file():

                candidate_paths.add(
                    path
                )

    weight_files = [
        path

        for path
        in candidate_paths

        if path.suffix in {
            ".safetensors",
            ".bin",
        }
    ]

    if not weight_files:

        raise ValueError(
            "Base model directory contains no "
            "supported model weight files."
        )

    fingerprints: list[
        ModelArtifactFile
    ] = []

    for path in sorted(
        candidate_paths,

        key=lambda item: (
            str(
                item.relative_to(
                    resolved
                )
            )
        ),
    ):

        fingerprints.append(
            ModelArtifactFile(
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

    aggregate_payload = [
        item.model_dump(
            mode="json"
        )

        for item
        in fingerprints
    ]

    architectures = (
        config_payload.get(
            "architectures"
        )
        or []
    )

    return (
        ModelArtifactFingerprint(
            model_type=(
                config_payload.get(
                    "model_type"
                )
            ),

            architectures=[
                str(
                    value
                )

                for value
                in architectures
            ],

            checkpoint_declares_quantization=False,

            content_sha256=(
                _sha256_text(
                    _canonical_json(
                        aggregate_payload
                    )
                )
            ),

            files=(
                fingerprints
            ),
        )
    )


# ============================================================
# DRY-RUN RESULT
# ============================================================


class DpoQloraDryRunResult(
    BaseModel
):
    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    schema_name: str = Field(
        default=(
            "dpo-qlora-dry-run-result.v1"
        ),
        alias="schema",
    )

    created_at: str

    dry_run_id: str

    source_split_id: str

    target_agent: str

    target_model_key: str

    train_record_count: int

    validation_record_count: int

    processed_train_record_count: int

    processed_validation_record_count: int

    train_artifact_sha256: str

    validation_artifact_sha256: str

    base_model_path: Path

    base_model: ModelArtifactFingerprint

    package_versions: dict[
        str,
        str,
    ]

    trainer_class: str

    model_class: str

    optimizer_class: str

    quantized_4bit: bool

    peft_enabled: bool

    trainable_parameter_count: int

    total_parameter_count: int

    trainable_parameter_percent: float

    first_batch_shape: list[
        int
    ] = Field(
        default_factory=list
    )

    peak_cuda_memory_gib: float = 0.0

    training_executed: bool = False


# ============================================================
# TRAINER DRY RUN
# ============================================================


class SpecialistDpoQloraDryRun:
    """
    Construct the real:

        bitsandbytes QLoRA model
        PEFT LoRA adapter
        TRL DPOTrainer
        processed datasets
        DPO data collator
        optimizer + scheduler

    trainer.train() is never called.

    Phase 4C.2 adds a mandatory fail-closed provenance contract
    before model construction.
    """

    def __init__(
        self,
        *,
        train_directory: Path,
        validation_directory: Path,
        settings: SpecialistDpoQloraSettings,
    ) -> None:

        self.train_directory = (
            train_directory
            .expanduser()
            .resolve()
        )

        self.validation_directory = (
            validation_directory
            .expanduser()
            .resolve()
        )

        self.settings = (
            settings.model_copy(
                deep=True,

                update={
                    "base_model_path":
                        settings
                        .base_model_path
                        .expanduser()
                        .resolve(),

                    "output_root":
                        settings
                        .output_root
                        .expanduser()
                        .resolve(),
                },
            )
        )

    # ========================================================
    # VERSION LOCK
    # ========================================================

    def _assert_versions(
        self,
    ) -> dict[
        str,
        str,
    ]:

        versions = (
            _installed_versions()
        )

        problems = []

        for (
            package,
            expected,
        ) in (
            EXPECTED_TRAINING_VERSIONS
            .items()
        ):

            observed = (
                versions.get(
                    package,
                    "missing",
                )
            )

            if (
                observed
                != expected
            ):

                problems.append(
                    f"{package}: expected "
                    f"{expected}, found {observed}"
                )

        if problems:

            raise ValueError(
                "Training API version lock failed: "
                + "; ".join(
                    problems
                )
            )

        return versions

    # ========================================================
    # INPUTS
    # ========================================================

    def _load_inputs(
        self,
    ) -> tuple[
        VerifiedDpoPartition,
        VerifiedDpoPartition,
    ]:

        train = (
            load_specialist_dpo_partition(
                directory=(
                    self.train_directory
                ),

                expected_partition=(
                    "train"
                ),
            )
        )

        validation = (
            load_specialist_dpo_partition(
                directory=(
                    self.validation_directory
                ),

                expected_partition=(
                    "validation"
                ),
            )
        )

        validate_specialist_dpo_training_contract(
            train=(
                train
            ),

            validation=(
                validation
            ),

            expected_model_key=(
                self.settings
                .expected_model_key
            ),
        )

        if not train.records:

            raise ValueError(
                "Specialist DPO training partition "
                "is empty."
            )

        if not validation.records:

            raise ValueError(
                "Specialist DPO validation partition "
                "is empty."
            )

        return (
            train,
            validation,
        )

    # ========================================================
    # TRAINER CONSTRUCTION
    # ========================================================

    def _construct_trainer(
        self,
        *,
        train: VerifiedDpoPartition,
        validation: VerifiedDpoPartition,
    ):

        import torch

        from datasets import (
            Dataset,
        )

        from peft import (
            LoraConfig,
        )

        from transformers import (
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        from trl import (
            DPOConfig,
            DPOTrainer,
        )

        if not torch.cuda.is_available():

            raise ValueError(
                "CUDA is required for the Phase 4A "
                "QLoRA trainer dry run."
            )

        if (
            self.settings.compute_dtype
            == "bfloat16"
            and not torch.cuda.is_bf16_supported()
        ):

            raise ValueError(
                "BF16 was requested but the CUDA "
                "device does not report BF16 support."
            )

        dtype = (
            torch.bfloat16

            if (
                self.settings.compute_dtype
                == "bfloat16"
            )

            else torch.float16
        )

        train_dataset = (
            Dataset.from_list(
                build_dpo_dataset_rows(
                    train.records
                )
            )
        )

        validation_dataset = (
            Dataset.from_list(
                build_dpo_dataset_rows(
                    validation.records
                )
            )
        )

        tokenizer = (
            AutoTokenizer
            .from_pretrained(
                str(
                    self.settings
                    .base_model_path
                ),

                local_files_only=True,

                fix_mistral_regex=True,
            )
        )

        tokenizer.padding_side = (
            "left"
        )

        if tokenizer.pad_token is None:

            if tokenizer.eos_token is None:

                raise ValueError(
                    "Tokenizer has neither pad_token "
                    "nor eos_token."
                )

            tokenizer.pad_token = (
                tokenizer.eos_token
            )

        quantization_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,

                bnb_4bit_quant_type=(
                    self.settings
                    .bnb_4bit_quant_type
                ),

                bnb_4bit_compute_dtype=(
                    dtype
                ),

                bnb_4bit_use_double_quant=(
                    self.settings
                    .bnb_4bit_use_double_quant
                ),
            )
        )

        peft_config = (
            LoraConfig(
                r=(
                    self.settings
                    .lora_r
                ),

                lora_alpha=(
                    self.settings
                    .lora_alpha
                ),

                lora_dropout=(
                    self.settings
                    .lora_dropout
                ),

                bias=(
                    "none"
                ),

                task_type=(
                    "CAUSAL_LM"
                ),

                target_modules=(
                    self.settings
                    .lora_target_modules
                ),
            )
        )

        trainer_output = (
            self.settings
            .output_root
            / "trainer-work"
            / train.manifest.source_split_id
            / train.manifest.target_agent
        )

        training_args = (
            DPOConfig(
                output_dir=(
                    str(
                        trainer_output
                    )
                ),

                model_init_kwargs={
                    "dtype":
                        self.settings
                        .compute_dtype,

                    "local_files_only":
                        True,
                },

                learning_rate=(
                    self.settings
                    .learning_rate
                ),

                weight_decay=(
                    self.settings
                    .weight_decay
                ),

                warmup_steps=(
                    self.settings
                    .warmup_ratio
                ),

                num_train_epochs=(
                    self.settings
                    .num_train_epochs
                ),

                per_device_train_batch_size=(
                    self.settings
                    .per_device_train_batch_size
                ),

                per_device_eval_batch_size=(
                    self.settings
                    .per_device_eval_batch_size
                ),

                gradient_accumulation_steps=(
                    self.settings
                    .gradient_accumulation_steps
                ),

                max_length=(
                    self.settings
                    .max_length
                ),

                truncation_mode=(
                    "keep_start"
                ),

                gradient_checkpointing=(
                    self.settings
                    .gradient_checkpointing
                ),

                gradient_checkpointing_kwargs={
                    "use_reentrant":
                        False,
                },

                beta=(
                    self.settings
                    .beta
                ),

                loss_type=[
                    "sigmoid"
                ],

                bf16=(
                    self.settings
                    .compute_dtype
                    == "bfloat16"
                ),

                fp16=(
                    self.settings
                    .compute_dtype
                    == "float16"
                ),

                optim=(
                    self.settings
                    .optimizer
                ),

                seed=(
                    self.settings
                    .seed
                ),

                data_seed=(
                    self.settings
                    .seed
                ),

                precompute_ref_log_probs=False,

                disable_dropout=True,

                padding_free=False,

                report_to=(
                    "none"
                ),

                logging_strategy=(
                    "no"
                ),

                save_strategy=(
                    "no"
                ),

                eval_strategy=(
                    "no"
                ),

                dataloader_num_workers=0,

                remove_unused_columns=True,
            )
        )

        trainer = (
            DPOTrainer(
                model=(
                    str(
                        self.settings
                        .base_model_path
                    )
                ),

                ref_model=None,

                args=(
                    training_args
                ),

                train_dataset=(
                    train_dataset
                ),

                eval_dataset=(
                    validation_dataset
                ),

                processing_class=(
                    tokenizer
                ),

                quantization_config=(
                    quantization_config
                ),

                peft_config=(
                    peft_config
                ),
            )
        )

        return trainer

    # ========================================================
    # DRY RUN
    # ========================================================

    def run(
        self,
    ) -> DpoQloraDryRunResult:

        versions = (
            self._assert_versions()
        )

        (
            train,
            validation,
        ) = (
            self._load_inputs()
        )

        # ====================================================
        # PHASE 4C.2-C BASE CHECKPOINT LOCK
        #
        # This executes BEFORE DPOTrainer construction and uses a
        # fresh disk fingerprint rather than runtime cache state.
        # ====================================================

        provenance_model_fingerprint = (
            validate_training_base_model_identity(
                train=(
                    train
                ),

                validation=(
                    validation
                ),

                base_model_path=(
                    self.settings
                    .base_model_path
                ),
            )
        )

        # Existing Phase-4 training fingerprint semantics remain
        # unchanged because historical synthetic manifests depend
        # on this representation.
        model_fingerprint = (
            fingerprint_base_model(
                self.settings
                .base_model_path
            )
        )

        identity_payload = {
            "source_split_id":
                train.manifest.source_split_id,

            "target_agent":
                train.manifest.target_agent,

            "target_model_key":
                train.manifest.target_model_key,

            "train_sha256":
                train.manifest.content_sha256,

            "validation_sha256":
                validation.manifest.content_sha256,

            "base_model_sha256":
                model_fingerprint.content_sha256,

            "provenance_model_sha256":
                provenance_model_fingerprint
                .content_sha256,

            "settings":
                self.settings.model_dump(
                    mode="json",
                    by_alias=True,
                ),

            "package_versions":
                versions,
        }

        dry_run_id = (
            "dpo-dry-"
            + _sha256_text(
                _canonical_json(
                    identity_payload
                )
            )[
                :24
            ]
        )

        trainer = None

        try:

            import torch

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

                torch.cuda.reset_peak_memory_stats()

            trainer = (
                self._construct_trainer(
                    train=(
                        train
                    ),

                    validation=(
                        validation
                    ),
                )
            )

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
                    "training records. Increase max_length "
                    "or fix oversized prompts."
                )

            if (
                processed_validation_count
                != len(
                    validation.records
                )
            ):

                raise ValueError(
                    "TRL preprocessing dropped one or more "
                    "validation records. Increase max_length "
                    "or fix oversized prompts."
                )

            if (
                processed_train_count
                <= 0
            ):

                raise ValueError(
                    "TRL produced an empty training dataset."
                )

            first_example = (
                trainer.train_dataset[
                    0
                ]
            )

            batch = (
                trainer.data_collator(
                    [
                        first_example
                    ]
                )
            )

            if (
                "input_ids"
                not in batch
            ):

                raise ValueError(
                    "DPO data collator did not produce "
                    "input_ids."
                )

            first_batch_shape = [
                int(
                    value
                )

                for value
                in batch[
                    "input_ids"
                ].shape
            ]

            if (
                "completion_mask"
                not in batch
                or int(
                    batch[
                        "completion_mask"
                    ]
                    .sum()
                    .item()
                )
                <= 0
            ):

                raise ValueError(
                    "DPO data collator produced no "
                    "completion learning signal."
                )

            model = (
                trainer.model
            )

            quantized_4bit = (
                bool(
                    getattr(
                        model,
                        "is_loaded_in_4bit",
                        False,
                    )
                )
            )

            if not quantized_4bit:

                raise ValueError(
                    "Trainer model is not loaded in 4-bit."
                )

            peft_enabled = (
                bool(
                    getattr(
                        model,
                        "peft_config",
                        None,
                    )
                )
            )

            if not peft_enabled:

                raise ValueError(
                    "Trainer model is not PEFT-wrapped."
                )

            trainable_parameters = [
                (
                    name,
                    parameter,
                )

                for (
                    name,
                    parameter,
                )
                in model.named_parameters()

                if parameter.requires_grad
            ]

            if not trainable_parameters:

                raise ValueError(
                    "QLoRA model contains no trainable "
                    "parameters."
                )

            unexpected_trainable = [
                name

                for (
                    name,
                    _,
                )
                in trainable_parameters

                if (
                    "lora_"
                    not in name
                )
            ]

            if unexpected_trainable:

                raise ValueError(
                    "Unexpected non-LoRA trainable "
                    "parameters detected: "
                    + ", ".join(
                        unexpected_trainable[
                            :10
                        ]
                    )
                )

            trainable_parameter_count = (
                sum(
                    parameter.numel()

                    for (
                        _,
                        parameter,
                    )
                    in trainable_parameters
                )
            )

            total_parameter_count = (
                sum(
                    parameter.numel()

                    for parameter
                    in model.parameters()
                )
            )

            if (
                total_parameter_count
                <= 0
            ):

                raise ValueError(
                    "Trainer model reports zero parameters."
                )

            trainable_parameter_percent = (
                (
                    trainable_parameter_count
                    / total_parameter_count
                )
                * 100.0
            )

            trainer.create_optimizer_and_scheduler(
                num_training_steps=1
            )

            if trainer.optimizer is None:

                raise ValueError(
                    "Trainer failed to construct optimizer."
                )

            optimizer_class = (
                trainer.optimizer
                .__class__
                .__name__
            )

            peak_cuda_memory_gib = (
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

            return (
                DpoQloraDryRunResult(
                    created_at=(
                        _utc_now()
                    ),

                    dry_run_id=(
                        dry_run_id
                    ),

                    source_split_id=(
                        train.manifest
                        .source_split_id
                    ),

                    target_agent=(
                        train.manifest
                        .target_agent
                    ),

                    target_model_key=(
                        train.manifest
                        .target_model_key
                    ),

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

                    train_artifact_sha256=(
                        train.manifest
                        .content_sha256
                    ),

                    validation_artifact_sha256=(
                        validation.manifest
                        .content_sha256
                    ),

                    base_model_path=(
                        self.settings
                        .base_model_path
                    ),

                    base_model=(
                        model_fingerprint
                    ),

                    package_versions=(
                        versions
                    ),

                    trainer_class=(
                        trainer
                        .__class__
                        .__name__
                    ),

                    model_class=(
                        model
                        .__class__
                        .__name__
                    ),

                    optimizer_class=(
                        optimizer_class
                    ),

                    quantized_4bit=(
                        True
                    ),

                    peft_enabled=(
                        True
                    ),

                    trainable_parameter_count=(
                        trainable_parameter_count
                    ),

                    total_parameter_count=(
                        total_parameter_count
                    ),

                    trainable_parameter_percent=(
                        trainable_parameter_percent
                    ),

                    first_batch_shape=(
                        first_batch_shape
                    ),

                    peak_cuda_memory_gib=(
                        peak_cuda_memory_gib
                    ),

                    training_executed=False,
                )
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
