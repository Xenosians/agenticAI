import hashlib
import json

from pathlib import (
    Path,
)

import pytest

from learning.evidence.execution_provenance import (
    SpecialistExecutionProvenance,
    fingerprint_runtime_model_artifact,
)

from learning.training.dpo_materializer import (
    SpecialistDpoManifest,
    SpecialistDpoRecord,
)

from learning.training.dpo_qlora import (
    SpecialistDpoQloraDryRun,
    SpecialistDpoQloraSettings,
    load_specialist_dpo_partition,
    validate_specialist_dpo_training_contract,
    validate_training_base_model_identity,
)


AGENT_NAME = (
    "account-specialist"
)

MODEL_KEY = (
    "qwen2.5-0.5b-funccall"
)

SYSTEM_PROMPT = (
    "Synthetic worker system prompt."
)

USER_REQUEST = (
    "Check whether jdoe is locked."
)

TASK_INSTRUCTIONS = (
    "Inspect the requested account."
)

TASK_CONTEXT_PREFIX = (
    "Additional task context "
    "from the routing stage:\n"
)


def canonical_json(
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


def sha256_text(
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


def write_model(
    root: Path,
    *,
    weight_bytes: bytes = (
        b"synthetic-model-v1"
    ),
) -> Path:

    model_path = (
        root
        / "model"
    )

    model_path.mkdir(
        parents=True,
        exist_ok=False,
    )

    (
        model_path
        / "config.json"
    ).write_text(
        json.dumps(
            {
                "model_type":
                    "synthetic-qwen",

                "architectures": [
                    "SyntheticQwenForCausalLM"
                ],
            }
        ),
        encoding="utf-8",
    )

    (
        model_path
        / "generation_config.json"
    ).write_text(
        json.dumps(
            {
                "max_new_tokens":
                    256,
            }
        ),
        encoding="utf-8",
    )

    (
        model_path
        / "tokenizer.json"
    ).write_text(
        json.dumps(
            {
                "version":
                    "1.0",
            }
        ),
        encoding="utf-8",
    )

    (
        model_path
        / "tokenizer_config.json"
    ).write_text(
        json.dumps(
            {
                "chat_template":
                    (
                        "{% for message in messages %}"
                        "{{ message['role'] }}:"
                        "{{ message['content'] }}"
                        "{% endfor %}"
                    ),
            }
        ),
        encoding="utf-8",
    )

    (
        model_path
        / "model.safetensors"
    ).write_bytes(
        weight_bytes
    )

    return (
        model_path
    )


def prompt_messages(
) -> list[
    dict[
        str,
        str,
    ]
]:

    return [
        {
            "role":
                "system",

            "content":
                SYSTEM_PROMPT,
        },

        {
            "role":
                "user",

            "content":
                USER_REQUEST,
        },

        {
            "role":
                "user",

            "content":
                (
                    TASK_CONTEXT_PREFIX
                    + TASK_INSTRUCTIONS
                ),
        },
    ]


def provenance_for_model(
    model_path: Path,
) -> SpecialistExecutionProvenance:

    fingerprint = (
        fingerprint_runtime_model_artifact(
            model_path,
            use_cache=False,
        )
    )

    messages = (
        prompt_messages()
    )

    return (
        SpecialistExecutionProvenance(
            provenance_complete=True,

            agent_name=(
                AGENT_NAME
            ),

            model_key=(
                MODEL_KEY
            ),

            backend=(
                "qwen-funccall"
            ),

            quantization=(
                "bnb4"
            ),

            compute_dtype=(
                "bfloat16"
            ),

            device_map=(
                "auto"
            ),

            bnb_4bit_quant_type=(
                "nf4"
            ),

            bnb_4bit_use_double_quant=True,

            model_profile_sha256=(
                "1"
                * 64
            ),

            model_artifact_sha256=(
                fingerprint
                .content_sha256
            ),

            model_weights_sha256=(
                fingerprint
                .weights_sha256
            ),

            model_config_sha256=(
                fingerprint
                .config_sha256
            ),

            generation_config_sha256=(
                fingerprint
                .generation_config_sha256
            ),

            tokenizer_artifact_sha256=(
                fingerprint
                .tokenizer_sha256
            ),

            tokenizer_config_sha256=(
                fingerprint
                .tokenizer_config_sha256
            ),

            tokenizer_json_sha256=(
                fingerprint
                .tokenizer_json_sha256
            ),

            chat_template_sha256=(
                fingerprint
                .chat_template_sha256
            ),

            agent_definition_sha256=(
                "2"
                * 64
            ),

            capability_catalog_sha256=(
                "3"
                * 64
            ),

            system_prompt_sha256=(
                sha256_text(
                    SYSTEM_PROMPT
                )
            ),

            messages_sha256=(
                sha256_text(
                    canonical_json(
                        messages
                    )
                )
            ),

            user_request_sha256=(
                sha256_text(
                    USER_REQUEST
                )
            ),

            task_instructions_sha256=(
                sha256_text(
                    TASK_INSTRUCTIONS
                )
            ),

            max_new_tokens=256,
        )
    )


def tool_call(
    user_id: str,
) -> str:

    return (
        canonical_json(
            [
                {
                    "name":
                        "account_status",

                    "arguments": {
                        "user_id":
                            user_id,
                    },
                }
            ]
        )
    )


def dpo_record(
    *,
    record_id: str,
    provenance: (
        SpecialistExecutionProvenance
        | None
    ),
) -> SpecialistDpoRecord:

    return (
        SpecialistDpoRecord(
            dpo_record_id=(
                f"dpo-{record_id}"
            ),

            source_record_id=(
                record_id
            ),

            source_trajectory_id=(
                f"trajectory-{record_id}"
            ),

            source_correction_id=(
                f"correction-{record_id}"
            ),

            target_agent=(
                AGENT_NAME
            ),

            target_model_key=(
                MODEL_KEY
            ),

            change_type=(
                "arguments"
            ),

            prompt_messages=(
                prompt_messages()
            ),

            chosen=(
                tool_call(
                    "jdoe"
                )
            ),

            rejected=(
                tool_call(
                    "wrong-user"
                )
            ),

            source_execution_provenance=(
                provenance
            ),
        )
    )


def write_partition(
    *,
    root: Path,
    partition: str,
    record: SpecialistDpoRecord,
    provenance: SpecialistExecutionProvenance,
    provenance_enforced: bool = True,
) -> Path:

    directory = (
        root
        / partition
    )

    directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    content = (
        canonical_json(
            record.model_dump(
                mode="json",
                by_alias=True,
            )
        )
        + "\n"
    )

    (
        directory
        / "records.jsonl"
    ).write_text(
        content,
        encoding="utf-8",
    )

    manifest = (
        SpecialistDpoManifest(
            created_at=(
                "2026-09-17T00:00:00+00:00"
            ),

            target_agent=(
                AGENT_NAME
            ),

            target_model_key=(
                MODEL_KEY
            ),

            source_split_id=(
                "split-provenance-test"
            ),

            source_partition=(
                partition
            ),

            source_sha256=(
                f"{partition}-source"
            ),

            record_count=1,

            excluded_record_count=0,

            exclusion_reason_counts={},

            content_sha256=(
                sha256_text(
                    content
                )
            ),

            source_record_ids=[
                record.source_record_id
            ],

            execution_provenance_enforced=(
                provenance_enforced
            ),

            target_model_artifact_sha256=(
                provenance
                .model_artifact_sha256
            ),

            target_model_weights_sha256=(
                provenance
                .model_weights_sha256
            ),

            target_tokenizer_artifact_sha256=(
                provenance
                .tokenizer_artifact_sha256
            ),

            target_model_profile_sha256=(
                provenance
                .model_profile_sha256
            ),

            target_agent_definition_sha256=(
                provenance
                .agent_definition_sha256
            ),

            target_capability_catalog_sha256=(
                provenance
                .capability_catalog_sha256
            ),

            target_system_prompt_sha256=(
                provenance
                .system_prompt_sha256
            ),

            specialist_max_new_tokens=(
                provenance
                .max_new_tokens
            ),
        )
    )

    (
        directory
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

    return (
        directory
    )


def build_pair(
    tmp_path: Path,
    *,
    provenance_enforced: bool = True,
    include_provenance: bool = True,
):

    model_path = (
        write_model(
            tmp_path
            / "base"
        )
    )

    provenance = (
        provenance_for_model(
            model_path
        )
    )

    record_provenance = (
        provenance

        if include_provenance

        else None
    )

    train_directory = (
        write_partition(
            root=(
                tmp_path
                / "train-root"
            ),

            partition=(
                "train"
            ),

            record=(
                dpo_record(
                    record_id=(
                        "train-1"
                    ),

                    provenance=(
                        record_provenance
                    ),
                )
            ),

            provenance=(
                provenance
            ),

            provenance_enforced=(
                provenance_enforced
            ),
        )
    )

    validation_directory = (
        write_partition(
            root=(
                tmp_path
                / "validation-root"
            ),

            partition=(
                "validation"
            ),

            record=(
                dpo_record(
                    record_id=(
                        "validation-1"
                    ),

                    provenance=(
                        record_provenance
                    ),
                )
            ),

            provenance=(
                provenance
            ),

            provenance_enforced=(
                provenance_enforced
            ),
        )
    )

    train = (
        load_specialist_dpo_partition(
            directory=(
                train_directory
            ),

            expected_partition=(
                "train"
            ),
        )
    )

    validation = (
        load_specialist_dpo_partition(
            directory=(
                validation_directory
            ),

            expected_partition=(
                "validation"
            ),
        )
    )

    return (
        model_path,
        train,
        validation,
    )


def test_valid_provenance_enforced_pair_is_trainable(
    tmp_path: Path,
):

    (
        model_path,
        train,
        validation,
    ) = (
        build_pair(
            tmp_path
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
            MODEL_KEY
        ),
    )

    fingerprint = (
        validate_training_base_model_identity(
            train=(
                train
            ),

            validation=(
                validation
            ),

            base_model_path=(
                model_path
            ),
        )
    )

    assert (
        fingerprint.content_sha256
        == train
        .manifest
        .target_model_artifact_sha256
    )


def test_legacy_materialization_remains_readable_but_not_trainable(
    tmp_path: Path,
):

    (
        _model_path,
        train,
        validation,
    ) = (
        build_pair(
            tmp_path,
            provenance_enforced=False,
        )
    )

    assert (
        train.manifest.record_count
        == 1
    )

    with pytest.raises(
        ValueError,
        match=(
            "not provenance-enforced"
        ),
    ):

        validate_specialist_dpo_training_contract(
            train=(
                train
            ),

            validation=(
                validation
            ),

            expected_model_key=(
                MODEL_KEY
            ),
        )


def test_record_without_source_provenance_is_not_trainable(
    tmp_path: Path,
):

    (
        _model_path,
        train,
        validation,
    ) = (
        build_pair(
            tmp_path,
            include_provenance=False,
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "missing source execution provenance"
        ),
    ):

        validate_specialist_dpo_training_contract(
            train=(
                train
            ),

            validation=(
                validation
            ),

            expected_model_key=(
                MODEL_KEY
            ),
        )


def test_prompt_provenance_mismatch_is_refused(
    tmp_path: Path,
):

    model_path = (
        write_model(
            tmp_path
            / "base"
        )
    )

    provenance = (
        provenance_for_model(
            model_path
        )
    )

    broken = (
        provenance.model_copy(
            update={
                "messages_sha256":
                    (
                        "f"
                        * 64
                    ),
            }
        )
    )

    train_directory = (
        write_partition(
            root=(
                tmp_path
                / "train-root"
            ),

            partition=(
                "train"
            ),

            record=(
                dpo_record(
                    record_id=(
                        "train-1"
                    ),

                    provenance=(
                        broken
                    ),
                )
            ),

            provenance=(
                broken
            ),
        )
    )

    validation_directory = (
        write_partition(
            root=(
                tmp_path
                / "validation-root"
            ),

            partition=(
                "validation"
            ),

            record=(
                dpo_record(
                    record_id=(
                        "validation-1"
                    ),

                    provenance=(
                        broken
                    ),
                )
            ),

            provenance=(
                broken
            ),
        )
    )

    train = (
        load_specialist_dpo_partition(
            directory=(
                train_directory
            ),

            expected_partition=(
                "train"
            ),
        )
    )

    validation = (
        load_specialist_dpo_partition(
            directory=(
                validation_directory
            ),

            expected_partition=(
                "validation"
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "messages do not match"
        ),
    ):

        validate_specialist_dpo_training_contract(
            train=(
                train
            ),

            validation=(
                validation
            ),

            expected_model_key=(
                MODEL_KEY
            ),
        )


def test_train_and_validation_provenance_environment_mismatch_is_refused(
    tmp_path: Path,
):

    model_a = (
        write_model(
            tmp_path
            / "model-a",
            weight_bytes=(
                b"model-a"
            ),
        )
    )

    model_b = (
        write_model(
            tmp_path
            / "model-b",
            weight_bytes=(
                b"model-b"
            ),
        )
    )

    provenance_a = (
        provenance_for_model(
            model_a
        )
    )

    provenance_b = (
        provenance_for_model(
            model_b
        )
    )

    train_directory = (
        write_partition(
            root=(
                tmp_path
                / "train-root"
            ),

            partition=(
                "train"
            ),

            record=(
                dpo_record(
                    record_id=(
                        "train-1"
                    ),

                    provenance=(
                        provenance_a
                    ),
                )
            ),

            provenance=(
                provenance_a
            ),
        )
    )

    validation_directory = (
        write_partition(
            root=(
                tmp_path
                / "validation-root"
            ),

            partition=(
                "validation"
            ),

            record=(
                dpo_record(
                    record_id=(
                        "validation-1"
                    ),

                    provenance=(
                        provenance_b
                    ),
                )
            ),

            provenance=(
                provenance_b
            ),
        )
    )

    train = (
        load_specialist_dpo_partition(
            directory=(
                train_directory
            ),

            expected_partition=(
                "train"
            ),
        )
    )

    validation = (
        load_specialist_dpo_partition(
            directory=(
                validation_directory
            ),

            expected_partition=(
                "validation"
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "different provenance environments"
        ),
    ):

        validate_specialist_dpo_training_contract(
            train=(
                train
            ),

            validation=(
                validation
            ),

            expected_model_key=(
                MODEL_KEY
            ),
        )


def test_base_model_changed_after_materialization_is_refused(
    tmp_path: Path,
):

    (
        model_path,
        train,
        validation,
    ) = (
        build_pair(
            tmp_path
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
            MODEL_KEY
        ),
    )

    (
        model_path
        / "model.safetensors"
    ).write_bytes(
        b"checkpoint-was-replaced"
    )

    with pytest.raises(
        ValueError,
        match=(
            "base model artifact"
        ),
    ):

        validate_training_base_model_identity(
            train=(
                train
            ),

            validation=(
                validation
            ),

            base_model_path=(
                model_path
            ),
        )


def test_dry_run_load_inputs_refuses_legacy_artifacts_before_cuda(
    tmp_path: Path,
):

    (
        model_path,
        train,
        validation,
    ) = (
        build_pair(
            tmp_path,
            provenance_enforced=False,
        )
    )

    settings = (
        SpecialistDpoQloraSettings(
            base_model_path=(
                model_path
            ),

            expected_model_key=(
                MODEL_KEY
            ),

            output_root=(
                tmp_path
                / "training-output"
            ),
        )
    )

    dry_run = (
        SpecialistDpoQloraDryRun(
            train_directory=(
                train.directory
            ),

            validation_directory=(
                validation.directory
            ),

            settings=(
                settings
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "not provenance-enforced"
        ),
    ):

        dry_run._load_inputs()
