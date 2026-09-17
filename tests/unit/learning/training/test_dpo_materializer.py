import json

from pathlib import (
    Path,
)

import pytest

from config import (
    ModelProfileSettings,
)

from learning.evidence.execution_provenance import (
    SpecialistExecutionProvenance,
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

from learning.training.provenance_gate import (
    SPECIALIST_TRAINING_MAX_NEW_TOKENS,
)

from subagents.core.tooling.capabilities import (
    build_agent_capability_catalog,
)

from subagents.core.tooling.prompt import (
    build_worker_system_prompt,
)


# ============================================================
# TEST FIXTURES
# ============================================================


def write_agent(
    tmp_path: Path,
) -> Path:

    path = (
        tmp_path
        / "account-specialist.md"
    )

    path.write_text(
        """---
name: account-specialist
description: Handles account operations.
tools:
  - account_status
  - unlock_user
model: qwen2.5-0.5b-funccall
max_steps: 3
---

You are an account specialist.
""",
        encoding="utf-8",
    )

    return path


def write_model_artifact(
    tmp_path: Path,
) -> Path:

    model_path = (
        tmp_path
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
        b"synthetic-model-weights-v1"
    )

    return model_path


def build_profile(
    model_path: Path,
) -> ModelProfileSettings:

    return (
        ModelProfileSettings(
            backend=(
                "qwen-funccall"
            ),

            model_path=(
                model_path
            ),

            enabled=True,

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
        )
    )


def materializer(
    tmp_path: Path,
) -> tuple[
    SpecialistDpoMaterializer,
    ModelProfileSettings,
    Path,
]:

    agent_path = (
        write_agent(
            tmp_path
        )
    )

    model_path = (
        write_model_artifact(
            tmp_path
        )
    )

    profile = (
        build_profile(
            model_path
        )
    )

    builder = (
        SpecialistDpoMaterializer(
            agent_definition_path=(
                agent_path
            ),

            model_profile=(
                profile
            ),

            output_root=(
                tmp_path
                / "dpo"
            ),
        )
    )

    return (
        builder,
        profile,
        model_path,
    )


# ============================================================
# PROVENANCE
# ============================================================


def valid_execution_provenance(
    *,
    builder: SpecialistDpoMaterializer,
    profile: ModelProfileSettings,
    user_request: str,
    task_instructions: (
        str
        | None
    ),
) -> SpecialistExecutionProvenance:

    capability_catalog = (
        build_agent_capability_catalog(
            builder.agent,
            include_arguments=True,
        )
    )

    system_prompt = (
        build_worker_system_prompt(
            builder.agent,
            capability_catalog=(
                capability_catalog
            ),
        )
    )

    messages = [
        {
            "role":
                "system",

            "content":
                system_prompt,
        },

        {
            "role":
                "user",

            "content":
                user_request,
        },
    ]

    normalized_instructions = (
        task_instructions.strip()

        if (
            task_instructions
            is not None
            and task_instructions.strip()
        )

        else None
    )

    if (
        normalized_instructions
        is not None
    ):

        messages.append(
            {
                "role":
                    "user",

                "content":
                    (
                        "Additional task context "
                        "from the routing stage:\n"
                        f"{normalized_instructions}"
                    ),
            }
        )

    return (
        build_specialist_execution_provenance(
            agent=(
                builder.agent
            ),

            model_profile=(
                profile
            ),

            capability_catalog=(
                capability_catalog
            ),

            messages=(
                messages
            ),

            user_request=(
                user_request
            ),

            task_instructions=(
                normalized_instructions
            ),

            max_new_tokens=(
                SPECIALIST_TRAINING_MAX_NEW_TOKENS
            ),
        )
    )


# ============================================================
# DATASET RECORD
# ============================================================


def record(
    *,
    builder: SpecialistDpoMaterializer,
    profile: ModelProfileSettings,
    record_id: str = "record-1",
    rejected_agent: str = "account-specialist",
    chosen_agent: str = "account-specialist",
    rejected_tool: str = "account_status",
    chosen_tool: str = "account_status",
    rejected_arguments: dict | None = None,
    chosen_arguments: dict | None = None,
    rejected_answer: str | None = None,
    chosen_answer: str | None = None,
    execution_provenance: (
        SpecialistExecutionProvenance
        | None
        | str
    ) = "auto",
) -> PreferenceDatasetRecord:

    if (
        rejected_arguments
        is None
    ):

        rejected_arguments = {
            "user_id":
                "wrong-user"
        }

    if (
        chosen_arguments
        is None
    ):

        chosen_arguments = {
            "user_id":
                "jdoe"
        }

    user_request = (
        "Is jdoe locked?"
    )

    task_instructions = (
        "Check the requested account."
    )

    if (
        execution_provenance
        == "auto"
    ):

        resolved_provenance = (
            valid_execution_provenance(
                builder=(
                    builder
                ),

                profile=(
                    profile
                ),

                user_request=(
                    user_request
                ),

                task_instructions=(
                    task_instructions
                ),
            )
        )

    else:

        resolved_provenance = (
            execution_provenance
        )

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

            execution_provenance=(
                resolved_provenance
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
                        rejected_agent
                    ),

                    tool=(
                        rejected_tool
                    ),

                    arguments=(
                        rejected_arguments
                    ),

                    answer=(
                        rejected_answer
                    ),
                )
            ),

            chosen=(
                PreferenceOption(
                    agent=(
                        chosen_agent
                    ),

                    tool=(
                        chosen_tool
                    ),

                    arguments=(
                        chosen_arguments
                    ),

                    answer=(
                        chosen_answer
                    ),
                )
            ),

            promotion=(
                DatasetPromotion(
                    promoted_at=(
                        "2026-09-17T00:00:00+00:00"
                    ),

                    promoted_by=(
                        "trusted_review"
                    ),

                    reason=(
                        "Verified synthetic preference."
                    ),
                )
            ),

            dataset_eligible=True,
        )
    )


# ============================================================
# SUCCESS
# ============================================================


def test_argument_correction_materializes_for_target_specialist(
    tmp_path: Path,
):

    (
        builder,
        profile,
        _,
    ) = (
        materializer(
            tmp_path
        )
    )

    result = (
        builder.build(
            records=[
                record(
                    builder=(
                        builder
                    ),

                    profile=(
                        profile
                    ),
                )
            ],

            source_split_id=(
                "split-test"
            ),

            source_partition=(
                "train"
            ),

            source_sha256=(
                "source-hash"
            ),
        )
    )

    assert (
        result.manifest.record_count
        == 1
    )

    assert (
        result.manifest.target_agent
        == "account-specialist"
    )

    assert (
        result.manifest.target_model_key
        == "qwen2.5-0.5b-funccall"
    )

    assert (
        result
        .manifest
        .execution_provenance_enforced
        is True
    )

    assert (
        result
        .manifest
        .target_model_artifact_sha256
        is not None
    )

    payload = (
        json.loads(
            (
                result.output_directory
                / "records.jsonl"
            )
            .read_text(
                encoding="utf-8"
            )
            .splitlines()[
                0
            ]
        )
    )

    assert (
        payload[
            "change_type"
        ]
        == "arguments"
    )

    assert (
        json.loads(
            payload[
                "chosen"
            ]
        )[
            0
        ][
            "arguments"
        ][
            "user_id"
        ]
        == "jdoe"
    )

    assert (
        payload[
            "prompt_messages"
        ][
            1
        ][
            "content"
        ]
        == "Is jdoe locked?"
    )

    assert (
        payload[
            "prompt_messages"
        ][
            2
        ][
            "content"
        ]
        == (
            "Additional task context "
            "from the routing stage:\n"
            "Check the requested account."
        )
    )

    assert (
        payload[
            "source_execution_provenance"
        ][
            "provenance_complete"
        ]
        is True
    )


def test_tool_correction_materializes(
    tmp_path: Path,
):

    (
        builder,
        profile,
        _,
    ) = (
        materializer(
            tmp_path
        )
    )

    source = (
        record(
            builder=(
                builder
            ),

            profile=(
                profile
            ),

            rejected_tool=(
                "unlock_user"
            ),

            chosen_tool=(
                "account_status"
            ),

            rejected_arguments={
                "user_id":
                    "jdoe"
            },

            chosen_arguments={
                "user_id":
                    "jdoe"
            },
        )
    )

    result = (
        builder.build(
            records=[
                source
            ],

            source_split_id=(
                "split-test"
            ),

            source_partition=(
                "train"
            ),

            source_sha256=(
                "source-hash"
            ),
        )
    )

    payload = (
        json.loads(
            (
                result.output_directory
                / "records.jsonl"
            )
            .read_text(
                encoding="utf-8"
            )
            .splitlines()[
                0
            ]
        )
    )

    assert (
        payload[
            "change_type"
        ]
        == "tool"
    )


# ============================================================
# TARGET FILTERING
# ============================================================


def test_router_correction_is_not_specialist_training_data(
    tmp_path: Path,
):

    (
        builder,
        profile,
        _,
    ) = (
        materializer(
            tmp_path
        )
    )

    source = (
        record(
            builder=(
                builder
            ),

            profile=(
                profile
            ),

            rejected_agent=(
                "access-specialist"
            ),

            chosen_agent=(
                "account-specialist"
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "No provenance-verified "
            "target-compatible specialist DPO"
        ),
    ):

        builder.build(
            records=[
                source
            ],

            source_split_id=(
                "split-test"
            ),

            source_partition=(
                "train"
            ),

            source_sha256=(
                "source-hash"
            ),
        )


def test_final_answer_correction_is_not_specialist_training_data(
    tmp_path: Path,
):

    (
        builder,
        profile,
        _,
    ) = (
        materializer(
            tmp_path
        )
    )

    source = (
        record(
            builder=(
                builder
            ),

            profile=(
                profile
            ),

            rejected_arguments={
                "user_id":
                    "jdoe"
            },

            chosen_arguments={
                "user_id":
                    "jdoe"
            },

            rejected_answer=(
                "Wrong answer."
            ),

            chosen_answer=(
                "Correct answer."
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "No provenance-verified "
            "target-compatible specialist DPO"
        ),
    ):

        builder.build(
            records=[
                source
            ],

            source_split_id=(
                "split-test"
            ),

            source_partition=(
                "train"
            ),

            source_sha256=(
                "source-hash"
            ),
        )


def test_different_specialist_is_excluded(
    tmp_path: Path,
):

    (
        builder,
        profile,
        _,
    ) = (
        materializer(
            tmp_path
        )
    )

    source = (
        record(
            builder=(
                builder
            ),

            profile=(
                profile
            ),

            rejected_agent=(
                "access-specialist"
            ),

            chosen_agent=(
                "access-specialist"
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "different_specialist"
        ),
    ):

        builder.build(
            records=[
                source
            ],

            source_split_id=(
                "split-test"
            ),

            source_partition=(
                "train"
            ),

            source_sha256=(
                "source-hash"
            ),
        )


def test_mixed_tool_and_argument_change_is_refused(
    tmp_path: Path,
):

    (
        builder,
        profile,
        _,
    ) = (
        materializer(
            tmp_path
        )
    )

    source = (
        record(
            builder=(
                builder
            ),

            profile=(
                profile
            ),

            rejected_tool=(
                "unlock_user"
            ),

            chosen_tool=(
                "account_status"
            ),

            rejected_arguments={
                "user_id":
                    "wrong-user"
            },

            chosen_arguments={
                "user_id":
                    "jdoe"
            },
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "mixed_specialist_change"
        ),
    ):

        builder.build(
            records=[
                source
            ],

            source_split_id=(
                "split-test"
            ),

            source_partition=(
                "train"
            ),

            source_sha256=(
                "source-hash"
            ),
        )


def test_chosen_tool_must_belong_to_target_specialist(
    tmp_path: Path,
):

    (
        builder,
        profile,
        _,
    ) = (
        materializer(
            tmp_path
        )
    )

    source = (
        record(
            builder=(
                builder
            ),

            profile=(
                profile
            ),

            rejected_tool=(
                "account_status"
            ),

            chosen_tool=(
                "totally_fake_tool"
            ),

            rejected_arguments={
                "user_id":
                    "jdoe"
            },

            chosen_arguments={
                "user_id":
                    "jdoe"
            },
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "chosen_tool_not_allowed"
        ),
    ):

        builder.build(
            records=[
                source
            ],

            source_split_id=(
                "split-test"
            ),

            source_partition=(
                "train"
            ),

            source_sha256=(
                "source-hash"
            ),
        )


# ============================================================
# FAIL-CLOSED PROVENANCE
# ============================================================


def test_missing_execution_provenance_fails_closed(
    tmp_path: Path,
):

    (
        builder,
        profile,
        _,
    ) = (
        materializer(
            tmp_path
        )
    )

    source = (
        record(
            builder=(
                builder
            ),

            profile=(
                profile
            ),

            execution_provenance=None,
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "missing_execution_provenance"
        ),
    ):

        builder.build(
            records=[
                source
            ],

            source_split_id=(
                "split-test"
            ),

            source_partition=(
                "train"
            ),

            source_sha256=(
                "source-hash"
            ),
        )


def test_incomplete_execution_provenance_fails_closed(
    tmp_path: Path,
):

    (
        builder,
        profile,
        _,
    ) = (
        materializer(
            tmp_path
        )
    )

    valid = (
        valid_execution_provenance(
            builder=(
                builder
            ),

            profile=(
                profile
            ),

            user_request=(
                "Is jdoe locked?"
            ),

            task_instructions=(
                "Check the requested account."
            ),
        )
    )

    incomplete = (
        valid.model_copy(
            update={
                "provenance_complete":
                    False,
            }
        )
    )

    source = (
        record(
            builder=(
                builder
            ),

            profile=(
                profile
            ),

            execution_provenance=(
                incomplete
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "incomplete_execution_provenance"
        ),
    ):

        builder.build(
            records=[
                source
            ],

            source_split_id=(
                "split-test"
            ),

            source_partition=(
                "train"
            ),

            source_sha256=(
                "source-hash"
            ),
        )


def test_system_prompt_hash_mismatch_fails_closed(
    tmp_path: Path,
):

    (
        builder,
        profile,
        _,
    ) = (
        materializer(
            tmp_path
        )
    )

    valid = (
        valid_execution_provenance(
            builder=(
                builder
            ),

            profile=(
                profile
            ),

            user_request=(
                "Is jdoe locked?"
            ),

            task_instructions=(
                "Check the requested account."
            ),
        )
    )

    mismatched = (
        valid.model_copy(
            update={
                "system_prompt_sha256":
                    (
                        "b"
                        * 64
                    ),
            }
        )
    )

    source = (
        record(
            builder=(
                builder
            ),

            profile=(
                profile
            ),

            execution_provenance=(
                mismatched
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "execution_provenance_system_prompt_mismatch"
        ),
    ):

        builder.build(
            records=[
                source
            ],

            source_split_id=(
                "split-test"
            ),

            source_partition=(
                "train"
            ),

            source_sha256=(
                "source-hash"
            ),
        )


def test_model_bytes_changed_after_execution_fail_closed(
    tmp_path: Path,
):

    (
        builder,
        profile,
        model_path,
    ) = (
        materializer(
            tmp_path
        )
    )

    source = (
        record(
            builder=(
                builder
            ),

            profile=(
                profile
            ),
        )
    )

    # Historical execution provenance has already fingerprinted v1.
    #
    # Change the on-disk checkpoint afterward.
    #
    # The training gate MUST inspect fresh disk contents rather than
    # trusting the process-snapshot fingerprint cache.
    (
        model_path
        / "model.safetensors"
    ).write_bytes(
        b"synthetic-model-weights-v2"
    )

    with pytest.raises(
        ValueError,
        match=(
            "execution_provenance_model_checkpoint_mismatch"
        ),
    ):

        builder.build(
            records=[
                source
            ],

            source_split_id=(
                "split-test"
            ),

            source_partition=(
                "train"
            ),

            source_sha256=(
                "source-hash"
            ),
        )


def test_request_context_hash_mismatch_fails_closed(
    tmp_path: Path,
):

    (
        builder,
        profile,
        _,
    ) = (
        materializer(
            tmp_path
        )
    )

    valid = (
        valid_execution_provenance(
            builder=(
                builder
            ),

            profile=(
                profile
            ),

            user_request=(
                "Is jdoe locked?"
            ),

            task_instructions=(
                "Check the requested account."
            ),
        )
    )

    mismatched = (
        valid.model_copy(
            update={
                "task_instructions_sha256":
                    (
                        "c"
                        * 64
                    ),
            }
        )
    )

    source = (
        record(
            builder=(
                builder
            ),

            profile=(
                profile
            ),

            execution_provenance=(
                mismatched
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "execution_provenance_request_context_mismatch"
        ),
    ):

        builder.build(
            records=[
                source
            ],

            source_split_id=(
                "split-test"
            ),

            source_partition=(
                "train"
            ),

            source_sha256=(
                "source-hash"
            ),
        )


def test_generation_contract_mismatch_fails_closed(
    tmp_path: Path,
):

    (
        builder,
        profile,
        _,
    ) = (
        materializer(
            tmp_path
        )
    )

    valid = (
        valid_execution_provenance(
            builder=(
                builder
            ),

            profile=(
                profile
            ),

            user_request=(
                "Is jdoe locked?"
            ),

            task_instructions=(
                "Check the requested account."
            ),
        )
    )

    mismatched = (
        valid.model_copy(
            update={
                "max_new_tokens":
                    128,
            }
        )
    )

    source = (
        record(
            builder=(
                builder
            ),

            profile=(
                profile
            ),

            execution_provenance=(
                mismatched
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "execution_provenance_generation_contract_mismatch"
        ),
    ):

        builder.build(
            records=[
                source
            ],

            source_split_id=(
                "split-test"
            ),

            source_partition=(
                "train"
            ),

            source_sha256=(
                "source-hash"
            ),
        )