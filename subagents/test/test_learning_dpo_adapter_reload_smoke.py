from pathlib import (
    Path,
)

import pytest

from learning.run_dpo_adapter_reload_smoke import (
    EXPECTED_ARGUMENTS,
    EXPECTED_LORA_TARGET_MODULES,
    EXPECTED_TOOL,
    VALIDATION_TASK_CONTEXT,
    VALIDATION_USER_REQUEST,
    _extract_input_ids,
    build_parser,
    build_validation_runtime_messages,
    validate_adapter_config,
    validate_training_manifest_contract,
)

from learning.run_dpo_qlora_train_smoke import (
    OneStepTrainingSmokeManifest,
    TrainingAdapterArtifact,
    TrainingArtifactFile,
)

from subagents.core.types import (
    AgentDefinition,
)


def _adapter_artifact(
) -> TrainingAdapterArtifact:

    return (
        TrainingAdapterArtifact(
            directory=(
                Path(
                    "/tmp/"
                    "one-step-test/"
                    "adapter"
                )
            ),

            content_sha256=(
                "a" * 64
            ),

            files=[
                TrainingArtifactFile(
                    path=(
                        "adapter_config.json"
                    ),

                    size_bytes=1,

                    sha256=(
                        "b" * 64
                    ),
                ),

                TrainingArtifactFile(
                    path=(
                        "adapter_model.safetensors"
                    ),

                    size_bytes=1,

                    sha256=(
                        "c" * 64
                    ),
                ),
            ],
        )
    )


def _manifest(
) -> OneStepTrainingSmokeManifest:

    return (
        OneStepTrainingSmokeManifest(
            created_at=(
                "2026-09-16T00:00:00+00:00"
            ),

            run_id=(
                "one-step-test"
            ),

            synthetic_only=True,

            training_executed=True,

            requested_max_steps=1,

            observed_global_step=1,

            source_split_id=(
                "synthetic-one-step-training-smoke"
            ),

            target_agent=(
                "account-specialist"
            ),

            target_model_key=(
                "qwen2.5-0.5b-funccall"
            ),

            max_length=1536,

            gradient_accumulation_steps=1,

            train_record_count=2,

            validation_record_count=1,

            processed_train_record_count=2,

            processed_validation_record_count=1,

            optimizer_class=(
                "AdamW"
            ),

            optimizer_module=(
                "bitsandbytes.optim.adamw"
            ),

            optimizer_bits=8,

            optimizer_is_paged=True,

            trainable_parameter_sha256_before=(
                "1" * 64
            ),

            trainable_parameter_sha256_after=(
                "2" * 64
            ),

            weight_update_observed=True,

            training_loss=(
                0.6931471824645996
            ),

            peak_cuda_allocated_gib=(
                4.0
            ),

            peak_cuda_reserved_gib=(
                5.0
            ),

            base_model_sha256_before=(
                "3" * 64
            ),

            base_model_sha256_after=(
                "3" * 64
            ),

            base_model_unchanged=True,

            adapter=(
                _adapter_artifact()
            ),

            package_versions={
                "torch":
                    "2.13.0",

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
            },
        )
    )


def _adapter_config(
    base_model_path: Path,
) -> dict:

    return {
        "base_model_name_or_path":
            str(
                base_model_path
            ),

        "bias":
            "none",

        "inference_mode":
            True,

        "lora_alpha":
            32,

        "lora_dropout":
            0.05,

        "peft_type":
            "LORA",

        "peft_version":
            "0.20.0",

        "r":
            16,

        "target_modules":
            sorted(
                EXPECTED_LORA_TARGET_MODULES
            ),

        "task_type":
            "CAUSAL_LM",

        "use_dora":
            False,

        "use_qalora":
            False,
    }


def test_valid_one_step_manifest_is_accepted(
):

    validate_training_manifest_contract(
        _manifest()
    )


def test_manifest_refuses_non_synthetic_artifact(
):

    manifest = (
        _manifest()
        .model_copy(
            update={
                "synthetic_only":
                    False,
            }
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "synthetic"
        ),
    ):

        validate_training_manifest_contract(
            manifest
        )


def test_manifest_refuses_more_than_one_step(
):

    manifest = (
        _manifest()
        .model_copy(
            update={
                "observed_global_step":
                    2,
            }
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "exactly one optimizer step"
        ),
    ):

        validate_training_manifest_contract(
            manifest
        )


def test_manifest_requires_weight_change_proof(
):

    manifest = (
        _manifest()
        .model_copy(
            update={
                "weight_update_observed":
                    False,
            }
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "parameter update"
        ),
    ):

        validate_training_manifest_contract(
            manifest
        )


def test_manifest_requires_base_model_immutability(
):

    manifest = (
        _manifest()
        .model_copy(
            update={
                "base_model_unchanged":
                    False,
            }
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "immutability"
        ),
    ):

        validate_training_manifest_contract(
            manifest
        )


def test_valid_adapter_config_is_accepted(
    tmp_path: Path,
):

    base_model_path = (
        tmp_path
        / "model"
    )

    validate_adapter_config(
        payload=(
            _adapter_config(
                base_model_path
            )
        ),

        base_model_path=(
            base_model_path
        ),

        manifest=(
            _manifest()
        ),
    )


def test_adapter_config_refuses_wrong_base_model(
    tmp_path: Path,
):

    expected = (
        tmp_path
        / "expected"
    )

    wrong = (
        tmp_path
        / "wrong"
    )

    payload = (
        _adapter_config(
            wrong
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "base-model path mismatch"
        ),
    ):

        validate_adapter_config(
            payload=(
                payload
            ),

            base_model_path=(
                expected
            ),

            manifest=(
                _manifest()
            ),
        )


def test_adapter_config_requires_exact_lora_modules(
    tmp_path: Path,
):

    base_model_path = (
        tmp_path
        / "model"
    )

    payload = (
        _adapter_config(
            base_model_path
        )
    )

    payload[
        "target_modules"
    ] = [
        "q_proj",
    ]

    with pytest.raises(
        ValueError,
        match=(
            "target modules"
        ),
    ):

        validate_adapter_config(
            payload=(
                payload
            ),

            base_model_path=(
                base_model_path
            ),

            manifest=(
                _manifest()
            ),
        )


def test_runtime_messages_match_current_worker_contract(
):

    agent = (
        AgentDefinition(
            name=(
                "account-specialist"
            ),

            description=(
                "Account specialist."
            ),

            model=(
                "qwen2.5-0.5b-funccall"
            ),

            tools=[
                "account_status",
                "unlock_user",
                "reset_password",
            ],

            max_steps=3,

            system_prompt=(
                "Test account specialist."
            ),
        )
    )

    messages = (
        build_validation_runtime_messages(
            agent
        )
    )

    assert (
        len(
            messages
        )
        == 3
    )

    assert (
        messages[
            1
        ][
            "content"
        ]
        == VALIDATION_USER_REQUEST
    )

    assert (
        messages[
            2
        ][
            "content"
        ]
        == (
            "Additional task context "
            "from the routing stage:\n"
            f"{VALIDATION_TASK_CONTEXT}"
        )
    )


def test_extract_input_ids_handles_flat_list(
):

    result = (
        _extract_input_ids(
            {
                "input_ids":
                    [
                        1,
                        2,
                        3,
                    ]
            }
        )
    )

    assert (
        result
        == [
            1,
            2,
            3,
        ]
    )


def test_extract_input_ids_handles_single_batch(
):

    result = (
        _extract_input_ids(
            {
                "input_ids":
                    [
                        [
                            1,
                            2,
                            3,
                        ]
                    ]
            }
        )
    )

    assert (
        result
        == [
            1,
            2,
            3,
        ]
    )


def test_reload_cli_has_no_training_authorization_switch(
):

    parser = (
        build_parser()
    )

    option_strings = {
        option

        for action
        in parser._actions

        for option
        in action.option_strings
    }

    assert (
        "--allow-training"
        not in option_strings
    )

    assert (
        "--max-steps"
        not in option_strings
    )


def test_expected_runtime_output_contract(
):

    assert (
        EXPECTED_TOOL
        == "account_status"
    )

    assert (
        EXPECTED_ARGUMENTS
        == {
            "user_id":
                "jdoe",
        }
    )