from __future__ import annotations

import argparse
import gc
import json

from importlib import (
    metadata,
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

from learning.training.dpo_qlora import (
    ModelArtifactFingerprint,
    fingerprint_base_model,
)

from learning.cli.run_dpo_qlora_smoke import (
    DEFAULT_AGENT_PATH,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_PATH,
)

from learning.cli.run_dpo_qlora_train_smoke import (
    OneStepTrainingSmokeManifest,
    TrainingAdapterArtifact,
    fingerprint_adapter_directory,
)

from subagents.core.loader import (
    load_agent_definition,
)

from subagents.core.tool_parser import (
    parse_tool_calls,
)

from subagents.core.tool_prompt import (
    build_worker_system_prompt,
)


# ============================================================
# CURRENT PHASE-4B.2 CONTRACT
# ============================================================


EXPECTED_AGENT = (
    "account-specialist"
)


EXPECTED_MODEL_KEY = (
    "qwen2.5-0.5b-funccall"
)


EXPECTED_SOURCE_SPLIT = (
    "synthetic-one-step-training-smoke"
)


EXPECTED_TOOL = (
    "account_status"
)


EXPECTED_ARGUMENTS = {
    "user_id":
        "jdoe",
}


VALIDATION_USER_REQUEST = (
    "Is jdoe locked?"
)


VALIDATION_TASK_CONTEXT = (
    "Check the account state for jdoe."
)


EXPECTED_LORA_TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


# ============================================================
# HELPERS
# ============================================================


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


def _read_json_object(
    path: Path,
) -> dict[
    str,
    Any,
]:

    try:

        payload = (
            json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        )

    except Exception as exc:

        raise ValueError(
            f"Invalid JSON file: {path}"
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):

        raise ValueError(
            "Expected JSON object in "
            f"{path}."
        )

    return payload


def _artifact_file_signature(
    artifact: TrainingAdapterArtifact,
) -> list[
    tuple[
        str,
        int,
        str,
    ]
]:

    return [
        (
            item.path,
            item.size_bytes,
            item.sha256,
        )

        for item
        in artifact.files
    ]


def _extract_input_ids(
    encoded,
) -> list[
    int
]:

    if isinstance(
        encoded,
        dict,
    ):

        values = (
            encoded[
                "input_ids"
            ]
        )

    else:

        values = (
            encoded[
                "input_ids"
            ]
        )

    if hasattr(
        values,
        "tolist",
    ):

        values = (
            values.tolist()
        )

    if (
        values
        and isinstance(
            values[
                0
            ],
            list,
        )
    ):

        if (
            len(
                values
            )
            != 1
        ):

            raise ValueError(
                "Expected exactly one tokenized "
                "conversation."
            )

        values = (
            values[
                0
            ]
        )

    return [
        int(
            value
        )

        for value
        in values
    ]


# ============================================================
# CURRENT RUNTIME MESSAGE CONTRACT
# ============================================================


def build_validation_runtime_messages(
    agent,
) -> list[
    dict[
        str,
        str,
    ]
]:
    """
    Reproduce AgentRuntime message construction exactly:

        1. capability-aware system prompt
        2. original user request
        3. optional Hub task context

    This validates compatibility against the current runtime.

    Historical exact-prompt provenance is not available in the
    Phase-4B.1 manifest and is intentionally not claimed here.
    """

    return [
        {
            "role":
                "system",

            "content":
                build_worker_system_prompt(
                    agent
                ),
        },

        {
            "role":
                "user",

            "content":
                VALIDATION_USER_REQUEST,
        },

        {
            "role":
                "user",

            "content":
                (
                    "Additional task context "
                    "from the routing stage:\n"
                    f"{VALIDATION_TASK_CONTEXT}"
                ),
        },
    ]


# ============================================================
# MANIFEST CONTRACT
# ============================================================


def validate_training_manifest_contract(
    manifest: OneStepTrainingSmokeManifest,
) -> None:

    if (
        manifest.schema_name
        != "dpo-qlora-one-step-smoke.v1"
    ):

        raise ValueError(
            "Unsupported one-step training "
            f"manifest schema: "
            f"{manifest.schema_name}"
        )

    if (
        manifest.synthetic_only
        is not True
    ):

        raise ValueError(
            "Phase 4B.2 currently accepts only "
            "synthetic Phase-4B.1 artifacts."
        )

    if (
        manifest.training_executed
        is not True
    ):

        raise ValueError(
            "Source artifact does not record "
            "successful training execution."
        )

    if (
        manifest.requested_max_steps
        != 1
        or manifest.observed_global_step
        != 1
    ):

        raise ValueError(
            "Source artifact does not represent "
            "exactly one optimizer step."
        )

    if (
        manifest.weight_update_observed
        is not True
    ):

        raise ValueError(
            "Source artifact does not prove a "
            "LoRA parameter update."
        )

    if (
        manifest.trainable_parameter_sha256_before
        == manifest.trainable_parameter_sha256_after
    ):

        raise ValueError(
            "Source artifact reports identical "
            "pre/post trainable parameter hashes."
        )

    if (
        manifest.base_model_unchanged
        is not True
    ):

        raise ValueError(
            "Source artifact does not prove base "
            "checkpoint immutability."
        )

    if (
        manifest.base_model_sha256_before
        != manifest.base_model_sha256_after
    ):

        raise ValueError(
            "Source artifact contains mismatched "
            "base-model fingerprints."
        )

    if (
        manifest.target_agent
        != EXPECTED_AGENT
    ):

        raise ValueError(
            "Unexpected target agent: "
            f"{manifest.target_agent}"
        )

    if (
        manifest.target_model_key
        != EXPECTED_MODEL_KEY
    ):

        raise ValueError(
            "Unexpected target model key: "
            f"{manifest.target_model_key}"
        )

    if (
        manifest.source_split_id
        != EXPECTED_SOURCE_SPLIT
    ):

        raise ValueError(
            "Unexpected synthetic source split: "
            f"{manifest.source_split_id}"
        )

    if (
        manifest.max_length
        != DEFAULT_MAX_LENGTH
    ):

        raise ValueError(
            "Unexpected sequence length in source "
            f"artifact: {manifest.max_length}"
        )

    if (
        manifest.processed_train_record_count
        != manifest.train_record_count
    ):

        raise ValueError(
            "Source training run dropped one or "
            "more training records."
        )

    if (
        manifest.processed_validation_record_count
        != manifest.validation_record_count
    ):

        raise ValueError(
            "Source training run dropped one or "
            "more validation records."
        )

    if (
        manifest.optimizer_module
        != "bitsandbytes.optim.adamw"
    ):

        raise ValueError(
            "Unexpected optimizer module: "
            f"{manifest.optimizer_module}"
        )

    if (
        manifest.optimizer_bits
        != 8
    ):

        raise ValueError(
            "Source artifact did not use an "
            "8-bit optimizer."
        )

    if (
        manifest.optimizer_is_paged
        is not True
    ):

        raise ValueError(
            "Source artifact did not use a "
            "paged optimizer."
        )


# ============================================================
# ADAPTER CONFIG CONTRACT
# ============================================================


def validate_adapter_config(
    *,
    payload: dict[
        str,
        Any,
    ],
    base_model_path: Path,
    manifest: OneStepTrainingSmokeManifest,
) -> None:

    if (
        payload.get(
            "peft_type"
        )
        != "LORA"
    ):

        raise ValueError(
            "Adapter is not a LoRA adapter."
        )

    if (
        payload.get(
            "task_type"
        )
        != "CAUSAL_LM"
    ):

        raise ValueError(
            "Adapter task_type is not "
            "CAUSAL_LM."
        )

    if (
        payload.get(
            "inference_mode"
        )
        is not True
    ):

        raise ValueError(
            "Saved adapter is not marked for "
            "inference mode."
        )

    if (
        payload.get(
            "bias"
        )
        != "none"
    ):

        raise ValueError(
            "Unexpected LoRA bias configuration."
        )

    if (
        payload.get(
            "r"
        )
        != 16
    ):

        raise ValueError(
            "Unexpected LoRA rank."
        )

    if (
        payload.get(
            "lora_alpha"
        )
        != 32
    ):

        raise ValueError(
            "Unexpected LoRA alpha."
        )

    observed_dropout = (
        payload.get(
            "lora_dropout"
        )
    )

    if (
        not isinstance(
            observed_dropout,
            (
                int,
                float,
            ),
        )
        or abs(
            float(
                observed_dropout
            )
            - 0.05
        )
        > 1e-12
    ):

        raise ValueError(
            "Unexpected LoRA dropout."
        )

    observed_targets = (
        payload.get(
            "target_modules"
        )
    )

    if not isinstance(
        observed_targets,
        list,
    ):

        raise ValueError(
            "Adapter target_modules is not a list."
        )

    if (
        set(
            observed_targets
        )
        != EXPECTED_LORA_TARGET_MODULES
    ):

        raise ValueError(
            "Adapter target modules do not match "
            "the Phase-4B.1 all-linear Qwen target."
        )

    if (
        payload.get(
            "use_dora"
        )
        is not False
    ):

        raise ValueError(
            "Unexpected DoRA configuration."
        )

    if (
        payload.get(
            "use_qalora"
        )
        is not False
    ):

        raise ValueError(
            "Unexpected QALoRA configuration."
        )

    configured_base_path = (
        payload.get(
            "base_model_name_or_path"
        )
    )

    if not isinstance(
        configured_base_path,
        str,
    ):

        raise ValueError(
            "Adapter does not record its "
            "base-model path."
        )

    expected_base = (
        base_model_path
        .expanduser()
        .resolve()
    )

    observed_base = (
        Path(
            configured_base_path
        )
        .expanduser()
        .resolve()
    )

    if (
        observed_base
        != expected_base
    ):

        raise ValueError(
            "Adapter base-model path mismatch: "
            f"adapter={observed_base}, "
            f"expected={expected_base}"
        )

    if (
        payload.get(
            "peft_version"
        )
        != manifest.package_versions.get(
            "peft"
        )
    ):

        raise ValueError(
            "Adapter PEFT version does not match "
            "the source training manifest."
        )


# ============================================================
# VERIFIED SOURCE ARTIFACT
# ============================================================


class VerifiedTrainingRun(
    BaseModel
):
    run_directory: Path

    adapter_directory: Path

    manifest: OneStepTrainingSmokeManifest

    adapter: TrainingAdapterArtifact

    base_model: ModelArtifactFingerprint

    adapter_config: dict[
        str,
        Any,
    ]

    current_package_versions: dict[
        str,
        str,
    ]


def verify_training_run(
    *,
    run_directory: Path,
    base_model_path: Path,
) -> VerifiedTrainingRun:

    run_directory = (
        run_directory
        .expanduser()
        .resolve()
    )

    base_model_path = (
        base_model_path
        .expanduser()
        .resolve()
    )

    manifest_path = (
        run_directory
        / "manifest.json"
    )

    adapter_directory = (
        run_directory
        / "adapter"
    )

    adapter_config_path = (
        adapter_directory
        / "adapter_config.json"
    )

    if not run_directory.is_dir():

        raise ValueError(
            "Training run directory does not exist: "
            f"{run_directory}"
        )

    if not manifest_path.is_file():

        raise ValueError(
            "Training run is missing manifest.json."
        )

    if not adapter_directory.is_dir():

        raise ValueError(
            "Training run is missing adapter directory."
        )

    if not adapter_config_path.is_file():

        raise ValueError(
            "Training adapter is missing "
            "adapter_config.json."
        )

    try:

        manifest = (
            OneStepTrainingSmokeManifest
            .model_validate_json(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )
        )

    except Exception as exc:

        raise ValueError(
            "Invalid Phase-4B.1 training manifest."
        ) from exc

    validate_training_manifest_contract(
        manifest
    )

    if (
        run_directory.name
        != manifest.run_id
    ):

        raise ValueError(
            "Training run directory name does not "
            "match manifest run_id."
        )

    expected_manifest_adapter_path = (
        manifest
        .adapter
        .directory
        .expanduser()
        .resolve()
    )

    if (
        expected_manifest_adapter_path
        != adapter_directory
    ):

        raise ValueError(
            "Manifest adapter directory does not "
            "match the selected training run."
        )

    observed_adapter = (
        fingerprint_adapter_directory(
            adapter_directory
        )
    )

    if (
        observed_adapter.content_sha256
        != manifest.adapter.content_sha256
    ):

        raise ValueError(
            "Adapter content SHA-256 does not "
            "match training manifest."
        )

    if (
        _artifact_file_signature(
            observed_adapter
        )
        != _artifact_file_signature(
            manifest.adapter
        )
    ):

        raise ValueError(
            "Adapter file membership or per-file "
            "fingerprints do not match manifest."
        )

    base_model = (
        fingerprint_base_model(
            base_model_path
        )
    )

    if (
        base_model.content_sha256
        != manifest.base_model_sha256_before
    ):

        raise ValueError(
            "Current base checkpoint does not match "
            "the checkpoint used for training."
        )

    if (
        base_model.content_sha256
        != manifest.base_model_sha256_after
    ):

        raise ValueError(
            "Current base checkpoint does not match "
            "the post-training immutability proof."
        )

    adapter_config = (
        _read_json_object(
            adapter_config_path
        )
    )

    validate_adapter_config(
        payload=(
            adapter_config
        ),

        base_model_path=(
            base_model_path
        ),

        manifest=(
            manifest
        ),
    )

    current_versions = (
        _installed_versions()
    )

    for (
        package,
        trained_version,
    ) in (
        manifest
        .package_versions
        .items()
    ):

        current_version = (
            current_versions.get(
                package,
                "missing",
            )
        )

        if (
            current_version
            != trained_version
        ):

            raise ValueError(
                "Reload environment differs from "
                "training environment: "
                f"{package}: trained="
                f"{trained_version}, current="
                f"{current_version}"
            )

    return (
        VerifiedTrainingRun(
            run_directory=(
                run_directory
            ),

            adapter_directory=(
                adapter_directory
            ),

            manifest=(
                manifest
            ),

            adapter=(
                observed_adapter
            ),

            base_model=(
                base_model
            ),

            adapter_config=(
                adapter_config
            ),

            current_package_versions=(
                current_versions
            ),
        )
    )


# ============================================================
# TOKENIZER / CHAT-TEMPLATE CONTRACT
# ============================================================


class TokenizerCompatibility(
    BaseModel
):
    base_vocab_size: int

    adapter_vocab_size: int

    base_special_token_ids: dict[
        str,
        int | None,
    ]

    adapter_special_token_ids: dict[
        str,
        int | None,
    ]

    chat_template_text_equal: bool

    runtime_prompt_tokens_equal: bool

    runtime_prompt_token_count: int


def _special_token_ids(
    tokenizer,
) -> dict[
    str,
    int | None,
]:

    return {
        "bos_token_id":
            tokenizer.bos_token_id,

        "eos_token_id":
            tokenizer.eos_token_id,

        "pad_token_id":
            tokenizer.pad_token_id,

        "unk_token_id":
            tokenizer.unk_token_id,
    }


def verify_tokenizer_compatibility(
    *,
    base_model_path: Path,
    adapter_directory: Path,
    agent_path: Path,
) -> TokenizerCompatibility:

    from transformers import (
        AutoTokenizer,
    )

    agent = (
        load_agent_definition(
            agent_path
        )
    )

    if (
        agent.name
        != EXPECTED_AGENT
    ):

        raise ValueError(
            "Unexpected specialist definition: "
            f"{agent.name}"
        )

    if (
        agent.model
        != EXPECTED_MODEL_KEY
    ):

        raise ValueError(
            "Current specialist model key differs "
            "from adapter target."
        )

    messages = (
        build_validation_runtime_messages(
            agent
        )
    )

    base_tokenizer = (
        AutoTokenizer
        .from_pretrained(
            base_model_path,
            local_files_only=True,
            fix_mistral_regex=True,
        )
    )

    adapter_tokenizer = (
        AutoTokenizer
        .from_pretrained(
            adapter_directory,
            local_files_only=True,
            fix_mistral_regex=True,
        )
    )

    base_vocab_size = (
        len(
            base_tokenizer
        )
    )

    adapter_vocab_size = (
        len(
            adapter_tokenizer
        )
    )

    if (
        base_vocab_size
        != adapter_vocab_size
    ):

        raise ValueError(
            "Base and saved adapter tokenizer "
            "vocabularies differ."
        )

    base_specials = (
        _special_token_ids(
            base_tokenizer
        )
    )

    adapter_specials = (
        _special_token_ids(
            adapter_tokenizer
        )
    )

    if (
        base_specials
        != adapter_specials
    ):

        raise ValueError(
            "Base and saved adapter tokenizer "
            "special-token IDs differ: "
            f"base={base_specials}, "
            f"adapter={adapter_specials}"
        )

    chat_template_text_equal = (
        base_tokenizer.chat_template
        == adapter_tokenizer.chat_template
    )

    if not chat_template_text_equal:

        raise ValueError(
            "Base and adapter tokenizer chat "
            "templates differ."
        )

    base_encoded = (
        base_tokenizer
        .apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
    )

    adapter_encoded = (
        adapter_tokenizer
        .apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
    )

    base_ids = (
        _extract_input_ids(
            base_encoded
        )
    )

    adapter_ids = (
        _extract_input_ids(
            adapter_encoded
        )
    )

    runtime_prompt_tokens_equal = (
        base_ids
        == adapter_ids
    )

    if not runtime_prompt_tokens_equal:

        raise ValueError(
            "Current runtime prompt tokenization "
            "differs between base and saved adapter "
            "tokenizers."
        )

    if not base_ids:

        raise ValueError(
            "Current runtime prompt tokenized to "
            "an empty sequence."
        )

    if (
        len(
            base_ids
        )
        >= DEFAULT_MAX_LENGTH
    ):

        raise ValueError(
            "Current runtime prompt no longer fits "
            "the Phase-4B training sequence budget: "
            f"prompt={len(base_ids)}, "
            f"max_length={DEFAULT_MAX_LENGTH}"
        )

    return (
        TokenizerCompatibility(
            base_vocab_size=(
                base_vocab_size
            ),

            adapter_vocab_size=(
                adapter_vocab_size
            ),

            base_special_token_ids=(
                base_specials
            ),

            adapter_special_token_ids=(
                adapter_specials
            ),

            chat_template_text_equal=(
                chat_template_text_equal
            ),

            runtime_prompt_tokens_equal=(
                runtime_prompt_tokens_equal
            ),

            runtime_prompt_token_count=(
                len(
                    base_ids
                )
            ),
        )
    )


# ============================================================
# RELOAD RESULT
# ============================================================


class AdapterReloadSmokeResult(
    BaseModel
):
    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    schema_name: str = Field(
        default=(
            "dpo-adapter-reload-smoke.v1"
        ),
        alias="schema",
    )

    source_run_id: str

    source_adapter_sha256: str

    base_model_sha256: str

    target_agent: str

    target_model_key: str

    artifact_verified: bool

    package_versions_verified: bool

    tokenizer_compatible: bool

    historical_prompt_provenance_verified: bool = False

    runtime_prompt_token_count: int

    model_class: str

    quantized_4bit: bool

    peft_enabled: bool

    adapter_trainable_parameter_count: int

    total_trainable_parameter_count: int

    raw_model_output: str

    parsed_tool_name: str

    parsed_arguments: dict[
        str,
        Any,
    ]

    exact_runtime_contract_passed: bool

    peak_cuda_allocated_gib: float


# ============================================================
# HARDWARE RELOAD
# ============================================================


def run_adapter_reload_smoke(
    *,
    verified: VerifiedTrainingRun,
    agent_path: Path,
) -> AdapterReloadSmokeResult:

    import torch

    from peft import (
        PeftModel,
    )

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    if not torch.cuda.is_available():

        raise ValueError(
            "CUDA is required for the Phase-4B.2 "
            "adapter reload smoke."
        )

    if not torch.cuda.is_bf16_supported():

        raise ValueError(
            "Phase-4B.2 requires BF16-capable CUDA."
        )

    tokenizer_compatibility = (
        verify_tokenizer_compatibility(
            base_model_path=(
                DEFAULT_MODEL_PATH
                if verified.base_model is None
                else Path(
                    verified
                    .adapter_config[
                        "base_model_name_or_path"
                    ]
                )
            ),

            adapter_directory=(
                verified.adapter_directory
            ),

            agent_path=(
                agent_path
            ),
        )
    )

    base_model_path = (
        Path(
            verified
            .adapter_config[
                "base_model_name_or_path"
            ]
        )
        .expanduser()
        .resolve()
    )

    agent = (
        load_agent_definition(
            agent_path
        )
    )

    messages = (
        build_validation_runtime_messages(
            agent
        )
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            base_model_path,
            local_files_only=True,
            fix_mistral_regex=True,
        )
    )

    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,

            bnb_4bit_quant_type=(
                "nf4"
            ),

            bnb_4bit_compute_dtype=(
                torch.bfloat16
            ),

            bnb_4bit_use_double_quant=True,
        )
    )

    model = (
        None
    )

    try:

        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()

        base_model = (
            AutoModelForCausalLM
            .from_pretrained(
                base_model_path,

                local_files_only=True,

                dtype=(
                    torch.bfloat16
                ),

                device_map=(
                    "auto"
                ),

                quantization_config=(
                    quantization_config
                ),
            )
        )

        model = (
            PeftModel
            .from_pretrained(
                base_model,

                verified.adapter_directory,

                is_trainable=False,

                local_files_only=True,
            )
        )

        model.eval()

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
                "Reloaded adapter model is not "
                "using the expected 4-bit base."
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
                "Reloaded model is not PEFT-wrapped."
            )

        adapter_trainable_parameter_count = (
            sum(
                parameter.numel()

                for (
                    name,
                    parameter,
                )
                in model.named_parameters()

                if (
                    "lora_"
                    in name
                    and parameter.requires_grad
                )
            )
        )

        if (
            adapter_trainable_parameter_count
            != 0
        ):

            raise ValueError(
                "Reloaded inference adapter still has "
                "trainable LoRA parameters."
            )

        total_trainable_parameter_count = (
            sum(
                parameter.numel()

                for parameter
                in model.parameters()

                if parameter.requires_grad
            )
        )

        if (
            total_trainable_parameter_count
            != 0
        ):

            raise ValueError(
                "Reloaded inference model contains "
                "unexpected trainable parameters."
            )

        inputs = (
            tokenizer
            .apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        )

        model_device = (
            next(
                model.parameters()
            )
            .device
        )

        inputs = (
            inputs.to(
                model_device
            )
        )

        with torch.no_grad():

            output = (
                model.generate(
                    **inputs,

                    max_new_tokens=256,

                    do_sample=False,

                    pad_token_id=(
                        tokenizer
                        .eos_token_id
                    ),
                )
            )

        generated_tokens = (
            output[
                0,
                inputs[
                    "input_ids"
                ].shape[
                    1
                ]:
            ]
        )

        raw_response = (
            tokenizer
            .decode(
                generated_tokens,
                skip_special_tokens=True,
            )
            .strip()
        )

        tool_calls = (
            parse_tool_calls(
                raw_response
            )
        )

        if (
            len(
                tool_calls
            )
            != 1
        ):

            raise ValueError(
                "Reloaded specialist must return "
                "exactly one tool call."
            )

        tool_call = (
            tool_calls[
                0
            ]
        )

        tool_name = (
            tool_call[
                "name"
            ]
        )

        arguments = (
            tool_call[
                "arguments"
            ]
        )

        if (
            tool_name
            not in agent.tools
        ):

            raise ValueError(
                "Reloaded adapter proposed a tool "
                "outside the specialist capability set."
            )

        if (
            tool_name
            != EXPECTED_TOOL
        ):

            raise ValueError(
                "Reloaded adapter selected the wrong "
                "tool for the synthetic validation "
                f"request: {tool_name}"
            )

        if (
            arguments
            != EXPECTED_ARGUMENTS
        ):

            raise ValueError(
                "Reloaded adapter produced incorrect "
                "arguments: "
                f"{arguments}"
            )

        peak_cuda_allocated_gib = (
            float(
                torch.cuda
                .max_memory_allocated()
            )
            / (
                1024
                ** 3
            )
        )

        return (
            AdapterReloadSmokeResult(
                source_run_id=(
                    verified
                    .manifest
                    .run_id
                ),

                source_adapter_sha256=(
                    verified
                    .adapter
                    .content_sha256
                ),

                base_model_sha256=(
                    verified
                    .base_model
                    .content_sha256
                ),

                target_agent=(
                    verified
                    .manifest
                    .target_agent
                ),

                target_model_key=(
                    verified
                    .manifest
                    .target_model_key
                ),

                artifact_verified=True,

                package_versions_verified=True,

                tokenizer_compatible=True,

                # Phase 4B.1 did not persist a historical
                # prompt fingerprint. Do not claim otherwise.
                historical_prompt_provenance_verified=False,

                runtime_prompt_token_count=(
                    tokenizer_compatibility
                    .runtime_prompt_token_count
                ),

                model_class=(
                    model
                    .__class__
                    .__name__
                ),

                quantized_4bit=(
                    quantized_4bit
                ),

                peft_enabled=(
                    peft_enabled
                ),

                adapter_trainable_parameter_count=(
                    adapter_trainable_parameter_count
                ),

                total_trainable_parameter_count=(
                    total_trainable_parameter_count
                ),

                raw_model_output=(
                    raw_response
                ),

                parsed_tool_name=(
                    tool_name
                ),

                parsed_arguments=(
                    arguments
                ),

                exact_runtime_contract_passed=True,

                peak_cuda_allocated_gib=(
                    peak_cuda_allocated_gib
                ),
            )
        )

    finally:

        if model is not None:

            del model

        gc.collect()

        try:

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

        except Exception:

            pass


# ============================================================
# CLI
# ============================================================


def build_parser(
) -> argparse.ArgumentParser:

    parser = (
        argparse.ArgumentParser(
            description=(
                "Phase 4B.2: verify and reload a "
                "synthetic one-step QLoRA adapter, then "
                "execute the current specialist runtime "
                "message/tool-call contract. No training "
                "is performed."
            )
        )
    )

    parser.add_argument(
        "--run-directory",
        type=Path,
        required=True,
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
        "--json",
        action="store_true",
    )

    return parser


def main(
) -> int:

    args = (
        build_parser()
        .parse_args()
    )

    print(
        "Phase 4B.2 DPO/QLoRA Adapter Reload Smoke"
    )

    print(
        "=========================================="
    )

    print(
        "Training execution: DISABLED"
    )

    print(
        f"Run: "
        f"{args.run_directory.expanduser().resolve()}"
    )

    print(
        f"Base model: "
        f"{args.model_path.expanduser().resolve()}"
    )

    print(
        f"Agent: "
        f"{args.agent_path.expanduser().resolve()}"
    )

    print()

    try:

        verified = (
            verify_training_run(
                run_directory=(
                    args.run_directory
                ),

                base_model_path=(
                    args.model_path
                ),
            )
        )

        result = (
            run_adapter_reload_smoke(
                verified=(
                    verified
                ),

                agent_path=(
                    args.agent_path
                ),
            )
        )

    except Exception as exc:

        print()

        print(
            "ADAPTER RELOAD SMOKE: FAIL"
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
            "ADAPTER RELOAD SMOKE: PASS"
        )

        print(
            "--------------------------"
        )

        print(
            f"Source run: "
            f"{result.source_run_id}"
        )

        print(
            f"Adapter SHA-256: "
            f"{result.source_adapter_sha256}"
        )

        print(
            f"Base model SHA-256: "
            f"{result.base_model_sha256}"
        )

        print(
            f"Artifact verified: "
            f"{result.artifact_verified}"
        )

        print(
            f"Package versions verified: "
            f"{result.package_versions_verified}"
        )

        print(
            f"Tokenizer compatible: "
            f"{result.tokenizer_compatible}"
        )

        print(
            "Historical prompt provenance verified: "
            f"{result.historical_prompt_provenance_verified}"
        )

        print(
            "Current runtime prompt tokens: "
            f"{result.runtime_prompt_token_count}"
        )

        print(
            f"Model class: "
            f"{result.model_class}"
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
            "Adapter trainable parameters: "
            f"{result.adapter_trainable_parameter_count}"
        )

        print(
            "Total trainable parameters: "
            f"{result.total_trainable_parameter_count}"
        )

        print(
            f"RAW MODEL OUTPUT: "
            f"{result.raw_model_output}"
        )

        print(
            f"Parsed tool: "
            f"{result.parsed_tool_name}"
        )

        print(
            f"Parsed arguments: "
            f"{result.parsed_arguments}"
        )

        print(
            "Runtime contract passed: "
            f"{result.exact_runtime_contract_passed}"
        )

        print(
            "Peak CUDA allocated: "
            f"{result.peak_cuda_allocated_gib:.3f} GiB"
        )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )