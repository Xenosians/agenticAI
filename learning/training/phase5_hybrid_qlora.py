from __future__ import annotations

import gc
import importlib.metadata
import math
import re
import shutil
import uuid

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from config.path_portability import (
    resolve_portable_path,
)

from learning.continual.checkpoints import AdapterCheckpointStore
from learning.continual.storage import (
    fingerprint_directory,
    immutable_write_json,
    load_jsonl_models,
    sha256_file,
)
from learning.paths import (
    REPOSITORY_ROOT,
    RUNTIME_LEARNING_ROOT,
)
from learning.training.phase5_contracts import (
    build_hub_training_environment,
    validate_hub_response_contract,
)
from learning.training.phase5_materializer import (
    Phase5DpoRecord,
    Phase5MaterializationManifest,
    Phase5SftRecord,
)

from subagents.core.definitions.loader import (
    load_agent_directory,
)


DEFAULT_PHASE5_TRAINING_ROOT = (
    RUNTIME_LEARNING_ROOT
    / "phase5"
    / "training-runs"
)


class Phase5TrainingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    compute_dtype: str = "bfloat16"

    lora_r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(
        default=0.05,
        ge=0.0,
        lt=1.0,
    )

    lora_target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    learning_rate: float = Field(
        default=5e-6,
        gt=0.0,
    )
    weight_decay: float = Field(
        default=0.0,
        ge=0.0,
    )
    beta: float = Field(
        default=0.1,
        gt=0.0,
    )

    epochs: int = Field(
        default=3,
        ge=1,
    )
    gradient_accumulation_steps: int = Field(
        default=4,
        ge=1,
    )
    max_length: int = Field(
        default=1024,
        ge=128,
    )
    max_optimizer_steps: int = Field(
        default=64,
        ge=1,
    )
    checkpoint_every_steps: int = Field(
        default=4,
        ge=1,
    )
    early_stop_patience: int = Field(
        default=3,
        ge=1,
    )
    max_generation_tokens: int = Field(
        default=256,
        ge=16,
    )
    seed: int = 42


class Phase5CheckpointObservation(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default="phase5-checkpoint-observation.v1",
        alias="schema",
    )

    step: int
    checkpoint_directory: str

    train_loss: float
    eval_sft_loss: float | None = None
    eval_dpo_loss: float | None = None

    contract_pass_rate: float = Field(
        ge=0.0,
        le=1.0,
    )

    evaluation_metric_name: str = (
        "hub_contract_pass_rate"
    )

    evaluation_metric_value: (
        float
        | None
    ) = None

    score: float


class Phase5TrainingRunManifest(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default="phase5-training-run.v1",
        alias="schema",
    )

    run_id: str
    created_at: str

    materialization_id: str
    materialization_directory: str
    materialization_manifest_sha256: str

    target_model_key: str

    target_component: str = "hub"

    evaluation_contract: str = (
        "hub_contract"
    )

    base_model_path: str
    base_model_sha256_before: str
    base_model_sha256_after: str
    base_model_unchanged: bool

    package_versions: dict[str, str]
    settings: dict[str, Any]

    optimizer_steps: int
    sft_optimizer_steps: int
    dpo_optimizer_steps: int

    checkpoint_observations: list[
        Phase5CheckpointObservation
    ] = Field(default_factory=list)

    best_checkpoint_directory: str
    best_score: float
    early_stopped: bool
    fit_diagnosis: str

    adapter_directory: str
    adapter_sha256: str

    registered_checkpoint_id: str

    training_executed: bool = True
    production_activation_performed: bool = False


def _utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def _installed_versions() -> dict[str, str]:
    result = {}

    for package in [
        "torch",
        "transformers",
        "accelerate",
        "peft",
        "bitsandbytes",
        "trl",
        "datasets",
    ]:
        try:
            result[package] = (
                importlib.metadata.version(
                    package
                )
            )
        except Exception:
            result[package] = "missing"

    return result


def _require_training_packages(
    versions: dict[str, str],
) -> None:
    required = [
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
    ]

    missing = [
        package
        for package in required
        if versions.get(package)
        in {
            None,
            "missing",
        }
    ]

    if missing:
        raise RuntimeError(
            "Missing training dependencies: "
            + ", ".join(missing)
        )


def _clean_response(response: str) -> str:
    response = response.strip()

    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        response,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if fenced:
        response = fenced.group(1).strip()

    response = re.sub(
        r"^(?:<s>\s*)+",
        "",
        response,
        flags=re.IGNORECASE,
    )
    response = re.sub(
        r"(?:\s*</s>)+$",
        "",
        response,
        flags=re.IGNORECASE,
    )

    return response.strip()


def _load_materialization(
    directory: Path,
) -> tuple[
    Phase5MaterializationManifest,
    list[Phase5SftRecord],
    list[Phase5SftRecord],
    list[Phase5DpoRecord],
    list[Phase5DpoRecord],
]:
    directory = (
        resolve_portable_path(
            directory,
            base=REPOSITORY_ROOT,
        )
    )

    manifest_path = (
        directory / "manifest.json"
    )

    if not manifest_path.is_file():
        raise ValueError(
            "Materialization manifest does not exist."
        )

    manifest = (
        Phase5MaterializationManifest
        .model_validate_json(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )
    )

    if not manifest.ready_for_training:
        raise PermissionError(
            "Materialization is not ready for training."
        )

    if (
        manifest.training_authorized is not False
        or manifest.promotion_authorized is not False
    ):
        raise ValueError(
            "Materialization governance flags are invalid."
        )

    specs = [
        (
            "sft-train.jsonl",
            manifest.sft_train_sha256,
            Phase5SftRecord,
        ),
        (
            "sft-validation.jsonl",
            manifest.sft_validation_sha256,
            Phase5SftRecord,
        ),
        (
            "dpo-train.jsonl",
            manifest.dpo_train_sha256,
            Phase5DpoRecord,
        ),
        (
            "dpo-validation.jsonl",
            manifest.dpo_validation_sha256,
            Phase5DpoRecord,
        ),
    ]

    loaded = []

    for filename, expected_sha, model_type in specs:
        path = directory / filename
        observed = sha256_file(path)

        if observed != expected_sha:
            raise ValueError(
                "Materialized partition SHA mismatch: "
                f"{filename}"
            )

        loaded.append(
            load_jsonl_models(
                path,
                model_type,
            )
        )

    return (
        manifest,
        loaded[0],
        loaded[1],
        loaded[2],
        loaded[3],
    )


def _cuda_training_preflight(
    torch,
) -> None:
    """Fail early if the CUDA context is unusable before model loading."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available for Phase-5 training."
        )

    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(
        device_index
    )

    probe = torch.empty(
        1,
        device=f"cuda:{device_index}",
    )

    torch.cuda.synchronize(
        device_index
    )

    del probe
    torch.cuda.empty_cache()

    free_bytes, total_bytes = torch.cuda.mem_get_info(
        device_index
    )

    gib = float(1024 ** 3)

    print(
        "[PHASE5] CUDA preflight "
        f"device={device_index} "
        f"name={properties.name!r} "
        f"free={free_bytes / gib:.2f}GiB "
        f"total={total_bytes / gib:.2f}GiB"
    )



def _cuda_memory_snapshot(
    torch,
    *,
    stage: str,
) -> None:
    if not torch.cuda.is_available():
        return

    device_index = torch.cuda.current_device()

    free_bytes, total_bytes = (
        torch.cuda.mem_get_info(
            device_index
        )
    )

    allocated = (
        torch.cuda.memory_allocated(
            device_index
        )
    )

    reserved = (
        torch.cuda.memory_reserved(
            device_index
        )
    )

    gib = float(
        1024 ** 3
    )

    print(
        "[PHASE5][CUDA] "
        f"stage={stage} "
        f"free={free_bytes / gib:.3f}GiB "
        f"allocated={allocated / gib:.3f}GiB "
        f"reserved={reserved / gib:.3f}GiB "
        f"total={total_bytes / gib:.3f}GiB",
        flush=True,
    )


def _dtype(torch, value: str):
    normalized = value.strip().lower()

    if normalized == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError(
                "BF16 requested but CUDA device does not "
                "report BF16 support."
            )

        return torch.bfloat16

    if normalized == "float16":
        return torch.float16

    raise ValueError(
        "compute_dtype must be bfloat16 or float16."
    )


class _Loaded:
    def __init__(
        self,
        *,
        model,
        tokenizer,
        backend: str,
        torch,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.backend = backend
        self.torch = torch


def _load_model(
    *,
    model_path: Path,
    backend: str,
    settings: Phase5TrainingSettings,
    seed_adapter_directory: Path | None = None,
) -> _Loaded:
    import torch

    from peft import (
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Phase-5 QLoRA training requires CUDA."
        )

    _cuda_training_preflight(
        torch
    )

    dtype = _dtype(
        torch,
        settings.compute_dtype,
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )

    normalized_backend = (
        backend.strip().lower()
    )

    if normalized_backend == "ministral":
        from transformers import (
            Mistral3ForConditionalGeneration,
            MistralCommonBackend,
        )

        tokenizer = (
            MistralCommonBackend
            .from_pretrained(
                str(model_path)
            )
        )

        model = (
            Mistral3ForConditionalGeneration
            .from_pretrained(
                str(model_path),
                local_files_only=True,
                torch_dtype=(
                    dtype
                ),
                quantization_config=(
                    quantization_config
                ),
            )
        )

    else:
        tokenizer = (
            AutoTokenizer
            .from_pretrained(
                str(model_path),
                local_files_only=True,
            )
        )

        model = (
            AutoModelForCausalLM
            .from_pretrained(
                str(model_path),
                local_files_only=True,
                torch_dtype=(
                    dtype
                ),
                quantization_config=(
                    quantization_config
                ),
            )
        )

        if (
            getattr(
                tokenizer,
                "pad_token_id",
                None,
            )
            is None
            and getattr(
                tokenizer,
                "eos_token_id",
                None,
            )
            is not None
        ):
            tokenizer.pad_token = (
                tokenizer.eos_token
            )

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },
    )

    if seed_adapter_directory is not None:
        seed_adapter_directory = (
            resolve_portable_path(
                seed_adapter_directory,
                base=REPOSITORY_ROOT,
            )
        )

        if not seed_adapter_directory.is_dir():
            raise ValueError(
                "Seed adapter directory does not exist: "
                f"{seed_adapter_directory}"
            )

        model = (
            PeftModel
            .from_pretrained(
                model,
                str(
                    seed_adapter_directory
                ),
                is_trainable=True,
            )
        )

    else:
        lora = LoraConfig(
            r=settings.lora_r,
            lora_alpha=settings.lora_alpha,
            lora_dropout=settings.lora_dropout,
            bias="none",
            target_modules=(
                settings.lora_target_modules
            ),
        )

        model = get_peft_model(
            model,
            lora,
        )

    model.train()

    return _Loaded(
        model=model,
        tokenizer=tokenizer,
        backend=normalized_backend,
        torch=torch,
    )


def _input_device(model):
    try:
        embeddings = model.get_input_embeddings()
        device = embeddings.weight.device

        if device.type != "meta":
            return device

    except Exception:
        pass

    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device

    raise RuntimeError(
        "Could not resolve model input device."
    )


def _template_ids(
    tokenizer,
    messages,
    *,
    add_generation_prompt: bool,
):
    try:
        value = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=(
                add_generation_prompt
            ),
        )
    except TypeError:
        # MistralCommonBackend currently owns the Hub's runtime
        # template and does not require the HF tokenizer flag.
        value = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
        )

    if isinstance(value, dict):
        return value["input_ids"][0]

    return value[0]


def _fallback_ids(
    tokenizer,
    messages,
    *,
    add_generation_prompt: bool,
):
    from subagents.llm.runtime.hf_prompt import (
        render_hf_causal_fallback_prompt,
    )

    if add_generation_prompt:
        rendered = (
            render_hf_causal_fallback_prompt(
                messages
            )
        )
    else:
        prompt = (
            render_hf_causal_fallback_prompt(
                messages[:-1]
            )
        )
        rendered = (
            prompt
            + messages[-1]["content"]
        )

    return tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=True,
    )["input_ids"][0]


def _encode_prompt_and_completion(
    *,
    tokenizer,
    prompt_messages: list[dict[str, str]],
    completion: str,
    max_length: int,
):
    full_messages = [
        *prompt_messages,
        {
            "role": "assistant",
            "content": completion,
        },
    ]

    try:
        prompt_ids = _template_ids(
            tokenizer,
            prompt_messages,
            add_generation_prompt=True,
        )
        full_ids = _template_ids(
            tokenizer,
            full_messages,
            add_generation_prompt=False,
        )

    except Exception:
        prompt_ids = _fallback_ids(
            tokenizer,
            prompt_messages,
            add_generation_prompt=True,
        )
        full_ids = _fallback_ids(
            tokenizer,
            full_messages,
            add_generation_prompt=False,
        )

    prompt_ids = prompt_ids.flatten()
    full_ids = full_ids.flatten()

    common = 0
    limit = min(
        int(prompt_ids.numel()),
        int(full_ids.numel()),
    )

    while (
        common < limit
        and int(prompt_ids[common])
        == int(full_ids[common])
    ):
        common += 1

    if common <= 0:
        raise ValueError(
            "Prompt/full encodings do not share a stable prefix."
        )

    if full_ids.numel() > max_length:
        if common >= max_length:
            raise ValueError(
                "Prompt alone exceeds max_length."
            )

        full_ids = full_ids[:max_length]

    if full_ids.numel() <= common:
        raise ValueError(
            "No completion tokens remain after truncation."
        )

    labels = full_ids.clone()
    labels[:common] = -100

    attention_mask = full_ids.new_ones(
        full_ids.shape
    )

    return (
        full_ids.unsqueeze(0),
        attention_mask.unsqueeze(0),
        labels.unsqueeze(0),
    )


def _completion_logprob(
    *,
    model,
    input_ids,
    attention_mask,
    labels,
):
    import torch

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )

    logits = output.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    mask = labels[:, 1:] != -100

    log_probs = (
        torch.nn.functional
        .log_softmax(
            logits.float(),
            dim=-1,
        )
    )

    token_log_probs = log_probs.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    selected = token_log_probs[mask]

    if selected.numel() <= 0:
        raise ValueError(
            "No completion tokens available for DPO."
        )

    return selected.sum()


def _sft_loss(
    *,
    loaded: _Loaded,
    record: Phase5SftRecord,
    max_length: int,
):
    (
        input_ids,
        attention_mask,
        labels,
    ) = _encode_prompt_and_completion(
        tokenizer=loaded.tokenizer,
        prompt_messages=record.prompt_messages,
        completion=record.chosen,
        max_length=max_length,
    )

    device = _input_device(
        loaded.model
    )

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    output = loaded.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
    )

    return output.loss


def _dpo_loss(
    *,
    loaded: _Loaded,
    record: Phase5DpoRecord,
    settings: Phase5TrainingSettings,
):
    torch = loaded.torch

    chosen = _encode_prompt_and_completion(
        tokenizer=loaded.tokenizer,
        prompt_messages=record.prompt_messages,
        completion=record.chosen,
        max_length=settings.max_length,
    )

    rejected = _encode_prompt_and_completion(
        tokenizer=loaded.tokenizer,
        prompt_messages=record.prompt_messages,
        completion=record.rejected,
        max_length=settings.max_length,
    )

    device = _input_device(
        loaded.model
    )

    chosen = tuple(
        value.to(device)
        for value in chosen
    )
    rejected = tuple(
        value.to(device)
        for value in rejected
    )

    policy_chosen = _completion_logprob(
        model=loaded.model,
        input_ids=chosen[0],
        attention_mask=chosen[1],
        labels=chosen[2],
    )

    policy_rejected = _completion_logprob(
        model=loaded.model,
        input_ids=rejected[0],
        attention_mask=rejected[1],
        labels=rejected[2],
    )

    disable_adapter = getattr(
        loaded.model,
        "disable_adapter",
        None,
    )

    if not callable(disable_adapter):
        raise RuntimeError(
            "PEFT model does not expose disable_adapter(); "
            "cannot compute frozen-base DPO reference."
        )

    with torch.no_grad():
        with disable_adapter():
            ref_chosen = _completion_logprob(
                model=loaded.model,
                input_ids=chosen[0],
                attention_mask=chosen[1],
                labels=chosen[2],
            )

            ref_rejected = _completion_logprob(
                model=loaded.model,
                input_ids=rejected[0],
                attention_mask=rejected[1],
                labels=rejected[2],
            )

    policy_ratio = (
        policy_chosen
        - policy_rejected
    )
    reference_ratio = (
        ref_chosen
        - ref_rejected
    )

    logits = settings.beta * (
        policy_ratio
        - reference_ratio
    )

    return (
        -torch.nn.functional
        .logsigmoid(logits)
    )


def _evaluate_losses(
    *,
    loaded: _Loaded,
    sft_records,
    dpo_records,
    settings: Phase5TrainingSettings,
):
    torch = loaded.torch

    loaded.model.eval()

    sft_values = []
    dpo_values = []

    with torch.no_grad():
        for record in sft_records:
            value = _sft_loss(
                loaded=loaded,
                record=record,
                max_length=settings.max_length,
            )
            sft_values.append(
                float(
                    value.detach().float().item()
                )
            )

        for record in dpo_records:
            value = _dpo_loss(
                loaded=loaded,
                record=record,
                settings=settings,
            )
            dpo_values.append(
                float(
                    value.detach().float().item()
                )
            )

    loaded.model.train()

    return (
        (
            sum(sft_values) / len(sft_values)
            if sft_values
            else None
        ),
        (
            sum(dpo_values) / len(dpo_values)
            if dpo_values
            else None
        ),
    )


def _generation_inputs(
    *,
    tokenizer,
    prompt_messages,
):
    try:
        try:
            value = tokenizer.apply_chat_template(
                prompt_messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            )
        except TypeError:
            value = tokenizer.apply_chat_template(
                prompt_messages,
                return_tensors="pt",
                return_dict=True,
            )

        if isinstance(value, dict):
            return value

    except Exception:
        pass

    from subagents.llm.runtime.hf_prompt import (
        render_hf_causal_fallback_prompt,
    )

    prompt = render_hf_causal_fallback_prompt(
        prompt_messages
    )

    return tokenizer(
        prompt,
        return_tensors="pt",
    )


def _decode_generated(
    tokenizer,
    token_ids,
) -> str:
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
        )
    except TypeError:
        return tokenizer.decode(
            token_ids
        )


def _contract_pass_rate(
    *,
    loaded: _Loaded,
    records,
    registry,
    settings: Phase5TrainingSettings,
) -> float:
    if not records:
        return 0.0

    torch = loaded.torch
    loaded.model.eval()
    passed = 0

    with torch.inference_mode():
        for record in records:
            inputs = _generation_inputs(
                tokenizer=loaded.tokenizer,
                prompt_messages=record.prompt_messages,
            )

            device = _input_device(
                loaded.model
            )

            inputs = {
                key:
                    value.to(device)
                for key, value in inputs.items()
                if hasattr(value, "to")
            }

            input_length = int(
                inputs["input_ids"].shape[-1]
            )

            generated = loaded.model.generate(
                **inputs,
                max_new_tokens=(
                    settings.max_generation_tokens
                ),
                do_sample=False,
            )[0]

            response = _decode_generated(
                loaded.tokenizer,
                generated[input_length:],
            )

            response = _clean_response(
                response
            )

            try:
                validate_hub_response_contract(
                    response=response,
                    registry=registry,
                )
            except Exception:
                continue

            passed += 1

    loaded.model.train()

    return passed / len(records)


def _score(
    *,
    contract_pass_rate: float,
    eval_sft_loss: float | None,
    eval_dpo_loss: float | None,
) -> float:
    losses = [
        value
        for value in [
            eval_sft_loss,
            eval_dpo_loss,
        ]
        if value is not None
    ]

    mean_loss = (
        sum(losses) / len(losses)
        if losses
        else 0.0
    )

    return (
        contract_pass_rate
        - min(mean_loss, 20.0) * 0.001
    )


def _save_checkpoint(
    *,
    loaded: _Loaded,
    directory: Path,
) -> str:
    if directory.exists():
        shutil.rmtree(directory)

    directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    loaded.model.save_pretrained(
        directory,
        safe_serialization=True,
    )

    if hasattr(
        loaded.tokenizer,
        "save_pretrained",
    ):
        try:
            loaded.tokenizer.save_pretrained(
                directory
            )
        except Exception:
            pass

    return fingerprint_directory(
        directory
    )


def _best_observation(
    observations,
):
    if not observations:
        raise ValueError(
            "No training checkpoints were observed."
        )

    return max(
        observations,
        key=lambda item: (
            item.score,
            -(
                item.eval_sft_loss
                if item.eval_sft_loss is not None
                else math.inf
            ),
            -item.step,
        ),
    )



def _resolve_contract_path(
    value: str,
) -> Path:

    return resolve_portable_path(
        value,
        base=REPOSITORY_ROOT,
    )


def _validate_target_training_environment(
    *,
    materialization: Phase5MaterializationManifest,
    agent_directory: Path,
):

    if (
        materialization.evaluation_contract
        == "hub_contract"
    ):

        environment = (
            build_hub_training_environment(
                agent_directory=(
                    agent_directory
                )
            )
        )

        if (
            environment[
                "system_prompt_sha256"
            ]
            != materialization
            .hub_system_prompt_sha256

            or environment[
                "capability_catalog_sha256"
            ]
            != materialization
            .hub_capability_catalog_sha256

            or environment[
                "agent_definitions_sha256"
            ]
            != materialization
            .hub_agent_definitions_sha256
        ):
            raise ValueError(
                "Trusted Hub training environment changed "
                "after materialization. Re-materialize "
                "before training."
            )

        return environment

    if (
        materialization.evaluation_contract
        == "developer_sft_loss"
    ):

        if (
            materialization.target_component
            != "developer-specialist"
        ):
            raise ValueError(
                "developer_sft_loss is valid only for "
                "developer-specialist."
            )

        if not (
            materialization
            .target_contract_path

            and materialization
            .target_contract_sha256
        ):
            raise ValueError(
                "Developer materialization has no pinned "
                "specialist contract."
            )

        agents = load_agent_directory(
            agent_directory
        )

        developer = next(
            (
                agent

                for agent
                in agents

                if (
                    agent.name
                    == "developer-specialist"
                )
            ),
            None,
        )

        if developer is None:
            raise ValueError(
                "developer-specialist definition is missing."
            )

        if (
            developer.model
            != materialization.target_model_key
        ):
            raise ValueError(
                "Developer model assignment changed after "
                "materialization."
            )

        expected_contract_path = (
            Path(
                agent_directory
            )
            .expanduser()
            .resolve()
            / "developer-specialist.md"
        )

        observed_contract_path = (
            _resolve_contract_path(
                materialization
                .target_contract_path
            )
        )

        if (
            observed_contract_path
            != expected_contract_path
        ):
            raise ValueError(
                "Developer target contract path changed "
                "after materialization."
            )

        observed_sha = sha256_file(
            observed_contract_path
        )

        if (
            observed_sha
            != materialization
            .target_contract_sha256
        ):
            raise ValueError(
                "Developer specialist contract changed "
                "after materialization."
            )

        return None

    raise ValueError(
        "Unsupported training evaluation contract: "
        + materialization.evaluation_contract
    )


def _target_checkpoint_metrics(
    *,
    materialization: Phase5MaterializationManifest,
    loaded,
    sft_validation,
    environment,
    settings: Phase5TrainingSettings,
    eval_sft_loss: float | None,
    eval_dpo_loss: float | None,
) -> tuple[
    float,
    str,
    float,
    float,
]:

    if (
        materialization.evaluation_contract
        == "hub_contract"
    ):

        if environment is None:
            raise RuntimeError(
                "Hub evaluation environment is missing."
            )

        contract_rate = (
            _contract_pass_rate(
                loaded=loaded,
                records=(
                    sft_validation
                ),
                registry=(
                    environment[
                        "registry"
                    ]
                ),
                settings=settings,
            )
        )

        score = _score(
            contract_pass_rate=(
                contract_rate
            ),
            eval_sft_loss=(
                eval_sft_loss
            ),
            eval_dpo_loss=(
                eval_dpo_loss
            ),
        )

        return (
            contract_rate,
            "hub_contract_pass_rate",
            contract_rate,
            score,
        )

    if (
        materialization.evaluation_contract
        == "developer_sft_loss"
    ):

        if eval_sft_loss is None:
            raise RuntimeError(
                "Developer SFT candidate has no "
                "validation loss."
            )

        # Lower validation loss is better.
        #
        # Phase5 best-observation selection maximizes score,
        # therefore negate the loss.
        score = (
            -float(
                eval_sft_loss
            )
        )

        return (
            0.0,
            "developer_validation_loss",
            float(
                eval_sft_loss
            ),
            score,
        )

    raise ValueError(
        "Unsupported checkpoint evaluation contract: "
        + materialization.evaluation_contract
    )



def _validate_sequence_budget(
    *,
    materialization: Phase5MaterializationManifest,
    settings: Phase5TrainingSettings,
) -> None:

    if (
        materialization
        .evaluation_contract
        != "developer_sft_loss"
    ):
        return

    if not (
        materialization
        .sequence_budget_verified
    ):
        raise PermissionError(
            "Developer materialization has no verified "
            "sequence budget. Re-materialize the corpus bridge."
        )

    budget = (
        materialization
        .sequence_budget_tokens
    )

    if (
        budget is None
        or budget < 128
    ):
        raise ValueError(
            "Developer materialization sequence budget is invalid."
        )

    if (
        settings.max_length
        < budget
    ):
        raise ValueError(
            "Training max_length is smaller than the "
            "materialization's verified sequence budget: "
            f"{settings.max_length} < {budget}."
        )


def train_phase5_adapter(
    *,
    materialization_directory: Path,
    settings: Phase5TrainingSettings,
    allow_training: bool,
    backend: str,
    output_root: Path = DEFAULT_PHASE5_TRAINING_ROOT,
    agent_directory: Path,
    seed_adapter_directory: Path | None = None,
) -> Phase5TrainingRunManifest:
    if allow_training is not True:
        raise PermissionError(
            "Real training requires explicit allow_training=True."
        )

    materialization_directory = (
        resolve_portable_path(
            materialization_directory,
            base=REPOSITORY_ROOT,
        )
    )

    (
        materialization,
        sft_train,
        sft_validation,
        dpo_train,
        dpo_validation,
    ) = _load_materialization(
        materialization_directory
    )

    _validate_sequence_budget(
        materialization=(
            materialization
        ),
        settings=settings,
    )

    versions = _installed_versions()
    _require_training_packages(versions)

    base_model_path = (
        resolve_portable_path(
            materialization.base_model_path,
            base=REPOSITORY_ROOT,
        )
    )

    base_before = fingerprint_directory(
        base_model_path
    )

    if base_before != materialization.base_model_sha256:
        raise ValueError(
            "Base model changed after materialization."
        )

    environment = (
        _validate_target_training_environment(
            materialization=(
                materialization
            ),
            agent_directory=(
                agent_directory
            ),
        )
    )

    run_id = (
        "phase5-train-"
        + uuid.uuid4().hex
    )

    run_root = (
        resolve_portable_path(
            output_root,
            base=REPOSITORY_ROOT,
        )
        / run_id
    )

    run_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    loaded = None
    observations = []

    optimizer_steps = 0
    sft_optimizer_steps = 0
    dpo_optimizer_steps = 0
    early_stopped = False
    recent_non_improving = 0
    micro_steps_since_update = 0
    running_losses = []

    try:
        loaded = _load_model(
            model_path=base_model_path,
            backend=backend,
            settings=settings,
            seed_adapter_directory=(
                seed_adapter_directory
            ),
        )

        torch = loaded.torch
        torch.manual_seed(settings.seed)

        from bitsandbytes.optim import (
            PagedAdamW8bit,
        )

        trainable_parameters = [
            parameter
            for parameter in loaded.model.parameters()
            if parameter.requires_grad
        ]

        if not trainable_parameters:
            raise RuntimeError(
                "No trainable LoRA parameters were created."
            )

        optimizer = PagedAdamW8bit(
            trainable_parameters,
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        dtype = _dtype(
            torch,
            settings.compute_dtype,
        )

        def evaluate_and_checkpoint(
            *,
            train_loss: float,
        ) -> bool:
            nonlocal recent_non_improving

            checkpoint_dir = (
                run_root
                / "checkpoints"
                / f"step-{optimizer_steps:06d}"
            )

            _save_checkpoint(
                loaded=loaded,
                directory=checkpoint_dir,
            )

            eval_sft, eval_dpo = (
                _evaluate_losses(
                    loaded=loaded,
                    sft_records=sft_validation,
                    dpo_records=dpo_validation,
                    settings=settings,
                )
            )

            (
                contract_rate,
                evaluation_metric_name,
                evaluation_metric_value,
                score,
            ) = _target_checkpoint_metrics(
                materialization=(
                    materialization
                ),
                loaded=loaded,
                sft_validation=(
                    sft_validation
                ),
                environment=environment,
                settings=settings,
                eval_sft_loss=(
                    eval_sft
                ),
                eval_dpo_loss=(
                    eval_dpo
                ),
            )

            observation = (
                Phase5CheckpointObservation(
                    step=optimizer_steps,
                    checkpoint_directory=str(
                        checkpoint_dir
                    ),
                    train_loss=train_loss,
                    eval_sft_loss=eval_sft,
                    eval_dpo_loss=eval_dpo,
                    contract_pass_rate=(
                        contract_rate
                    ),

                    evaluation_metric_name=(
                        evaluation_metric_name
                    ),

                    evaluation_metric_value=(
                        evaluation_metric_value
                    ),

                    score=score,
                )
            )

            previous_best = (
                _best_observation(observations)
                if observations
                else None
            )

            observations.append(observation)

            current_best = _best_observation(
                observations
            )

            if (
                previous_best is not None
                and current_best.step
                == previous_best.step
            ):
                recent_non_improving += 1
            else:
                recent_non_improving = 0

            return (
                recent_non_improving
                >= settings.early_stop_patience
            )

        pending_stage: str | None = None
        pending_losses: list[float] = []

        def flush_pending() -> bool:
            nonlocal optimizer_steps
            nonlocal sft_optimizer_steps
            nonlocal dpo_optimizer_steps
            nonlocal micro_steps_since_update
            nonlocal pending_stage
            nonlocal pending_losses

            if micro_steps_since_update <= 0:
                return False

            if pending_stage not in {
                "sft",
                "dpo",
            }:
                raise RuntimeError(
                    "Pending gradient stage is invalid."
                )

            _cuda_memory_snapshot(
                torch,
                stage=(
                    "before_optimizer_step:"
                    + str(
                        optimizer_steps
                        + 1
                    )
                ),
            )

            optimizer.step()

            _cuda_memory_snapshot(
                torch,
                stage=(
                    "after_optimizer_step:"
                    + str(
                        optimizer_steps
                        + 1
                    )
                ),
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            optimizer_steps += 1

            if pending_stage == "sft":
                sft_optimizer_steps += 1
            else:
                dpo_optimizer_steps += 1

            mean_train_loss = (
                sum(
                    pending_losses
                )
                / len(
                    pending_losses
                )
            )

            micro_steps_since_update = 0
            pending_stage = None
            pending_losses = []

            should_checkpoint = (
                optimizer_steps
                % settings.checkpoint_every_steps
                == 0
                or optimizer_steps
                >= settings.max_optimizer_steps
            )

            if should_checkpoint:
                return evaluate_and_checkpoint(
                    train_loss=mean_train_loss
                )

            return False

        def optimize_loss(
            loss,
            *,
            stage: str,
        ) -> bool:
            nonlocal micro_steps_since_update
            nonlocal pending_stage
            nonlocal pending_losses

            if (
                pending_stage is not None
                and pending_stage != stage
            ):
                raise RuntimeError(
                    "SFT and DPO gradients may not share one "
                    "optimizer accumulation window."
                )

            pending_stage = stage

            scaled = (
                loss
                / settings.gradient_accumulation_steps
            )

            _cuda_memory_snapshot(
                torch,
                stage=(
                    "before_backward:"
                    + stage
                ),
            )

            scaled.backward()

            _cuda_memory_snapshot(
                torch,
                stage=(
                    "after_backward:"
                    + stage
                ),
            )

            value = float(
                loss.detach().float().item()
            )

            running_losses.append(
                value
            )

            pending_losses.append(
                value
            )

            micro_steps_since_update += 1

            if (
                micro_steps_since_update
                < settings.gradient_accumulation_steps
            ):
                return False

            return flush_pending()

        stop = False

        for _epoch in range(settings.epochs):
            if stop:
                break

            for record in sft_train:
                if (
                    optimizer_steps
                    >= settings.max_optimizer_steps
                ):
                    stop = True
                    break

                with torch.autocast(
                    device_type="cuda",
                    dtype=dtype,
                ):
                    loss = _sft_loss(
                        loaded=loaded,
                        record=record,
                        max_length=settings.max_length,
                    )

                if optimize_loss(
                    loss,
                    stage="sft",
                ):
                    early_stopped = True
                    stop = True
                    break

            if stop:
                break

            # Keep SFT and DPO optimizer accumulation windows separate.
            if micro_steps_since_update > 0:
                if flush_pending():
                    early_stopped = True
                    stop = True
                    break

            if (
                optimizer_steps
                >= settings.max_optimizer_steps
            ):
                stop = True
                break

            for record in dpo_train:
                if (
                    optimizer_steps
                    >= settings.max_optimizer_steps
                ):
                    stop = True
                    break

                with torch.autocast(
                    device_type="cuda",
                    dtype=dtype,
                ):
                    loss = _dpo_loss(
                        loaded=loaded,
                        record=record,
                        settings=settings,
                    )

                if optimize_loss(
                    loss,
                    stage="dpo",
                ):
                    early_stopped = True
                    stop = True
                    break

        if micro_steps_since_update > 0:
            if flush_pending():
                early_stopped = True

        if optimizer_steps <= 0:
            raise RuntimeError(
                "Training completed without an optimizer step."
            )

        if (
            not observations
            or observations[-1].step != optimizer_steps
        ):
            mean_loss = (
                sum(running_losses) / len(running_losses)
                if running_losses
                else 0.0
            )

            evaluate_and_checkpoint(
                train_loss=mean_loss
            )

        best = _best_observation(
            observations
        )

        best_source = Path(
            best.checkpoint_directory
        )

        best_adapter = (
            run_root / "best-adapter"
        )

        shutil.copytree(
            best_source,
            best_adapter,
        )

        adapter_sha = fingerprint_directory(
            best_adapter
        )

        base_after = fingerprint_directory(
            base_model_path
        )

        if base_after != base_before:
            raise RuntimeError(
                "Base model checkpoint changed during QLoRA training."
            )

        if early_stopped:

            fit_diagnosis = (
                "overfit_guard_triggered"
            )

        elif (
            materialization
            .evaluation_contract
            == "hub_contract"

            and best
            .contract_pass_rate
            < 0.75
        ):

            fit_diagnosis = (
                "underfit_suspected"
            )

        elif (
            materialization
            .evaluation_contract
            == "developer_sft_loss"
        ):

            # Validation loss can select a candidate checkpoint,
            # but does NOT constitute the separate developer
            # held-out evidence needed for promotion.
            fit_diagnosis = (
                "developer_candidate_requires_heldout"
            )

        else:

            fit_diagnosis = (
                "no_guard_signal"
            )

        checkpoint = (
            AdapterCheckpointStore()
            .register(
                adapter_directory=best_adapter,
                base_model_sha256=base_before,
                source_cycle_id=(
                    materialization.source_plan_id
                ),
                source_split_id=(
                    materialization.materialization_id
                ),
                target_agent=(
                    materialization
                    .target_component
                ),
                target_model_key=(
                    materialization.target_model_key
                ),
                label=(
                    "Phase-5 "
                    + materialization
                    .target_component
                    + " "
                    + materialization
                    .chapter_id
                ),
            )
        )

        manifest = Phase5TrainingRunManifest(
            run_id=run_id,
            created_at=_utc_now(),
            materialization_id=(
                materialization.materialization_id
            ),
            materialization_directory=str(
                Path(
                    materialization_directory
                )
                .expanduser()
                .resolve()
            ),
            materialization_manifest_sha256=(
                sha256_file(
                    Path(materialization_directory)
                    / "manifest.json"
                )
                or ""
            ),
            target_model_key=(
                materialization.target_model_key
            ),

            target_component=(
                materialization
                .target_component
            ),

            evaluation_contract=(
                materialization
                .evaluation_contract
            ),

            base_model_path=str(base_model_path),
            base_model_sha256_before=base_before,
            base_model_sha256_after=base_after,
            base_model_unchanged=True,
            package_versions=versions,
            settings=settings.model_dump(
                mode="json"
            ),
            optimizer_steps=optimizer_steps,
            sft_optimizer_steps=sft_optimizer_steps,
            dpo_optimizer_steps=dpo_optimizer_steps,
            checkpoint_observations=observations,
            best_checkpoint_directory=(
                best.checkpoint_directory
            ),
            best_score=best.score,
            early_stopped=early_stopped,
            fit_diagnosis=fit_diagnosis,
            adapter_directory=str(best_adapter),
            adapter_sha256=adapter_sha,
            registered_checkpoint_id=(
                checkpoint.checkpoint_id
            ),
            training_executed=True,
            production_activation_performed=False,
        )

        immutable_write_json(
            run_root / "manifest.json",
            manifest,
        )

        return manifest

    finally:
        if loaded is not None:
            try:
                del loaded.model
            except Exception:
                pass

            try:
                del loaded.tokenizer
            except Exception:
                pass

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass



def train_phase5_hub_adapter(
    *,
    materialization_directory: Path,
    settings: Phase5TrainingSettings,
    allow_training: bool,
    backend: str,
    output_root: Path = DEFAULT_PHASE5_TRAINING_ROOT,
    agent_directory: Path,
    seed_adapter_directory: Path | None = None,
) -> Phase5TrainingRunManifest:
    """
    Backwards-compatible Hub entry point.

    Target behavior is now determined by the immutable
    materialization evaluation_contract. Existing callers remain valid.
    """

    return train_phase5_adapter(
        materialization_directory=(
            materialization_directory
        ),
        settings=settings,
        allow_training=allow_training,
        backend=backend,
        output_root=output_root,
        agent_directory=(
            agent_directory
        ),
        seed_adapter_directory=(
            seed_adapter_directory
        ),
    )
