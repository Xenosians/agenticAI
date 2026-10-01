from __future__ import annotations

import gc
import json
import math
import shutil
import uuid

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from config import Settings

from learning.continual.checkpoints import (
    AdapterCheckpointStore,
)
from learning.continual.storage import (
    fingerprint_directory,
    immutable_write_json,
)
from learning.evaluation.eval_suite import (
    load_evaluation_cases,
)
from learning.paths import (
    REPOSITORY_ROOT,
    RUNTIME_LEARNING_ROOT,
)
from learning.training.hub_training_contracts import (
    build_hub_training_environment,
    validate_hub_response_contract,
)
from learning.training.hub_hybrid_qlora import (
    _clean_response,
    _decode_generated,
    _encode_prompt_and_completion,
    _generation_inputs,
    _input_device,
)


DEFAULT_PPO_ROOT = (
    RUNTIME_LEARNING_ROOT
    / "continuous"
    / "ppo-runs"
)


class SandboxPPOSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_episodes: int = Field(
        default=4,
        ge=1,
        le=64,
    )
    ppo_epochs: int = Field(
        default=1,
        ge=1,
        le=8,
    )
    max_new_tokens: int = Field(
        default=192,
        ge=16,
        le=512,
    )

    learning_rate: float = Field(
        default=1e-6,
        gt=0.0,
    )
    value_learning_rate: float = Field(
        default=2e-6,
        gt=0.0,
    )

    clip_range: float = Field(
        default=0.20,
        gt=0.0,
        le=0.5,
    )
    value_coef: float = Field(
        default=0.5,
        ge=0.0,
    )
    kl_coef: float = Field(
        default=0.02,
        ge=0.0,
    )
    max_grad_norm: float = Field(
        default=1.0,
        gt=0.0,
    )

    temperature: float = Field(
        default=0.7,
        gt=0.0,
    )


class SandboxPPOResult(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default="sandbox-sequence-ppo-result.v1",
        alias="schema",
    )

    run_id: str
    source_checkpoint_id: str
    checkpoint_id: str

    episode_count: int
    mean_raw_reward: float
    mean_adjusted_reward: float

    adapter_directory: str
    adapter_sha256: str

    simulator_only: bool = True
    live_tool_execution: bool = False


def _hidden_size(
    model,
) -> int:
    config = getattr(
        model,
        "config",
        None,
    )

    candidates = [
        getattr(
            config,
            "hidden_size",
            None,
        ),
        getattr(
            getattr(
                config,
                "text_config",
                None,
            ),
            "hidden_size",
            None,
        ),
    ]

    for value in candidates:
        if (
            isinstance(
                value,
                int,
            )
            and value > 0
        ):
            return value

    raise RuntimeError(
        "Could not resolve model hidden_size for PPO value head."
    )


def _load_policy(
    *,
    model_path: Path,
    backend: str,
    adapter_directory: Path,
):
    import torch

    from peft import (
        PeftModel,
        prepare_model_for_kbit_training,
    )

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Sandbox PPO requires CUDA."
        )

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(
                compute_dtype
            ),
            bnb_4bit_use_double_quant=True,
        )
    )

    normalized_backend = (
        backend
        .strip()
        .lower()
    )

    if normalized_backend == "ministral":
        from transformers import (
            Mistral3ForConditionalGeneration,
            MistralCommonBackend,
        )

        tokenizer = (
            MistralCommonBackend
            .from_pretrained(
                str(
                    model_path
                )
            )
        )

        base = (
            Mistral3ForConditionalGeneration
            .from_pretrained(
                str(
                    model_path
                ),
                local_files_only=True,
                torch_dtype=(
                    compute_dtype
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
                str(
                    model_path
                ),
                local_files_only=True,
            )
        )

        base = (
            AutoModelForCausalLM
            .from_pretrained(
                str(
                    model_path
                ),
                local_files_only=True,
                torch_dtype=(
                    compute_dtype
                ),
                quantization_config=(
                    quantization_config
                ),
            )
        )

    base = (
        prepare_model_for_kbit_training(
            base,
            use_gradient_checkpointing=False,
        )
    )

    model = (
        PeftModel
        .from_pretrained(
            base,
            str(
                adapter_directory
            ),
            is_trainable=True,
        )
    )

    model.train()

    return (
        model,
        tokenizer,
        torch,
    )


def _extract_hidden_states(
    output,
):
    hidden = getattr(
        output,
        "hidden_states",
        None,
    )

    if hidden is not None:
        return hidden

    for name in [
        "language_model_output",
        "text_model_output",
        "model_output",
    ]:
        nested = getattr(
            output,
            name,
            None,
        )

        if nested is None:
            continue

        hidden = getattr(
            nested,
            "hidden_states",
            None,
        )

        if hidden is not None:
            return hidden

    raise RuntimeError(
        "Model did not expose hidden states required by PPO value head."
    )


def _sequence_stats(
    *,
    model,
    value_head,
    tokenizer,
    prompt_messages: list[
        dict[str, str]
    ],
    completion: str,
    max_length: int = 2048,
):
    import torch

    (
        input_ids,
        attention_mask,
        labels,
    ) = _encode_prompt_and_completion(
        tokenizer=tokenizer,
        prompt_messages=(
            prompt_messages
        ),
        completion=(
            completion
        ),
        max_length=(
            max_length
        ),
    )

    device = _input_device(
        model
    )

    input_ids = input_ids.to(
        device
    )
    attention_mask = (
        attention_mask
        .to(
            device
        )
    )
    labels = labels.to(
        device
    )

    output = model(
        input_ids=input_ids,
        attention_mask=(
            attention_mask
        ),
        output_hidden_states=True,
        use_cache=False,
    )

    logits = output.logits[
        :,
        :-1,
        :
    ]

    targets = input_ids[
        :,
        1:
    ]

    mask = labels[
        :,
        1:
    ] != -100

    log_probs = (
        torch.nn.functional
        .log_softmax(
            logits.float(),
            dim=-1,
        )
    )

    selected = (
        log_probs
        .gather(
            dim=-1,
            index=targets.unsqueeze(
                -1
            ),
        )
        .squeeze(
            -1
        )[
            mask
        ]
    )

    if selected.numel() <= 0:
        raise RuntimeError(
            "PPO completion produced no trainable tokens."
        )

    mean_logprob = selected.mean()

    hidden_states = (
        _extract_hidden_states(
            output
        )
    )

    last_hidden = hidden_states[
        -1
    ][
        :,
        -1,
        :
    ]

    value = (
        value_head(
            last_hidden.float()
        )
        .squeeze(
            -1
        )
        .mean()
    )

    return (
        mean_logprob,
        value,
    )


def score_router_response(
    *,
    response: str,
    case,
    registry,
) -> float:
    """
    Deterministic simulator reward. No provider/tool is called.
    """
    reward = 0.0

    try:
        parsed = json.loads(
            response
        )
    except Exception:
        return -1.0

    if not isinstance(
        parsed,
        dict,
    ):
        return -1.0

    reward += 0.10

    try:
        requests = (
            validate_hub_response_contract(
                response=response,
                registry=registry,
            )
        )

        reward += 0.20

    except Exception:
        requests = []

    expected = case.expected

    expected_routes = (
        expected.routes
        if expected.routes
        is not None
        else []
    )

    observed_routes = [
        request.agent_name
        for request in requests
    ]

    if observed_routes == expected_routes:
        reward += 0.20

    if (
        expected.agent is None
        and not requests
    ):
        reward += 0.20

    if (
        expected.agent is not None
        and len(
            requests
        )
        == 1
        and requests[
            0
        ].agent_name
        == expected.agent
    ):
        reward += 0.10

        intent = requests[
            0
        ].semantic_intent

        if (
            intent is not None
            and expected.tool is not None
            and intent.allowed_tools
            == [
                expected.tool
            ]
        ):
            reward += 0.25

        if (
            intent is not None
            and expected.arguments
            is not None
        ):
            observed_arguments = {
                key:
                    (
                        values[
                            0
                        ]
                        if len(
                            values
                        )
                        == 1
                        else values
                    )
                for (
                    key,
                    values,
                ) in (
                    intent
                    .allowed_arguments
                    .items()
                )
            }

            if (
                observed_arguments
                == expected.arguments
            ):
                reward += 0.15

    return min(
        1.0,
        max(
            -1.0,
            reward,
        ),
    )


def run_sandbox_sequence_ppo(
    *,
    source_checkpoint_id: str,
    cycle_id: str,
    suite: str,
    settings: SandboxPPOSettings,
    output_root: Path = DEFAULT_PPO_ROOT,
) -> SandboxPPOResult:
    """
    Bounded sequence-level PPO.

    Episodes come only from version-controlled orchestrator evaluation cases.
    Rewards are deterministic comparisons against expected routing contracts.
    There is no live Jira/LDAP/Git/tool execution in this function.
    """
    checkpoint_store = (
        AdapterCheckpointStore()
    )

    source_checkpoint = (
        checkpoint_store
        .verify_adapter(
            source_checkpoint_id
        )
    )

    app_settings = Settings()

    model_key = (
        source_checkpoint
        .target_model_key
    )

    profile = (
        app_settings
        .require_model_profile(
            model_key
        )
    )

    if profile.model_path is None:
        raise RuntimeError(
            "PPO target model requires a local model_path."
        )

    eval_path = (
        REPOSITORY_ROOT
        / "learning"
        / "evaluation"
        / "evals"
        / f"{suite}.jsonl"
    )

    cases = [
        case
        for case in (
            load_evaluation_cases(
                eval_path
            )
        )
        if case.target
        == "orchestrator"
    ][
        :settings.max_episodes
    ]

    if not cases:
        raise RuntimeError(
            "No orchestrator simulator cases are available for PPO."
        )

    environment = (
        build_hub_training_environment(
            agent_directory=(
                REPOSITORY_ROOT
                / "subagents"
                / "agents"
            )
        )
    )

    model = None
    tokenizer = None
    torch = None

    run_id = (
        "ppo-sandbox-"
        + uuid.uuid4().hex
    )

    run_dir = (
        output_root
        .expanduser()
        .resolve()
        / run_id
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    try:
        (
            model,
            tokenizer,
            torch,
        ) = _load_policy(
            model_path=(
                profile.model_path
            ),
            backend=(
                profile.backend
            ),
            adapter_directory=Path(
                source_checkpoint
                .adapter_directory
            ),
        )

        device = _input_device(
            model
        )

        value_head = (
            torch.nn.Linear(
                _hidden_size(
                    model
                ),
                1,
            )
            .to(
                device
            )
        )

        policy_parameters = [
            parameter
            for parameter in (
                model.parameters()
            )
            if parameter.requires_grad
        ]

        optimizer = (
            torch.optim.AdamW(
                [
                    {
                        "params":
                            policy_parameters,

                        "lr":
                            settings.learning_rate,
                    },
                    {
                        "params":
                            value_head.parameters(),

                        "lr":
                            settings.value_learning_rate,
                    },
                ]
            )
        )

        samples = []

        model.eval()

        with torch.inference_mode():
            for case in cases:
                prompt_messages = [
                    {
                        "role":
                            "system",

                        "content":
                            environment[
                                "system_prompt"
                            ],
                    },
                    {
                        "role":
                            "user",

                        "content":
                            case.user_request,
                    },
                ]

                inputs = _generation_inputs(
                    tokenizer=tokenizer,
                    prompt_messages=(
                        prompt_messages
                    ),
                )

                inputs = {
                    key:
                        value.to(
                            device
                        )
                    for (
                        key,
                        value,
                    ) in inputs.items()
                    if hasattr(
                        value,
                        "to",
                    )
                }

                input_length = int(
                    inputs[
                        "input_ids"
                    ].shape[
                        -1
                    ]
                )

                generated = model.generate(
                    **inputs,
                    max_new_tokens=(
                        settings
                        .max_new_tokens
                    ),
                    do_sample=True,
                    temperature=(
                        settings
                        .temperature
                    ),
                )[
                    0
                ]

                response = _clean_response(
                    _decode_generated(
                        tokenizer,
                        generated[
                            input_length:
                        ],
                    )
                )

                raw_reward = (
                    score_router_response(
                        response=response,
                        case=case,
                        registry=(
                            environment[
                                "registry"
                            ]
                        ),
                    )
                )

                samples.append(
                    {
                        "case":
                            case,

                        "prompt_messages":
                            prompt_messages,

                        "response":
                            response,

                        "raw_reward":
                            raw_reward,
                    }
                )

        model.train()

        # Snapshot old-policy statistics and base-reference KL.
        for sample in samples:
            with torch.no_grad():
                old_logp, old_value = (
                    _sequence_stats(
                        model=model,
                        value_head=(
                            value_head
                        ),
                        tokenizer=(
                            tokenizer
                        ),
                        prompt_messages=(
                            sample[
                                "prompt_messages"
                            ]
                        ),
                        completion=(
                            sample[
                                "response"
                            ]
                        ),
                    )
                )

                disable_adapter = getattr(
                    model,
                    "disable_adapter",
                    None,
                )

                if not callable(
                    disable_adapter
                ):
                    raise RuntimeError(
                        "PEFT policy does not expose disable_adapter()."
                    )

                with disable_adapter():
                    ref_logp, _ = (
                        _sequence_stats(
                            model=model,
                            value_head=(
                                value_head
                            ),
                            tokenizer=(
                                tokenizer
                            ),
                            prompt_messages=(
                                sample[
                                    "prompt_messages"
                                ]
                            ),
                            completion=(
                                sample[
                                    "response"
                                ]
                            ),
                        )
                    )

            kl = max(
                0.0,
                float(
                    (
                        old_logp
                        - ref_logp
                    )
                    .detach()
                    .cpu()
                    .item()
                ),
            )

            adjusted_reward = (
                float(
                    sample[
                        "raw_reward"
                    ]
                )
                - settings.kl_coef
                * kl
            )

            sample[
                "old_logp"
            ] = old_logp.detach()

            sample[
                "old_value"
            ] = old_value.detach()

            sample[
                "adjusted_reward"
            ] = adjusted_reward

            sample[
                "advantage"
            ] = (
                torch.tensor(
                    adjusted_reward,
                    device=device,
                    dtype=torch.float32,
                )
                - old_value.detach()
            )

            sample[
                "return"
            ] = torch.tensor(
                adjusted_reward,
                device=device,
                dtype=torch.float32,
            )

        for _epoch in range(
            settings.ppo_epochs
        ):
            for sample in samples:
                optimizer.zero_grad(
                    set_to_none=True
                )

                new_logp, new_value = (
                    _sequence_stats(
                        model=model,
                        value_head=(
                            value_head
                        ),
                        tokenizer=(
                            tokenizer
                        ),
                        prompt_messages=(
                            sample[
                                "prompt_messages"
                            ]
                        ),
                        completion=(
                            sample[
                                "response"
                            ]
                        ),
                    )
                )

                ratio = torch.exp(
                    torch.clamp(
                        new_logp
                        - sample[
                            "old_logp"
                        ],
                        min=-8.0,
                        max=8.0,
                    )
                )

                advantage = sample[
                    "advantage"
                ]

                unclipped = (
                    ratio
                    * advantage
                )

                clipped = (
                    torch.clamp(
                        ratio,
                        1.0
                        - settings.clip_range,
                        1.0
                        + settings.clip_range,
                    )
                    * advantage
                )

                policy_loss = (
                    -torch.minimum(
                        unclipped,
                        clipped,
                    )
                )

                value_loss = (
                    torch.nn.functional
                    .mse_loss(
                        new_value,
                        sample[
                            "return"
                        ],
                    )
                )

                loss = (
                    policy_loss
                    + settings.value_coef
                    * value_loss
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    [
                        *policy_parameters,
                        *list(
                            value_head.parameters()
                        ),
                    ],
                    settings.max_grad_norm,
                )

                optimizer.step()

        adapter_dir = (
            run_dir
            / "adapter"
        )

        model.save_pretrained(
            adapter_dir,
            safe_serialization=True,
        )

        torch.save(
            value_head.state_dict(),
            run_dir
            / "value_head.pt",
        )

        adapter_sha = (
            fingerprint_directory(
                adapter_dir
            )
        )

        checkpoint = (
            checkpoint_store
            .register(
                adapter_directory=(
                    adapter_dir
                ),
                base_model_sha256=(
                    source_checkpoint
                    .base_model_sha256
                ),
                source_cycle_id=(
                    cycle_id
                ),
                source_split_id=(
                    run_id
                ),
                target_agent="hub",
                target_model_key=(
                    model_key
                ),
                label=(
                    "Phase-5.6 sandbox PPO "
                    + cycle_id
                ),
            )
        )

        raw_rewards = [
            float(
                sample[
                    "raw_reward"
                ]
            )
            for sample in samples
        ]

        adjusted_rewards = [
            float(
                sample[
                    "adjusted_reward"
                ]
            )
            for sample in samples
        ]

        result = SandboxPPOResult(
            run_id=run_id,
            source_checkpoint_id=(
                source_checkpoint_id
            ),
            checkpoint_id=(
                checkpoint
                .checkpoint_id
            ),
            episode_count=len(
                samples
            ),
            mean_raw_reward=(
                sum(
                    raw_rewards
                )
                / len(
                    raw_rewards
                )
            ),
            mean_adjusted_reward=(
                sum(
                    adjusted_rewards
                )
                / len(
                    adjusted_rewards
                )
            ),
            adapter_directory=str(
                adapter_dir
            ),
            adapter_sha256=(
                adapter_sha
            ),
            simulator_only=True,
            live_tool_execution=False,
        )

        immutable_write_json(
            run_dir
            / "manifest.json",
            result,
        )

        return result

    finally:
        try:
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
        except Exception:
            pass

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
