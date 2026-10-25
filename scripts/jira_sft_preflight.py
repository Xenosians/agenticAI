"""Admit the actual Jira training recipe without loading model weights."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Settings  # noqa: E402
from learning.training.jira_sft import (  # noqa: E402
    _tokenize_record,
    assert_training_versions,
    build_jira_training_arguments,
    jira_validation_guard_arguments,
    validate_jira_training_options,
    verify_jira_sft_corpus,
)

DEFAULT_CORPUS = PROJECT_ROOT / ".runtime" / "learning" / "jira-sft" / "v1"


def sequence_summary(*, tokenizer, records, system_prompt: str, max_length: int) -> dict:
    """Use the trainer's exact prompt/target boundary, including saved task messages.

    This is sequence admission only. It neither loads weights nor establishes
    CUDA availability, training completion, quality or live tool execution.
    """
    prompt_lengths, target_lengths, lengths = [], [], []
    for record in records:
        encoded = _tokenize_record(
            tokenizer=tokenizer, system_prompt=system_prompt,
            record=record, max_length=max_length,
        )
        prompt_lengths.append(sum(value == -100 for value in encoded["labels"]))
        target_lengths.append(sum(value != -100 for value in encoded["labels"]))
        lengths.append(len(encoded["input_ids"]))
    if not lengths:
        raise ValueError("Jira SFT requires non-empty train and validation splits.")
    return {
        "records": len(lengths), "max_prompt_tokens": max(prompt_lengths),
        "max_target_tokens": max(target_lengths), "max_sequence_tokens": max(lengths),
    }


def hardware_summary() -> dict:
    """Check the same CUDA/BF16 prerequisites as the QLoRA trainer."""
    import torch

    if not torch.cuda.is_available():
        raise ValueError("CUDA is required for Jira BLOOMZ QLoRA training.")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("The configured Jira training recipe requires BF16 support.")
    device = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "cuda_device": torch.cuda.get_device_name(device),
        "cuda_free_mib": round(free_bytes / 1024 / 1024, 1),
        "cuda_total_mib": round(total_bytes / 1024 / 1024, 1),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify Jira SFT corpus, sequence, recipe and hardware prerequisites without loading model weights.",
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--max-length", type=int, default=1280)
    parser.add_argument("--worker-prompt-profile", choices=("standard", "compact", "compact_grounded"))
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    args = parser.parse_args(argv)

    validate_jira_training_options(
        max_steps=args.max_steps, max_length=args.max_length,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        seed=42, checkpoint_interval=args.checkpoint_interval,
        early_stopping_patience=args.early_stopping_patience,
    )
    versions = assert_training_versions()
    manifest, train, validation, system_prompt = verify_jira_sft_corpus(
        project_root=PROJECT_ROOT, corpus_directory=args.corpus,
        worker_prompt_profile=args.worker_prompt_profile,
    )
    if not train or not validation:
        raise ValueError("Jira SFT requires non-empty train and validation splits.")
    profile = Settings().require_model_profile(manifest.base_model_key)
    if profile.backend != "hf-causal":
        raise ValueError("Jira SFT currently requires an hf-causal base profile.")
    if profile.model_path is None:
        raise ValueError("Base Jira model profile has no model_path.")
    model_path = profile.model_path.expanduser().resolve()
    if not model_path.is_dir():
        raise ValueError(f"Base model directory does not exist: {model_path}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("BLOOMZ tokenizer has neither pad nor EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    summary = sequence_summary(
        tokenizer=tokenizer, records=[*train, *validation],
        system_prompt=system_prompt, max_length=args.max_length,
    )
    hardware = hardware_summary()
    # Exercise the actual locked constructor and validation/checkpoint settings.
    # No model or optimizer is created, and no CPU fallback changes the recipe.
    with tempfile.TemporaryDirectory(prefix="jira-sft-preflight-") as temporary_output:
        training_args = build_jira_training_arguments(
            output_dir=Path(temporary_output), max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps, seed=42,
            guard_arguments=jira_validation_guard_arguments(args.checkpoint_interval),
        )

    print("JIRA SFT PREFLIGHT")
    print("base model key:", manifest.base_model_key)
    print("base model path:", model_path)
    print("prompt profile:", manifest.worker_prompt_profile)
    print("prompt layout:", manifest.prompt_layout)
    print("train records:", len(train))
    print("validation records:", len(validation))
    for key, value in summary.items():
        print(key.replace("_", " ") + ":", value)
    print("configured train max length:", args.max_length)
    print("configured optimizer step limit:", args.max_steps)
    print("configured early-stopping patience:", args.early_stopping_patience)
    for key, value in hardware.items():
        print(key.replace("_", " ") + ":", value)
    print("locked package versions:", versions)
    print("trainer warmup steps:", training_args.warmup_steps)
    print("trainer optimizer:", training_args.optim)
    print("trainer evaluation interval:", training_args.eval_steps)
    print("trainer checkpoint interval:", training_args.save_steps)
    print("trainer validation-best selection:", training_args.load_best_model_at_end)
    print("TrainingArguments constructor: PASS")
    print("No model weights were loaded. No optimizer was created. No Jira provider call occurred.")
    print("PREFLIGHT: PASS (prerequisites only; training and quality remain unverified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
