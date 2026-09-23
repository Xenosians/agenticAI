from __future__ import annotations

import argparse
import sys
import tempfile

from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from config import Settings  # noqa: E402

from learning.training.jira_sft import (  # noqa: E402
    assert_training_versions,
    build_jira_training_arguments,
    build_runtime_messages,
    verify_jira_sft_corpus,
)

from subagents.llm.runtime.hf_prompt import (  # noqa: E402
    render_hf_causal_fallback_prompt,
)


DEFAULT_CORPUS = (
    PROJECT_ROOT
    / ".runtime"
    / "learning"
    / "jira-sft"
    / "v1"
)


def _render_prompt(
    tokenizer,
    messages,
) -> str:
    template = getattr(
        tokenizer,
        "chat_template",
        None,
    )

    if (
        isinstance(template, str)
        and template.strip()
    ):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return render_hf_causal_fallback_prompt(
        messages
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify Jira BLOOMZ SFT prerequisites without loading model weights."
        )
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
    )

    args = parser.parse_args()

    versions = assert_training_versions()

    (
        manifest,
        train,
        validation,
        system_prompt,
    ) = verify_jira_sft_corpus(
        project_root=PROJECT_ROOT,
        corpus_directory=args.corpus,
    )

    settings = Settings()
    profile = settings.require_model_profile(
        manifest.base_model_key
    )

    if profile.model_path is None:
        raise SystemExit(
            "Base Jira model profile has no model_path."
        )

    model_path = (
        profile.model_path
        .expanduser()
        .resolve()
    )

    if not model_path.is_dir():
        raise SystemExit(
            f"Base model directory does not exist: {model_path}"
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
    )

    all_records = [
        *train,
        *validation,
    ]

    prompt_lengths = []
    target_lengths = []

    for record in all_records:
        messages = build_runtime_messages(
            system_prompt=system_prompt,
            user_request=record.user_request,
            semantic_context=record.semantic_context,
        )

        prompt = _render_prompt(
            tokenizer,
            messages,
        )

        prompt_lengths.append(
            len(
                tokenizer(
                    prompt,
                    add_special_tokens=True,
                )[
                    "input_ids"
                ]
            )
        )

        target_lengths.append(
            len(
                tokenizer(
                    record.target_response,
                    add_special_tokens=False,
                )[
                    "input_ids"
                ]
            )
        )

    # Instantiate the exact Trainer configuration before model weights
    # are touched. This catches Transformers TrainingArguments API drift
    # during preflight rather than after loading the checkpoint.
    with tempfile.TemporaryDirectory(
        prefix="jira-sft-preflight-"
    ) as temporary_output:
        training_args = (
            build_jira_training_arguments(
                output_dir=Path(
                    temporary_output
                ),
                max_steps=160,
                learning_rate=2e-4,
                gradient_accumulation_steps=8,
                seed=42,
            )
        )

    import torch

    print(
        "JIRA SFT PREFLIGHT"
    )
    print(
        "=================="
    )
    print(
        "base model key:",
        manifest.base_model_key,
    )
    print(
        "base model path:",
        model_path,
    )
    print(
        "prompt profile:",
        profile.worker_prompt_profile,
    )
    print(
        "records:",
        manifest.record_count,
    )
    print(
        "max prompt tokens:",
        max(prompt_lengths),
    )
    print(
        "max target tokens:",
        max(target_lengths),
    )
    print(
        "configured train max length:",
        args.max_length,
    )
    print(
        "cuda available:",
        torch.cuda.is_available(),
    )

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(
            device
        )
        print(
            "cuda device:",
            torch.cuda.get_device_name(
                device
            ),
        )
        print(
            "cuda free MiB:",
            round(
                free_bytes
                / 1024
                / 1024,
                1,
            ),
        )
        print(
            "cuda total MiB:",
            round(
                total_bytes
                / 1024
                / 1024,
                1,
            ),
        )
        print(
            "bf16 supported:",
            torch.cuda.is_bf16_supported(),
        )

    print(
        "locked package versions:",
        versions,
    )
    print(
        "trainer warmup steps:",
        training_args.warmup_steps,
    )
    print(
        "trainer optimizer:",
        training_args.optim,
    )
    print(
        "TrainingArguments constructor: PASS",
    )
    print()
    print(
        "No model weights were loaded."
    )
    print(
        "No optimizer was created."
    )
    print(
        "No Jira provider call occurred."
    )
    print(
        "PREFLIGHT: PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
