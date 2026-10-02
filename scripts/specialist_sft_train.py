from __future__ import annotations

import argparse
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from learning.training.specialist_sft import (
    train_specialist_sft,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train one governed specialist with QLoRA SFT."
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--external-sft-jsonl",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional grounded SFT JSONL containing prompt_messages "
            "+ chosen records. Repeat for multiple sources."
        ),
    )
    parser.add_argument("--allow-training", action="store_true")
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)

    args = parser.parse_args()

    manifest = train_specialist_sft(
        project_root=PROJECT_ROOT,
        corpus_directory=args.corpus,
        output_root=args.output_root,
        allow_training=args.allow_training,
        max_steps=args.max_steps,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        external_sft_jsonl=args.external_sft_jsonl,
    )

    print("SPECIALIST SFT TRAINING: PASS")
    print("specialist:", manifest.specialist)
    print("candidate:", manifest.candidate_model_key)
    print(
        "prompt profile:",
        manifest.worker_prompt_profile,
    )
    print(
        "external records:",
        sum(
            source.record_count
            for source in manifest.external_sft_sources
        ),
    )
    print("adapter:", manifest.adapter_directory)
    print("merged:", manifest.merged_model_directory)
    print("trainable ratio:", manifest.trainable_ratio)
    print("Production specialist binding was NOT modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
