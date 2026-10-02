from __future__ import annotations

import argparse
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from learning.training.specialist_sft import (
    preflight_specialist_sft,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify generic specialist QLoRA prerequisites."
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument(
        "--external-sft-jsonl",
        type=Path,
        action="append",
        default=[],
    )

    args = parser.parse_args()

    report = preflight_specialist_sft(
        project_root=PROJECT_ROOT,
        corpus_directory=args.corpus,
        max_length=args.max_length,
        external_sft_jsonl=args.external_sft_jsonl,
    )

    print("SPECIALIST SFT PREFLIGHT")
    print("========================")

    for key in [
        "specialist",
        "base_model_key",
        "candidate_model_key",
        "base_model_path",
        "model_class",
        "prompt_profile",
        "records",
        "train",
        "validation",
        "external_train",
        "max_prompt_tokens",
        "max_target_tokens",
        "configured_max_length",
        "warmup_steps",
        "optimizer",
    ]:
        print(f"{key}: {report[key]}")

    print(
        "external_sources:",
        report["external_sources"],
    )
    print(
        "training_versions:",
        report["training_versions"],
    )
    print("No model weights were loaded.")
    print("No optimizer was created.")
    print("PREFLIGHT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
