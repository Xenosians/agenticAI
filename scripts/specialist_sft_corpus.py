from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from learning.training.specialist_sft import build_specialist_sft_corpus, load_training_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a generic governed specialist SFT corpus.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_training_config(args.config)
    output = args.output or (
        PROJECT_ROOT / ".runtime" / "learning" / "specialist-sft" / config.specialist / "v1"
    )

    manifest = build_specialist_sft_corpus(
        project_root=PROJECT_ROOT,
        config_path=args.config,
        output_directory=output,
        force=args.force,
    )
    print("SPECIALIST SFT CORPUS: PASS")
    print("specialist:", manifest.specialist)
    print("base model:", manifest.base_model_key)
    print("candidate:", manifest.candidate_model_key)
    print("records:", manifest.record_count)
    print("train:", manifest.train_count)
    print("validation:", manifest.validation_count)
    print("output:", output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
