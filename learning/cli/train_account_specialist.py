from __future__ import annotations

import argparse
import json
from pathlib import Path

from learning.training.specialist_sft import (
    build_specialist_sft_corpus,
    preflight_specialist_sft,
    register_candidate_profile,
    train_specialist_sft,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "learning/config/specialists/account.json"
DEFAULT_CORPUS = PROJECT_ROOT / ".runtime/learning/specialist-sft/account-v1"
DEFAULT_OUTPUT = Path(
    "/mnt/c/project/agenticaiPersonal/models/Qwen2.5-0.5B-Account-SFT"
)


def _latest_manifest(output_root: Path) -> Path:
    pointer = output_root.expanduser().resolve() / "latest-training.json"
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    value = payload.get("training_manifest")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("latest-training.json has no training_manifest path")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"training manifest does not exist: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bounded adapter-native Account specialist SFT lifecycle."
    )
    parser.add_argument("action", choices=("build", "preflight", "train", "register"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--training-manifest", type=Path, default=None)
    parser.add_argument("--env-path", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.action == "build":
        result = build_specialist_sft_corpus(
            project_root=PROJECT_ROOT,
            config_path=args.config,
            output_directory=args.corpus_dir,
            force=args.force,
        )
    elif args.action == "preflight":
        result = preflight_specialist_sft(
            project_root=PROJECT_ROOT,
            corpus_directory=args.corpus_dir,
            max_length=args.max_length,
        )
    elif args.action == "train":
        result = train_specialist_sft(
            project_root=PROJECT_ROOT,
            corpus_directory=args.corpus_dir,
            output_root=args.output_root,
            allow_training=True,
            max_steps=args.max_steps,
            max_length=args.max_length,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
    else:
        manifest = args.training_manifest or _latest_manifest(args.output_root)
        result = register_candidate_profile(
            env_path=args.env_path,
            training_manifest_path=manifest,
        )

    if hasattr(result, "model_dump"):
        result = result.model_dump(mode="json", by_alias=True)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
