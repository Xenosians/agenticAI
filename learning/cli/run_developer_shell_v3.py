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


def _json(value) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", by_alias=True)
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build/preflight/train/register Developer Shell/Workspace v3."
    )
    parser.add_argument("action", choices=("build", "preflight", "train", "register"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("learning/config/specialists/developer-shell-v3.json"),
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path(".runtime/learning/specialist-sft/developer-shell-v3"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/mnt/c/project/agenticaiPersonal/models/"
            "Ministral-3-3B-Developer-Shell-v3-SFT"
        ),
    )
    parser.add_argument("--training-manifest", type=Path, default=None)
    parser.add_argument("--env-path", type=Path, default=Path(".env"))
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    project_root = Path.cwd().resolve()

    if args.action == "build":
        result = build_specialist_sft_corpus(
            project_root=project_root,
            config_path=args.config,
            output_directory=args.corpus_dir,
            force=args.force,
        )
    elif args.action == "preflight":
        result = preflight_specialist_sft(
            project_root=project_root,
            corpus_directory=args.corpus_dir,
            max_length=args.max_length,
        )
    elif args.action == "train":
        result = train_specialist_sft(
            project_root=project_root,
            corpus_directory=args.corpus_dir,
            output_root=args.output_root,
            allow_training=True,
            max_steps=args.max_steps,
            max_length=args.max_length,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
    else:
        manifest_path = args.training_manifest
        if manifest_path is None:
            pointer = args.output_root.expanduser().resolve() / "latest-training.json"
            payload = json.loads(pointer.read_text(encoding="utf-8"))
            manifest_path = Path(payload["training_manifest"])

        result = register_candidate_profile(
            env_path=args.env_path,
            training_manifest_path=manifest_path,
        )

    print(_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
