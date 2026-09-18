from __future__ import annotations

import argparse
import json
from pathlib import Path

from learning.continual.checkpoints import AdapterCheckpointStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Register, promote, inspect, or roll back Phase-5 adapter checkpoints.")
    sub = parser.add_subparsers(dest="command", required=True)

    register = sub.add_parser("register")
    register.add_argument("--adapter-directory", type=Path, required=True)
    register.add_argument("--base-model-sha256", required=True)
    register.add_argument("--source-cycle-id", required=True)
    register.add_argument("--source-split-id", required=True)
    register.add_argument("--target-agent", required=True)
    register.add_argument("--target-model-key", required=True)
    register.add_argument("--label", required=True)

    promote = sub.add_parser("promote")
    promote.add_argument("--checkpoint-id", required=True)
    promote.add_argument("--suite", required=True)
    promote.add_argument("--decision-id", required=True)

    sub.add_parser("status")
    sub.add_parser("rollback")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    store = AdapterCheckpointStore()

    if args.command == "register":
        result = store.register(
            adapter_directory=args.adapter_directory,
            base_model_sha256=args.base_model_sha256,
            source_cycle_id=args.source_cycle_id,
            source_split_id=args.source_split_id,
            target_agent=args.target_agent,
            target_model_key=args.target_model_key,
            label=args.label,
        )
    elif args.command == "promote":
        result = store.promote(checkpoint_id=args.checkpoint_id, suite=args.suite, decision_id=args.decision_id)
    elif args.command == "rollback":
        result = store.rollback()
    else:
        result = store.active()
        if result is None:
            print("No active Phase-5 checkpoint.")
            return 0

    print(json.dumps(result.model_dump(mode="json", by_alias=True), ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
