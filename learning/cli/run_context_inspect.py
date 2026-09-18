from __future__ import annotations

import argparse
import json
from pathlib import Path

from learning.context import assemble_context_bundle, load_context_records
from learning.integrations.runtime_hooks import DEFAULT_CONTEXT_RECORDS_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect one bounded, lineage-aware contextual-learning bundle.")
    parser.add_argument("--path", type=Path, default=DEFAULT_CONTEXT_RECORDS_PATH)
    parser.add_argument("--trajectory-id")
    parser.add_argument("--task-id")
    parser.add_argument("--job-id")
    parser.add_argument("--max-records", type=int, default=96)
    parser.add_argument("--max-payload-chars", type=int, default=40_000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    bundle = assemble_context_bundle(
        load_context_records(args.path),
        trajectory_id=args.trajectory_id,
        task_id=args.task_id,
        job_id=args.job_id,
        max_records=args.max_records,
        max_payload_chars=args.max_payload_chars,
    )
    print(json.dumps(bundle.model_dump(mode="json", by_alias=True), ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
