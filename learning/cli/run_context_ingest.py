from __future__ import annotations

import argparse
import json
from pathlib import Path

from learning.context.recorder import LearningContextRecorder
from learning.integrations.runtime_hooks import DEFAULT_CONTEXT_RECORDS_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Append one structured, sanitized learning-context record. "
            "Context is non-authoritative and never training-eligible by itself."
        )
    )
    parser.add_argument("--kind", choices=["git", "jira", "shell", "human", "task", "tool"])
    parser.add_argument("--source", choices=["git", "jira", "shell", "human"], help="Legacy alias for --kind")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--payload-json", default=None, help="Structured JSON object for the selected context kind")
    parser.add_argument("--content", default=None, help="Legacy text; stored inside a structured payload")
    parser.add_argument("--metadata-json", default="{}", help="Legacy metadata object used with --content")
    parser.add_argument("--trajectory-id")
    parser.add_argument("--task-id")
    parser.add_argument("--job-id")
    parser.add_argument("--source-tool")
    parser.add_argument("--path", default=str(DEFAULT_CONTEXT_RECORDS_PATH))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    kind = args.kind or args.source
    if not kind:
        raise SystemExit("provide --kind (or legacy --source)")

    if args.payload_json is not None:
        try:
            payload = json.loads(args.payload_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid --payload-json: {exc}") from exc
        if not isinstance(payload, dict):
            raise SystemExit("--payload-json must decode to an object")
    else:
        if args.content is None:
            raise SystemExit("provide --payload-json or legacy --content")
        try:
            metadata = json.loads(args.metadata_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid --metadata-json: {exc}") from exc
        if not isinstance(metadata, dict):
            raise SystemExit("--metadata-json must decode to an object")
        payload = {"content": args.content, "metadata": metadata, "legacy": True}

    trust = "human" if kind == "human" else "runtime"
    event = LearningContextRecorder(path=Path(args.path)).record(
        kind=kind,
        subject=args.subject,
        payload=payload,
        trust=trust,
        trajectory_id=args.trajectory_id,
        task_id=args.task_id,
        job_id=args.job_id,
        source_tool=args.source_tool,
    )
    if event is None:
        print("Context recording disabled.")
        return 0
    print(json.dumps(event.model_dump(mode="json", by_alias=True), ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
