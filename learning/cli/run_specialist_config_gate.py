from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from learning.evaluation.config_specialist_gate import (
    evaluate_specialist_config_gate,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a forward-only fixed specialist semantic gate from reviewed "
            "specialist configs. This never trains or executes provider tools."
        )
    )
    parser.add_argument("--config", action="append", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gate-name", default="specialist-config-gate")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--exclude-tool-prefix",
        action="append",
        default=[],
        help="Exclude eval cases whose expected tool starts with this prefix.",
    )
    return parser


async def _main() -> int:
    args = build_parser().parse_args()
    report = await evaluate_specialist_config_gate(
        project_root=PROJECT_ROOT,
        config_paths=args.config,
        model_key=args.model_key,
        report_path=args.report,
        gate_name=args.gate_name,
        max_new_tokens=args.max_new_tokens,
        exclude_tool_prefixes=tuple(args.exclude_tool_prefix),
    )
    return 0 if report.promotion_gate_passed else 1


def main() -> int:
    return asyncio.run(_main())


if __name__ == "__main__":
    raise SystemExit(main())
