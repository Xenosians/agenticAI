from __future__ import annotations

import argparse
import json
from pathlib import Path

from learning.continual.failure_diagnostics import (
    DEFAULT_FAILURE_MINING_ROOT,
    run_failure_mining,
)
from learning.paths import TRAJECTORIES_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Mine deterministic runtime failures into review-only "
            "candidate learning artifacts. This command never trains, "
            "promotes, or activates a model."
        )
    )
    parser.add_argument("--trajectories", type=Path, default=TRAJECTORIES_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_FAILURE_MINING_ROOT)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        manifest = run_failure_mining(
            trajectories_path=args.trajectories,
            output_root=args.output_root,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    if args.json:
        print(manifest.model_dump_json(by_alias=True, indent=2))
        return 0

    print("Governed Failure Mining")
    print("=======================")
    print(f"Run:         {manifest.run_id}")
    print(f"Trajectories:{manifest.trajectory_count:>8}")
    print(f"Diagnoses:   {manifest.diagnosis_count:>8}")
    print(f"Candidates:  {manifest.candidate_count:>8}")
    print(f"Failures:    {json.dumps(manifest.failure_counts, sort_keys=True)}")
    print(f"Output:      {manifest.output_directory}")
    print("Training:    DISABLED")
    print("Promotion:   DISABLED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
