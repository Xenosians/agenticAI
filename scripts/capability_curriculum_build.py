from __future__ import annotations

import argparse
import json
import shutil
import sys

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from learning.training.capability_curriculum import (  # noqa: E402
    materialize_agent_capability_curriculum,
)


DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / ".runtime"
    / "learning"
    / "capability-language"
    / "v2"
)


AGENTS = {
    "jira": Path("subagents/agents/jira-specialist.md"),
    "developer": Path("subagents/agents/developer-specialist.md"),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build catalog-derived Jira + developer capability-language "
            "curricula without executing any capability."
        )
    )

    parser.add_argument(
        "--target",
        choices=["all", *sorted(AGENTS)],
        default="all",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    targets = (
        sorted(AGENTS)
        if args.target == "all"
        else [args.target]
    )

    root = args.output_root.expanduser().resolve()

    if (
        args.force
        and args.target == "all"
        and root.exists()
    ):
        shutil.rmtree(root)

    root.mkdir(parents=True, exist_ok=True)

    reports = []

    for target in targets:
        output = root / target

        manifest = materialize_agent_capability_curriculum(
            project_root=PROJECT_ROOT,
            agent_definition=AGENTS[target],
            output_directory=output,
            force=args.force,
        )

        reports.append(
            manifest.model_dump(
                mode="json",
                by_alias=True,
            )
        )

    print(
        json.dumps(
            reports,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    print()
    print("No tool was executed.")
    print("No Jira/Git/shell mutation occurred.")
    print("Training remains unauthorized.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
