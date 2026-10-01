from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from subagents.core.definitions.loader import load_agent_definition  # noqa: E402
from subagents.core.orchestration.intent_contract import (  # noqa: E402
    trusted_tool_effect,
)
from subagents.core.tooling.capabilities import (  # noqa: E402
    build_agent_capability_catalog,
)
from tools.registry import get_tool  # noqa: E402


AGENTS = {
    "jira":
        PROJECT_ROOT
        / "subagents"
        / "agents"
        / "jira-specialist.md",

    "developer":
        PROJECT_ROOT
        / "subagents"
        / "agents"
        / "developer-specialist.md",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Jira/developer capability coverage and trusted "
            "guard metadata. No capability is executed."
        )
    )

    parser.add_argument(
        "--strict-mutation-approval",
        action="store_true",
        help=(
            "Fail if a current non-read capability has static "
            "requires_approval=false."
        ),
    )

    args = parser.parse_args()

    report = {
        "schema": "capability-guard-audit.v1",
        "agents": {},
    }

    failures = []

    for _label, path in AGENTS.items():
        agent = load_agent_definition(path)

        capabilities = build_agent_capability_catalog(
            agent,
            include_arguments=True,
        )

        capability_names = {
            item["name"]
            for item in capabilities
        }

        missing = [
            tool_name
            for tool_name in agent.tools
            if tool_name not in capability_names
        ]

        if missing:
            failures.append(
                f"{agent.name}: missing catalog tools {missing}"
            )

        tools = []

        for tool_name in agent.tools:
            tool = get_tool(tool_name)

            if tool is None:
                failures.append(
                    f"{agent.name}: unknown tool {tool_name}"
                )
                continue

            risk = tool.get("risk")
            approval = tool.get("requires_approval")
            effect = trusted_tool_effect(tool)

            if not isinstance(approval, bool):
                failures.append(
                    f"{tool_name}: requires_approval missing"
                )

            if (
                isinstance(risk, str)
                and risk.strip().lower() == "high"
                and approval is not True
            ):
                failures.append(
                    f"{tool_name}: HIGH risk without approval"
                )

            if (
                args.strict_mutation_approval
                and effect == "mutation"
                and approval is not True
            ):
                failures.append(
                    f"{tool_name}: mutation without static approval"
                )

            tools.append(
                {
                    "name": tool_name,
                    "effect": effect,
                    "risk": risk,
                    "requires_approval": approval,
                    "policy_owns_preconditions":
                        bool(
                            tool.get(
                                "policy_owns_preconditions",
                                False,
                            )
                        ),
                    "has_policy_resolver":
                        callable(
                            tool.get("policy_resolver")
                        ),
                }
            )

        report["agents"][agent.name] = {
            "tool_count": len(agent.tools),
            "tools": tools,
        }

    report["failures"] = failures

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    print()
    print("No capability was executed.")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
