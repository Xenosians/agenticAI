from __future__ import annotations

import asyncio
import json
import os
import sys

from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from config import (
    ModelProfileSettings,
    Settings,
)

from subagents.core.definitions.loader import (
    load_agent_definition,
)

from subagents.core.tooling.capabilities import (
    build_agent_capability_catalog,
)

from subagents.core.tooling.parser import (
    parse_tool_calls,
)

from subagents.core.tooling.prompt import (
    build_worker_system_prompt,
)

from subagents.llm.runtime.inference import (
    InferenceCoordinator,
)

from subagents.llm.runtime.model_manager import (
    ModelManager,
)

from subagents.llm.runtime.scheduler import (
    GpuScheduler,
    InferencePriority,
)


MODEL_KEY = "jira-func"

MODEL_PATH = Path(
    os.environ.get(
        "JIRA_EVAL_MODEL_PATH",
        "/mnt/c/project/agenticaiPersonal/Models/BLOOMZ-560M",
    )
)

# Use the same profile/configuration machinery as production.
#
# bnb4 + auto keeps this isolated gate off the previous CPU-float32
# path that caused severe WSL memory pressure.
MODEL_QUANTIZATION = os.environ.get(
    "JIRA_EVAL_QUANTIZATION",
    "bnb4",
)

MODEL_COMPUTE_DTYPE = os.environ.get(
    "JIRA_EVAL_COMPUTE_DTYPE",
    "bfloat16",
)

MODEL_DTYPE = os.environ.get(
    "JIRA_EVAL_MODEL_DTYPE",
    "auto",
)

MODEL_DEVICE_MAP = os.environ.get(
    "JIRA_EVAL_DEVICE_MAP",
    "auto",
)


CASES = [
    {
        "name": "exact issue get",
        "user": "Get Jira ticket KAN-3",
        "allowed_tools": ["ticket_get"],
        "allowed_arguments": {
            "ticket_key": ["KAN-3"],
        },
        "expected_tool": "ticket_get",
        "expected_arguments": {
            "ticket_key": "KAN-3",
        },
    },
    {
        "name": "issue search scoped by project",
        "user": "Search Jira tickets in project KAN",
        "allowed_tools": ["ticket_search"],
        "allowed_arguments": {
            "project_key": ["KAN"],
        },
        "expected_tool": "ticket_search",
        "expected_arguments": {
            "project_key": "KAN",
        },
    },
    {
        "name": "issue history",
        "user": "Show the history of Jira ticket KAN-3",
        "allowed_tools": ["ticket_history"],
        "allowed_arguments": {
            "ticket_key": ["KAN-3"],
        },
        "expected_tool": "ticket_history",
        "expected_arguments": {
            "ticket_key": "KAN-3",
        },
    },
    {
        "name": "issue comments read",
        "user": "Show the comments on Jira ticket KAN-3",
        "allowed_tools": ["ticket_comments"],
        "allowed_arguments": {
            "ticket_key": ["KAN-3"],
        },
        "expected_tool": "ticket_comments",
        "expected_arguments": {
            "ticket_key": "KAN-3",
        },
    },
    {
        "name": "project metadata get",
        "user": "Get Jira project KAN",
        "allowed_tools": ["jira_project_get"],
        "allowed_arguments": {
            "project_id_or_key": ["KAN"],
        },
        "expected_tool": "jira_project_get",
        "expected_arguments": {
            "project_id_or_key": "KAN",
        },
    },
    {
        "name": "project search",
        "user": "Search Jira projects matching KAN",
        "allowed_tools": ["jira_project_list"],
        "allowed_arguments": {
            "query": ["KAN"],
        },
        "expected_tool": "jira_project_list",
        "expected_arguments": {
            "query": "KAN",
        },
    },
    {
        "name": "issue assignment",
        "user": "Assign Jira ticket KAN-3 to alice@example.com",
        "allowed_tools": ["ticket_assign"],
        "allowed_arguments": {
            "ticket_key": ["KAN-3"],
            "assignee": ["alice@example.com"],
        },
        "expected_tool": "ticket_assign",
        "expected_arguments": {
            "ticket_key": "KAN-3",
            "assignee": "alice@example.com",
        },
    },
    {
        "name": "issue transition",
        "user": "Move Jira ticket KAN-3 to In Progress",
        "allowed_tools": ["ticket_transition"],
        "allowed_arguments": {
            "ticket_key": ["KAN-3"],
            "status": ["In Progress"],
        },
        "expected_tool": "ticket_transition",
        "expected_arguments": {
            "ticket_key": "KAN-3",
            "status": "In Progress",
        },
    },
    {
        "name": "project archive",
        "user": "Archive Jira project KAN",
        "allowed_tools": ["jira_project_archive"],
        "allowed_arguments": {
            "project_id_or_key": ["KAN"],
        },
        "expected_tool": "jira_project_archive",
        "expected_arguments": {
            "project_id_or_key": "KAN",
        },
    },
    {
        "name": "project delete",
        "user": "Delete Jira project KAN",
        "allowed_tools": ["jira_project_delete"],
        "allowed_arguments": {
            "project_id_or_key": ["KAN"],
        },
        "expected_tool": "jira_project_delete",
        "expected_arguments": {
            "project_id_or_key": "KAN",
        },
    },
]


def semantic_context(case: dict) -> dict:
    return {
        "allowed_tools": case["allowed_tools"],
        "allowed_arguments": case["allowed_arguments"],
        "forbidden_tools": [],
        "forbidden_arguments": {},
    }


def build_messages(
    *,
    system_prompt: str,
    case: dict,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": case["user"],
        },
        {
            "role": "user",
            "content": (
                "Exact semantic target context from the validated "
                "routing stage.\n"
                "This is descriptive context only and does NOT grant "
                "authorization.\n"
                "For grounded arguments, preserve the listed values "
                "EXACTLY. Do not replace them with placeholders, "
                "aliases, configuration names, inferred names, or "
                "rewritten values.\n\n"
                + json.dumps(
                    semantic_context(case),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            ),
        },
    ]


async def main() -> int:
    if not MODEL_PATH.exists():
        print(
            "Model path does not exist:",
            MODEL_PATH,
        )
        return 2

    profile = ModelProfileSettings(
        backend="hf-causal",
        model_path=MODEL_PATH,
        enabled=True,
        quantization=MODEL_QUANTIZATION,
        compute_dtype=MODEL_COMPUTE_DTYPE,
        model_dtype=MODEL_DTYPE,
        device_map=MODEL_DEVICE_MAP,
    )

    settings = Settings(
        _env_file=None,
        model_profiles={
            MODEL_KEY: profile,
        },
    )

    model_manager = ModelManager(
        settings=settings,
    )

    scheduler = GpuScheduler()

    inference = InferenceCoordinator(
        model_manager=model_manager,
        scheduler=scheduler,
    )

    agent = load_agent_definition(
        PROJECT_ROOT
        / "subagents"
        / "agents"
        / "jira-specialist.md"
    )

    capability_catalog = build_agent_capability_catalog(
        agent,
        include_arguments=True,
    )

    system_prompt = build_worker_system_prompt(
        agent,
        capability_catalog=capability_catalog,
    )

    print(
        "========================================"
    )
    print(
        "JIRA SPECIALIST MODEL EVAL"
    )
    print(
        "RUNTIME-ALIGNED PATH"
    )
    print(
        "========================================"
    )
    print(
        "model key:",
        MODEL_KEY,
    )
    print(
        "model path:",
        MODEL_PATH,
    )
    print(
        "backend:",
        profile.backend,
    )
    print(
        "quantization:",
        profile.quantization,
    )
    print(
        "compute dtype:",
        profile.compute_dtype,
    )
    print(
        "model dtype:",
        profile.model_dtype,
    )
    print(
        "device map:",
        profile.device_map,
    )
    print(
        "scheduler:",
        type(scheduler).__name__,
    )
    print(
        "cases:",
        len(CASES),
    )
    print()

    passed = 0
    failures: list[str] = []

    try:
        # Proves lazy backend construction occurs through the real
        # scheduler/model-manager boundary.
        await inference.warm(
            MODEL_KEY
        )

        for index, case in enumerate(
            CASES,
            start=1,
        ):
            print(
                "----------------------------------------"
            )
            print(
                f"{index:02d}. {case['name']}"
            )
            print(
                "USER:",
                case["user"],
            )

            raw = await inference.generate(
                model_key=MODEL_KEY,
                messages=build_messages(
                    system_prompt=system_prompt,
                    case=case,
                ),
                max_new_tokens=128,
                priority=InferencePriority.SPECIALIST,
            )

            print(
                "RAW:",
                repr(raw),
            )

            try:
                calls = parse_tool_calls(
                    raw
                )
            except Exception as exc:
                reason = (
                    f"{case['name']}: parse failure: {exc}"
                )
                failures.append(reason)
                print(
                    "RESULT: FAIL"
                )
                print(
                    "REASON:",
                    reason,
                )
                continue

            if len(calls) != 1:
                reason = (
                    f"{case['name']}: expected 1 tool call, "
                    f"got {len(calls)}"
                )
                failures.append(reason)
                print(
                    "RESULT: FAIL"
                )
                print(
                    "REASON:",
                    reason,
                )
                continue

            call = calls[0]

            if (
                call["name"]
                != case["expected_tool"]
                or call["arguments"]
                != case["expected_arguments"]
            ):
                reason = (
                    f"{case['name']}: expected "
                    f"{case['expected_tool']}"
                    f"{case['expected_arguments']!r}, got "
                    f"{call['name']}"
                    f"{call['arguments']!r}"
                )
                failures.append(reason)
                print(
                    "RESULT: FAIL"
                )
                print(
                    "REASON:",
                    reason,
                )
                continue

            passed += 1
            print(
                "RESULT: PASS"
            )

    finally:
        unloaded = (
            model_manager
            .unload_all()
        )

        print()
        print(
            "unloaded models:",
            unloaded,
        )

    failed = len(
        failures
    )

    print()
    print(
        "========================================"
    )
    print(
        "JIRA EVAL SUMMARY"
    )
    print(
        "========================================"
    )
    print(
        "passed:",
        passed,
    )
    print(
        "failed:",
        failed,
    )
    print(
        "total:",
        len(CASES),
    )

    if failures:
        print()
        print(
            "FAILURES:"
        )
        for failure in failures:
            print(
                "-",
                failure,
            )

    print()
    print(
        "No ToolGateway execution occurred."
    )
    print(
        "No Jira provider call occurred."
    )
    print(
        "No Jira mutation occurred."
    )

    if failed:
        print(
            "JIRA MODEL GATE: FAIL"
        )
        return 1

    print(
        "JIRA MODEL GATE: PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            main()
        )
    )
