from __future__ import annotations

import asyncio
import json
import os
import sys
import time

from dataclasses import (
    replace,
)

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
    Settings,
)

from learning.evaluation.specialist_model import (
    SpecialistModelCaseResult,
    finalize_specialist_model_report,
    write_specialist_model_report,
)

from learning.evidence.execution_provenance import (
    build_specialist_execution_provenance,
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


MODEL_KEY = os.environ.get(
    "JIRA_EVAL_MODEL_KEY",
    "jira-func",
).strip()

MAX_NEW_TOKENS = int(
    os.environ.get(
        "JIRA_EVAL_MAX_NEW_TOKENS",
        "128",
    )
)

REPORT_PATH = Path(
    os.environ.get(
        "JIRA_EVAL_REPORT_PATH",
        str(
            PROJECT_ROOT
            / ".runtime"
            / "evaluation"
            / f"jira_specialist_{MODEL_KEY}.json"
        ),
    )
)


CASES = [
    {
        "name": "exact issue get",
        "user": "Get Jira ticket KAN-3",
        "allowed_tools": ["ticket_get"],
        "allowed_arguments": {"ticket_key": ["KAN-3"]},
        "expected_tool": "ticket_get",
        "expected_arguments": {"ticket_key": "KAN-3"},
    },
    {
        "name": "issue search scoped by project",
        "user": "Search Jira tickets in project KAN",
        "allowed_tools": ["ticket_search"],
        "allowed_arguments": {"project_key": ["KAN"]},
        "expected_tool": "ticket_search",
        "expected_arguments": {"project_key": "KAN"},
    },
    {
        "name": "issue search text and project",
        "user": "Find Jira issues in OPS containing database timeout",
        "allowed_tools": ["ticket_search"],
        "allowed_arguments": {
            "project_key": ["OPS"],
            "text": ["database timeout"],
        },
        "expected_tool": "ticket_search",
        "expected_arguments": {
            "project_key": "OPS",
            "text": "database timeout",
        },
    },
    {
        "name": "issue history",
        "user": "Show the history of Jira ticket KAN-3",
        "allowed_tools": ["ticket_history"],
        "allowed_arguments": {"ticket_key": ["KAN-3"]},
        "expected_tool": "ticket_history",
        "expected_arguments": {"ticket_key": "KAN-3"},
    },
    {
        "name": "issue comments read",
        "user": "Show the comments on Jira ticket KAN-3",
        "allowed_tools": ["ticket_comments"],
        "allowed_arguments": {"ticket_key": ["KAN-3"]},
        "expected_tool": "ticket_comments",
        "expected_arguments": {"ticket_key": "KAN-3"},
    },
    {
        "name": "issue comment mutation",
        "user": "Add comment Customer confirmed the VPN is working. to Jira ticket KAN-3",
        "allowed_tools": ["ticket_add_comment"],
        "allowed_arguments": {
            "ticket_key": ["KAN-3"],
            "comment": ["Customer confirmed the VPN is working."],
        },
        "expected_tool": "ticket_add_comment",
        "expected_arguments": {
            "ticket_key": "KAN-3",
            "comment": "Customer confirmed the VPN is working.",
        },
    },
    {
        "name": "issue create",
        "user": "Create Jira ticket in KAN with summary VPN login fails after reboot",
        "allowed_tools": ["ticket_create"],
        "allowed_arguments": {
            "project_key": ["KAN"],
            "summary": ["VPN login fails after reboot"],
        },
        "expected_tool": "ticket_create",
        "expected_arguments": {
            "project_key": "KAN",
            "summary": "VPN login fails after reboot",
        },
    },
    {
        "name": "issue create with type",
        "user": "Create a Task in OPS with summary Rotate staging credentials",
        "allowed_tools": ["ticket_create"],
        "allowed_arguments": {
            "project_key": ["OPS"],
            "summary": ["Rotate staging credentials"],
            "ticket_type": ["Task"],
        },
        "expected_tool": "ticket_create",
        "expected_arguments": {
            "project_key": "OPS",
            "summary": "Rotate staging credentials",
            "ticket_type": "Task",
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
        "name": "project metadata get",
        "user": "Get Jira project KAN",
        "allowed_tools": ["jira_project_get"],
        "allowed_arguments": {"project_id_or_key": ["KAN"]},
        "expected_tool": "jira_project_get",
        "expected_arguments": {"project_id_or_key": "KAN"},
    },
    {
        "name": "project search",
        "user": "Search Jira projects matching KAN",
        "allowed_tools": ["jira_project_list"],
        "allowed_arguments": {"query": ["KAN"]},
        "expected_tool": "jira_project_list",
        "expected_arguments": {"query": "KAN"},
    },
    {
        "name": "project create",
        "user": "Create Jira project LAB named Automation Lab using software-kanban",
        "allowed_tools": ["jira_project_create"],
        "allowed_arguments": {
            "project_key": ["LAB"],
            "project_name": ["Automation Lab"],
            "template": ["software-kanban"],
        },
        "expected_tool": "jira_project_create",
        "expected_arguments": {
            "project_key": "LAB",
            "project_name": "Automation Lab",
            "template": "software-kanban",
        },
    },
    {
        "name": "project rename",
        "user": "Rename Jira project KAN to Platform Kanban",
        "allowed_tools": ["jira_project_update"],
        "allowed_arguments": {
            "project_id_or_key": ["KAN"],
            "new_name": ["Platform Kanban"],
        },
        "expected_tool": "jira_project_update",
        "expected_arguments": {
            "project_id_or_key": "KAN",
            "new_name": "Platform Kanban",
        },
    },
    {
        "name": "project archive",
        "user": "Archive Jira project KAN",
        "allowed_tools": ["jira_project_archive"],
        "allowed_arguments": {"project_id_or_key": ["KAN"]},
        "expected_tool": "jira_project_archive",
        "expected_arguments": {"project_id_or_key": "KAN"},
    },
    {
        "name": "project delete",
        "user": "Delete Jira project KAN",
        "allowed_tools": ["jira_project_delete"],
        "allowed_arguments": {"project_id_or_key": ["KAN"]},
        "expected_tool": "jira_project_delete",
        "expected_arguments": {"project_id_or_key": "KAN"},
    },
    {
        "name": "scope is project but resource is issue",
        "user": "List issues under OPS, not Jira projects",
        "allowed_tools": ["ticket_search"],
        "allowed_arguments": {"project_key": ["OPS"]},
        "expected_tool": "ticket_search",
        "expected_arguments": {"project_key": "OPS"},
    },
    {
        "name": "resource is project collection",
        "user": "List Jira projects matching OPS",
        "allowed_tools": ["jira_project_list"],
        "allowed_arguments": {"query": ["OPS"]},
        "expected_tool": "jira_project_list",
        "expected_arguments": {"query": "OPS"},
    },
    {
        "name": "comment read paraphrase",
        "user": "What discussion is on HELP-204?",
        "allowed_tools": ["ticket_comments"],
        "allowed_arguments": {"ticket_key": ["HELP-204"]},
        "expected_tool": "ticket_comments",
        "expected_arguments": {"ticket_key": "HELP-204"},
    },
    {
        "name": "assignment paraphrase",
        "user": "Set SEC-12 assignee to bob@example.com",
        "allowed_tools": ["ticket_assign"],
        "allowed_arguments": {
            "ticket_key": ["SEC-12"],
            "assignee": ["bob@example.com"],
        },
        "expected_tool": "ticket_assign",
        "expected_arguments": {
            "ticket_key": "SEC-12",
            "assignee": "bob@example.com",
        },
    },
]


BENIGN_OPTIONAL_ARGUMENTS = {
    "ticket_search": {
        "limit": lambda value: (
            isinstance(value, int)
            and not isinstance(value, bool)
            and 1 <= value <= 25
        ),
    },
    "ticket_history": {
        "limit": lambda value: (
            isinstance(value, int)
            and not isinstance(value, bool)
            and 1 <= value <= 50
        ),
    },
    "ticket_comments": {
        "limit": lambda value: (
            isinstance(value, int)
            and not isinstance(value, bool)
            and 1 <= value <= 50
        ),
    },
    "jira_project_list": {
        "limit": lambda value: (
            value is None
            or (
                isinstance(value, int)
                and not isinstance(value, bool)
                and 1 <= value <= 50
            )
        ),
    },
}


def arguments_semantically_match(
    *,
    tool_name: str,
    expected: dict,
    observed: dict,
) -> bool:
    if not isinstance(
        observed,
        dict,
    ):
        return False

    for (
        argument_name,
        expected_value,
    ) in expected.items():
        if (
            argument_name
            not in observed
            or observed[argument_name]
            != expected_value
        ):
            return False

    extras = (
        set(observed)
        - set(expected)
    )

    rules = (
        BENIGN_OPTIONAL_ARGUMENTS
        .get(
            tool_name,
            {},
        )
    )

    for argument_name in extras:
        validator = (
            rules.get(
                argument_name
            )
        )

        if (
            validator is None
            or not validator(
                observed[argument_name]
            )
        ):
            return False

    return True



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
    if not MODEL_KEY:
        print(
            "JIRA_EVAL_MODEL_KEY must not be empty."
        )
        return 2

    settings = Settings()

    try:
        profile = (
            settings
            .require_model_profile(
                MODEL_KEY
            )
        )

    except Exception as exc:
        print(
            "Jira candidate model profile is unavailable:",
            repr(exc),
        )
        return 2

    model_manager = ModelManager(
        settings=settings,
    )

    scheduler = GpuScheduler()

    inference = InferenceCoordinator(
        model_manager=model_manager,
        scheduler=scheduler,
    )

    base_agent = load_agent_definition(
        PROJECT_ROOT
        / "subagents"
        / "agents"
        / "jira-specialist.md"
    )

    # Candidate evaluation must fingerprint the candidate model key,
    # not the currently promoted production model key. Tool ownership,
    # system instructions and execution authority remain identical.
    agent = (
        replace(
            base_agent,
            model=MODEL_KEY,
        )
    )

    capability_catalog = build_agent_capability_catalog(
        agent,
        include_arguments=True,
    )

    system_prompt = build_worker_system_prompt(
        agent,
        capability_catalog=capability_catalog,
        prompt_profile=profile.worker_prompt_profile,
    )

    print(
        "========================================"
    )
    print(
        "JIRA SPECIALIST MODEL EVAL"
    )
    print(
        "MODEL-FLEET RUNTIME PATH"
    )
    print(
        "========================================"
    )
    print(
        "production Jira model:",
        base_agent.model,
    )
    print(
        "candidate model key:",
        MODEL_KEY,
    )
    print(
        "model path:",
        profile.model_path,
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
        "worker prompt profile:",
        profile.worker_prompt_profile,
    )
    print(
        "residency max:",
        settings.model_max_loaded_models,
    )
    print(
        "residency pinned:",
        model_manager
        .residency_snapshot()
        .pinned_models,
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

    case_results: list[
        SpecialistModelCaseResult
    ] = []

    try:
        # Loading is admitted through the same scheduler boundary as
        # production warm-up and generation. ModelManager applies the
        # same residency policy before constructing the backend.
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

            messages = build_messages(
                system_prompt=system_prompt,
                case=case,
            )

            try:
                provenance = (
                    build_specialist_execution_provenance(
                        agent=agent,
                        model_profile=profile,
                        capability_catalog=capability_catalog,
                        messages=messages,
                        user_request=case["user"],
                        task_instructions=None,
                        max_new_tokens=MAX_NEW_TOKENS,
                    )
                    .model_dump(
                        mode="json",
                        by_alias=True,
                    )
                )

            except Exception as exc:
                provenance = None
                print(
                    "PROVENANCE: unavailable",
                    repr(exc),
                )

            started_at = time.perf_counter()

            raw = await inference.generate(
                model_key=MODEL_KEY,
                messages=messages,
                max_new_tokens=MAX_NEW_TOKENS,
                priority=InferencePriority.SPECIALIST,
            )

            duration_seconds = (
                time.perf_counter()
                - started_at
            )

            print(
                "RAW:",
                repr(raw),
            )

            observed_tool = None
            observed_arguments = None
            parse_error = None
            passed = False

            try:
                calls = parse_tool_calls(
                    raw
                )

                if len(calls) != 1:
                    parse_error = (
                        "expected exactly one tool call; "
                        f"got {len(calls)}"
                    )

                else:
                    call = calls[0]
                    observed_tool = (
                        call["name"]
                    )
                    observed_arguments = (
                        call["arguments"]
                    )

                    passed = (
                        observed_tool
                        == case["expected_tool"]
                        and arguments_semantically_match(
                            tool_name=(
                                observed_tool
                            ),
                            expected=(
                                case["expected_arguments"]
                            ),
                            observed=(
                                observed_arguments
                            ),
                        )
                    )

            except Exception as exc:
                parse_error = str(
                    exc
                )

            case_results.append(
                SpecialistModelCaseResult(
                    name=case["name"],
                    user_request=case["user"],
                    expected_tool=case["expected_tool"],
                    expected_arguments=case["expected_arguments"],
                    observed_tool=observed_tool,
                    observed_arguments=observed_arguments,
                    raw_output=raw,
                    parse_error=parse_error,
                    passed=passed,
                    duration_seconds=duration_seconds,
                    execution_provenance=provenance,
                )
            )

            print(
                "RESULT:",
                "PASS" if passed else "FAIL",
            )

            if parse_error:
                print(
                    "REASON:",
                    parse_error,
                )

            elif not passed:
                print(
                    "EXPECTED:",
                    case["expected_tool"],
                    case["expected_arguments"],
                )
                print(
                    "OBSERVED:",
                    observed_tool,
                    observed_arguments,
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

    report = finalize_specialist_model_report(
        agent_name=agent.name,
        model_key=MODEL_KEY,
        backend=profile.backend,
        quantization=profile.quantization,
        compute_dtype=profile.compute_dtype,
        device_map=profile.device_map,
        cases=case_results,
    )

    written = write_specialist_model_report(
        report,
        REPORT_PATH,
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
        report.passed,
    )
    print(
        "failed:",
        report.failed,
    )
    print(
        "total:",
        report.total,
    )
    print(
        "pass rate:",
        f"{report.pass_rate:.3f}",
    )
    print(
        "mean duration seconds:",
        report.mean_duration_seconds,
    )
    print(
        "report:",
        written,
    )
    print(
        "report sha256:",
        report.report_sha256,
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

    if not report.promotion_gate_passed:
        print(
            "JIRA MODEL GATE: FAIL"
        )
        return 1

    print(
        "JIRA MODEL GATE: PASS"
    )
    print(
        "Candidate passed the isolated protocol gate. "
        "This does not automatically promote the model."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            main()
        )
    )
