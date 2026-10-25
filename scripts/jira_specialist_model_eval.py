from __future__ import annotations

import asyncio
import argparse
import hashlib
import json
import os
import sys
import time

from pydantic import TypeAdapter, ValidationError

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

from subagents.llm.runtime.stages import InferenceStage

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

from tools.capability_contract import build_capability_contract
from tools.registry import get_tool


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

CASE_FILE_SCHEMA = "jira-evaluation-cases.v1"

# Built-in development cases: they also select candidates during training work,
# so a pass here is development evidence. Held-out scoring uses --cases.
CASES = [
    {
        "name": "unfiltered project collection",
        "exact_arguments": True,
        "user": "List Jira projects without filtering the collection",
        "allowed_tools": ["jira_project_list"],
        "allowed_arguments": {},
        "expected_tool": "jira_project_list",
        "expected_arguments": {},
    },
    {
        "name": "unfiltered project collection question",
        "exact_arguments": True,
        "user": "Which Jira projects are available?",
        "allowed_tools": ["jira_project_list"],
        "allowed_arguments": {},
        "expected_tool": "jira_project_list",
        "expected_arguments": {},
    },
    {
        "name": "unfiltered project collection explicit omission",
        "exact_arguments": True,
        "user": "Show every Jira project, with no search text",
        "allowed_tools": ["jira_project_list"],
        "allowed_arguments": {},
        "expected_tool": "jira_project_list",
        "expected_arguments": {},
    },
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
    {
        "name": "issue priority and label update",
        "user": "Set the priority of Jira ticket KAN-3 to High and add the label customer-impact",
        "allowed_tools": ["ticket_update"],
        "allowed_arguments": {
            "ticket_key": ["KAN-3"],
            "priority": ["High"],
            "add_labels": ["customer-impact"],
        },
        "expected_tool": "ticket_update",
        "expected_arguments": {
            "ticket_key": "KAN-3",
            "priority": "High",
            "add_labels": ["customer-impact"],
        },
    },
    {
        "name": "issue due date and label removal",
        "user": "On OPS-17 remove the label stale and set the due date to 2026-12-15",
        "allowed_tools": ["ticket_update"],
        "allowed_arguments": {
            "ticket_key": ["OPS-17"],
            "remove_labels": ["stale"],
            "due_date": ["2026-12-15"],
        },
        "expected_tool": "ticket_update",
        "expected_arguments": {
            "ticket_key": "OPS-17",
            "remove_labels": ["stale"],
            "due_date": "2026-12-15",
        },
    },
    {
        "name": "issue link blocks",
        "user": "Link Jira ticket SUP-12 to SUP-9: SUP-12 blocks SUP-9",
        "allowed_tools": ["ticket_link"],
        "allowed_arguments": {
            "ticket_key": ["SUP-12"],
            "other_ticket_key": ["SUP-9"],
            "relationship": ["blocks"],
        },
        "expected_tool": "ticket_link",
        "expected_arguments": {
            "ticket_key": "SUP-12",
            "other_ticket_key": "SUP-9",
            "relationship": "blocks",
        },
    },
    {
        "name": "issue link inward phrase",
        "user": "Record that HELP-204 is duplicated by HELP-210",
        "allowed_tools": ["ticket_link"],
        "allowed_arguments": {
            "ticket_key": ["HELP-204"],
            "other_ticket_key": ["HELP-210"],
            "relationship": ["is duplicated by"],
        },
        "expected_tool": "ticket_link",
        "expected_arguments": {
            "ticket_key": "HELP-204",
            "other_ticket_key": "HELP-210",
            "relationship": "is duplicated by",
        },
    },
    {
        "name": "issue links read",
        "user": "Which issues are linked to Jira ticket KAN-3?",
        "allowed_tools": ["ticket_links"],
        "allowed_arguments": {"ticket_key": ["KAN-3"]},
        "expected_tool": "ticket_links",
        "expected_arguments": {"ticket_key": "KAN-3"},
    },
    {
        "name": "issue links paraphrase",
        "user": "Show what ENG-88 blocks or is blocked by",
        "allowed_tools": ["ticket_links"],
        "allowed_arguments": {"ticket_key": ["ENG-88"]},
        "expected_tool": "ticket_links",
        "expected_arguments": {"ticket_key": "ENG-88"},
    },
    {
        "name": "assigned tickets by email",
        "user": "Which Jira tickets are assigned to alice@example.com?",
        "allowed_tools": ["ticket_assigned"],
        "allowed_arguments": {"assignee": ["alice@example.com"]},
        "expected_tool": "ticket_assigned",
        "expected_arguments": {"assignee": "alice@example.com"},
    },
    {
        "name": "assigned tickets with status",
        "user": "List the In Progress tickets assigned to bob@example.com",
        "allowed_tools": ["ticket_assigned"],
        "allowed_arguments": {
            "assignee": ["bob@example.com"],
            "status": ["In Progress"],
        },
        "expected_tool": "ticket_assigned",
        "expected_arguments": {
            "assignee": "bob@example.com",
            "status": "In Progress",
        },
    },
    {
        "name": "worklog with comment",
        "user": "Log 1h 30m of work on Jira ticket KAN-3 with the comment: Investigated VPN logs",
        "allowed_tools": ["ticket_worklog_add"],
        "allowed_arguments": {
            "ticket_key": ["KAN-3"],
            "time_spent": ["1h 30m"],
            "comment": ["Investigated VPN logs"],
        },
        "expected_tool": "ticket_worklog_add",
        "expected_arguments": {
            "ticket_key": "KAN-3",
            "time_spent": "1h 30m",
            "comment": "Investigated VPN logs",
        },
    },
    {
        "name": "worklog without comment",
        "user": "Record 45m spent on SEC-12",
        "allowed_tools": ["ticket_worklog_add"],
        "allowed_arguments": {
            "ticket_key": ["SEC-12"],
            "time_spent": ["45m"],
        },
        "expected_tool": "ticket_worklog_add",
        "expected_arguments": {
            "ticket_key": "SEC-12",
            "time_spent": "45m",
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
    "ticket_assigned": {
        "limit": lambda value: (
            isinstance(value, int)
            and not isinstance(value, bool)
            and 1 <= value <= 25
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
    exact: bool = False,
) -> bool:
    if not isinstance(
        observed,
        dict,
    ):
        return False

    if exact:
        return observed == expected

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



def semantic_context(case: dict, capability_catalog: list[dict]) -> dict:
    # Bind only the grounded targets the router admits for this capability.
    # Derived proposal content (a ticket summary) is composed by the model in
    # production; the training corpus drops it from bindings the same way.
    grounded = {name for item in capability_catalog for name in item["grounded_arguments"]}
    return {
        "allowed_tools": case["allowed_tools"],
        "allowed_arguments": {
            name: values for name, values in case["allowed_arguments"].items() if name in grounded
        },
        "forbidden_tools": [],
        "forbidden_arguments": {},
    }


def build_messages(
    *,
    system_prompt: str,
    case: dict,
    capability_catalog: list[dict],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": system_prompt,
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
                    semantic_context(case, capability_catalog),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            ),
        },
        {"role": "user", "content": case["user"]},
    ]


def evaluation_options(argv=None):
    parser = argparse.ArgumentParser(description="Local Jira protocol evaluation; no tool execution or promotion.")
    parser.add_argument("--model-key", default=MODEL_KEY)
    parser.add_argument("--prompt-profile", choices=("standard", "compact", "compact_grounded"),
                        help="Explicit prompt experiment on an in-memory profile; never changes serving settings.")
    parser.add_argument("--report", type=Path, default=REPORT_PATH,
                        help="Fresh output path; existing evaluation evidence is never overwritten.")
    parser.add_argument("--cases", type=Path, default=None,
                        help=f"{CASE_FILE_SCHEMA} file to score instead of the built-in development cases.")
    return parser.parse_args(argv)


def case_set_digest(cases: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps(cases, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validated_case(case) -> dict:
    required = {"name", "user", "allowed_tools", "allowed_arguments", "expected_tool", "expected_arguments"}
    if not isinstance(case, dict) or not required <= set(case):
        raise ValueError("Each case needs name, user, allowed_tools, allowed_arguments, expected_tool and expected_arguments.")
    if not all(isinstance(case[key], str) and case[key].strip() for key in ("name", "user", "expected_tool")):
        raise ValueError(f"Case {case.get('name')!r} has an empty name, request or tool.")
    if case["allowed_tools"] != [case["expected_tool"]]:
        raise ValueError(f"Case {case['name']!r} must allow exactly its expected tool.")
    if not isinstance(case["allowed_arguments"], dict) or not isinstance(case["expected_arguments"], dict):
        raise ValueError(f"Case {case['name']!r} needs allowed_arguments and expected_arguments objects.")
    if "exact_arguments" in case and type(case["exact_arguments"]) is not bool:
        raise ValueError(f"Case {case['name']!r} exact_arguments must be a boolean.")
    return case


def _case_capabilities() -> dict:
    """Read the evaluator's trusted Jira boundary and executable requiredness.

    Signature registration uses explicit in-memory metadata fixtures, without
    deployment Settings, a transport, inference or any tool/provider call.
    """
    from mcp_server import register_capability_signatures
    register_capability_signatures()
    agent = load_agent_definition(PROJECT_ROOT / "subagents/agents/jira-specialist.md")
    return {
        item["name"]: (item, build_capability_contract(item["name"], get_tool(item["name"])))
        for item in build_agent_capability_catalog(agent, include_arguments=True)
    }


_CATALOG_ARGUMENT_TYPES = {
    "str": str, "string": str, "int": int, "integer": int,
    "float": float, "number": float, "bool": bool, "boolean": bool,
    "list[str]": list[str],
}


def _validate_case_capability(case: dict, capabilities: dict) -> None:
    name = case["expected_tool"]
    if name not in capabilities:
        raise ValueError(f"Case {case['name']!r} selects a capability outside the current Jira specialist.")
    descriptor, contract = capabilities[name]
    parameters = descriptor["argument_schema"]
    expected, allowed = case["expected_arguments"], case["allowed_arguments"]
    if (set(expected) | set(allowed)) - set(parameters) or set(contract.required_arguments) - set(expected):
        raise ValueError(f"Case {case['name']!r} arguments do not match the executable capability contract.")
    for key, value in expected.items():
        schema = parameters[key]
        annotation = _CATALOG_ARGUMENT_TYPES.get(schema["type"])
        if annotation is None:
            raise ValueError(f"Case {case['name']!r} uses an unsupported catalog argument type.")
        try:
            TypeAdapter(annotation).validate_python(value, strict=True)
        except ValidationError as exc:
            raise ValueError(f"Case {case['name']!r} has an invalid expected argument type for {key!r}.") from exc
        if "enum" in schema and value not in schema["enum"]:
            raise ValueError(f"Case {case['name']!r} has an expected argument outside its trusted enum.")
    for key, values in allowed.items():
        if (not isinstance(values, list) or not values
                or any(not isinstance(value, str) or not value or value != value.strip() for value in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"Case {case['name']!r} allowed_arguments must contain canonical nonempty string lists.")
        if "enum" in parameters[key] and any(value not in parameters[key]["enum"] for value in values):
            raise ValueError(f"Case {case['name']!r} has an allowed argument outside its trusted enum.")


def load_cases(path: Path | None) -> tuple[str, str, list[dict]]:
    """Return (case set, digest, cases): the built-in development cases or one case file."""
    if path is None:
        case_set, raw_cases = "development", CASES
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Case file must be a JSON object.")
        case_set = payload.get("case_set")
        if payload.get("schema") != CASE_FILE_SCHEMA or not isinstance(case_set, str) or not case_set.strip():
            raise ValueError(f"Case file must declare schema {CASE_FILE_SCHEMA} and a case_set name.")
        raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("Case file cases must be an explicit list.")
    cases = [_validated_case(case) for case in raw_cases]
    if not cases:
        raise ValueError("Case file declares no cases.")
    for field in ("name", "user"):
        if len({case[field] for case in cases}) != len(cases):
            raise ValueError(f"Case file repeats a case {field}.")
    capabilities = _case_capabilities()
    for case in cases:
        _validate_case_capability(case, capabilities)
    return case_set, case_set_digest(cases), cases


def evaluation_settings(settings, *, model_key, prompt_profile=None):
    candidate = settings.model_copy(deep=True)
    profile = candidate.require_model_profile(model_key)
    if prompt_profile is not None:
        values = profile.model_dump()
        values["worker_prompt_profile"] = prompt_profile
        candidate.model_profiles[model_key] = type(profile).model_validate(values)
    return candidate


def case_prompt(agent, case, prompt_profile):
    # Match production disclosure: one validated task's allowed capabilities,
    # not every Jira tool alongside a hidden expected-answer fixture.
    catalog = build_agent_capability_catalog(
        agent, include_arguments=True, allowed_tools=case["allowed_tools"],
    )
    return catalog, build_worker_system_prompt(
        agent, capability_catalog=catalog, prompt_profile=prompt_profile,
    )


async def main(argv=None) -> int:
    args = evaluation_options(argv)
    model_key = args.model_key.strip()
    report_path = args.report.expanduser().resolve()
    try:
        case_set, case_set_sha256, cases = load_cases(args.cases)
    except (OSError, ValueError) as exc:
        print("Evaluation case file is unusable:", exc)
        return 2
    if not model_key or MAX_NEW_TOKENS < 1:
        print(
            "Evaluation requires a model and a positive output budget."
        )
        return 2

    if report_path.exists():
        print("Evaluation output already exists; use a fresh path to preserve evidence.")
        return 2

    try:
        settings = evaluation_settings(Settings(), model_key=model_key, prompt_profile=args.prompt_profile)
        profile = (
            settings
            .require_model_profile(
                model_key
            )
        )

    except Exception as exc:
        print(
            "Jira candidate model profile is unavailable:",
            type(exc).__name__,
        )
        return 2

    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump({"schema": "jira-evaluation-intent.v1", "state": "incomplete",
                       "model_key": model_key, "worker_prompt_profile": profile.worker_prompt_profile,
                       "case_set": case_set, "case_set_sha256": case_set_sha256,
                       "declared_cases": [case["name"] for case in cases],
                       "tools_executed": False, "production_activation": False}, stream, indent=2)
    except FileExistsError:
        print("Evaluation output was claimed concurrently; no inference started.")
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
            model=model_key,
        )
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
        model_key,
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
        len(cases),
        f"({case_set}, sha256 {case_set_sha256})",
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
            model_key,
            stage=InferenceStage.SPECIALIST_REASONING,
        )

        for index, case in enumerate(
            cases,
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

            capability_catalog, system_prompt = case_prompt(agent, case, profile.worker_prompt_profile)
            messages = build_messages(
                system_prompt=system_prompt,
                case=case,
                capability_catalog=capability_catalog,
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
                stage=InferenceStage.SPECIALIST_REASONING,
                model_key=model_key,
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
                            exact=case.get("exact_arguments", False),
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
        model_key=model_key,
        backend=profile.backend,
        quantization=profile.quantization,
        compute_dtype=profile.compute_dtype,
        device_map=profile.device_map,
        cases=case_results,
        case_set=case_set,
        case_set_sha256=case_set_sha256,
    )

    written = write_specialist_model_report(
        report,
        report_path,
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
