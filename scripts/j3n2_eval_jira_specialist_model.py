from __future__ import annotations

import json
import sys

from pathlib import (
    Path,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if (
    str(PROJECT_ROOT)
    not in sys.path
):
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


import torch

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

from subagents.llm.backends.hf_causal_worker import (
    HFCausalWorkerBackend,
)


MODEL_PATH = Path(
    "/mnt/c/project/agenticaiPersonal/Models/BLOOMZ-560M"
)


CASES = [
    {
        "name":
            "exact issue get",

        "user":
            "Get Jira ticket KAN-3",

        "allowed_tools":
            [
                "ticket_get"
            ],

        "allowed_arguments":
            {
                "ticket_key":
                    [
                        "KAN-3"
                    ]
            },

        "expected_tool":
            "ticket_get",

        "expected_arguments":
            {
                "ticket_key":
                    "KAN-3"
            },
    },

    {
        "name":
            "issue search scoped by project",

        "user":
            "Search Jira tickets in project KAN",

        "allowed_tools":
            [
                "ticket_search"
            ],

        "allowed_arguments":
            {
                "project_key":
                    [
                        "KAN"
                    ]
            },

        "expected_tool":
            "ticket_search",

        "expected_arguments":
            {
                "project_key":
                    "KAN"
            },
    },

    {
        "name":
            "issue history",

        "user":
            "Show the history of Jira ticket KAN-3",

        "allowed_tools":
            [
                "ticket_history"
            ],

        "allowed_arguments":
            {
                "ticket_key":
                    [
                        "KAN-3"
                    ]
            },

        "expected_tool":
            "ticket_history",

        "expected_arguments":
            {
                "ticket_key":
                    "KAN-3"
            },
    },

    {
        "name":
            "issue comments read",

        "user":
            "Show the comments on Jira ticket KAN-3",

        "allowed_tools":
            [
                "ticket_comments"
            ],

        "allowed_arguments":
            {
                "ticket_key":
                    [
                        "KAN-3"
                    ]
            },

        "expected_tool":
            "ticket_comments",

        "expected_arguments":
            {
                "ticket_key":
                    "KAN-3"
            },
    },

    {
        "name":
            "project metadata get",

        "user":
            "Get Jira project KAN",

        "allowed_tools":
            [
                "jira_project_get"
            ],

        "allowed_arguments":
            {
                "project_id_or_key":
                    [
                        "KAN"
                    ]
            },

        "expected_tool":
            "jira_project_get",

        "expected_arguments":
            {
                "project_id_or_key":
                    "KAN"
            },
    },

    {
        "name":
            "project search",

        "user":
            "Search Jira projects matching KAN",

        "allowed_tools":
            [
                "jira_project_list"
            ],

        "allowed_arguments":
            {
                "query":
                    [
                        "KAN"
                    ]
            },

        "expected_tool":
            "jira_project_list",

        "expected_arguments":
            {
                "query":
                    "KAN"
            },
    },

    {
        "name":
            "issue assignment",

        "user":
            "Assign Jira ticket KAN-3 to alice@example.com",

        "allowed_tools":
            [
                "ticket_assign"
            ],

        "allowed_arguments":
            {
                "ticket_key":
                    [
                        "KAN-3"
                    ],

                "assignee":
                    [
                        "alice@example.com"
                    ],
            },

        "expected_tool":
            "ticket_assign",

        "expected_arguments":
            {
                "ticket_key":
                    "KAN-3",

                "assignee":
                    "alice@example.com",
            },
    },

    {
        "name":
            "issue transition",

        "user":
            "Move Jira ticket KAN-3 to In Progress",

        "allowed_tools":
            [
                "ticket_transition"
            ],

        "allowed_arguments":
            {
                "ticket_key":
                    [
                        "KAN-3"
                    ],

                "status":
                    [
                        "In Progress"
                    ],
            },

        "expected_tool":
            "ticket_transition",

        "expected_arguments":
            {
                "ticket_key":
                    "KAN-3",

                "status":
                    "In Progress",
            },
    },

    {
        "name":
            "project archive",

        "user":
            "Archive Jira project KAN",

        "allowed_tools":
            [
                "jira_project_archive"
            ],

        "allowed_arguments":
            {
                "project_id_or_key":
                    [
                        "KAN"
                    ]
            },

        "expected_tool":
            "jira_project_archive",

        "expected_arguments":
            {
                "project_id_or_key":
                    "KAN"
            },
    },

    {
        "name":
            "project delete",

        "user":
            "Delete Jira project KAN",

        "allowed_tools":
            [
                "jira_project_delete"
            ],

        "allowed_arguments":
            {
                "project_id_or_key":
                    [
                        "KAN"
                    ]
            },

        "expected_tool":
            "jira_project_delete",

        "expected_arguments":
            {
                "project_id_or_key":
                    "KAN"
            },
    },
]


def semantic_context(
    case: dict,
) -> dict:

    return {
        "allowed_tools":
            case[
                "allowed_tools"
            ],

        "allowed_arguments":
            case[
                "allowed_arguments"
            ],

        "forbidden_tools":
            [],

        "forbidden_arguments":
            {},
    }


def exact_arguments_match(
    actual: dict,
    expected: dict,
) -> bool:
    """
    For this model-selection gate we require exactly the expected
    grounded arguments.

    Optional schema defaults are NOT accepted here because the purpose
    is to measure whether the small model can preserve the semantic
    contract with minimal drift.
    """

    return (
        actual
        == expected
    )


def main() -> int:

    if not MODEL_PATH.exists():

        print(
            "Model path does not exist:",
            MODEL_PATH,
        )

        return 2

    agent = (
        load_agent_definition(
            "subagents/agents/jira-specialist.md"
        )
    )

    capability_catalog = (
        build_agent_capability_catalog(
            agent,
            include_arguments=True,
        )
    )

    system_prompt = (
        build_worker_system_prompt(
            agent,
            capability_catalog=(
                capability_catalog
            ),
        )
    )

    print(
        "========================================"
    )

    print(
        "J3-N2 JIRA SPECIALIST MODEL EVAL"
    )

    print(
        "========================================"
    )

    print(
        "model:",
        MODEL_PATH,
    )

    print(
        "device: CPU"
    )

    print(
        "cases:",
        len(
            CASES
        ),
    )

    print()

    backend = (
        HFCausalWorkerBackend(
            model_path=(
                MODEL_PATH
            ),

            model_load_kwargs={
                "local_files_only":
                    True,

                "torch_dtype":
                    torch.float32,

                "device_map":
                    "cpu",
            },
        )
    )

    passed = 0

    failed = 0

    failures: list[
        str
    ] = []

    for (
        index,
        case,
    ) in enumerate(
        CASES,
        start=1,
    ):

        print(
            "----------------------------------------"
        )

        print(
            f"{index:02d}. "
            f"{case['name']}"
        )

        print(
            "USER:",
            case[
                "user"
            ],
        )

        messages = [
            {
                "role":
                    "system",

                "content":
                    system_prompt,
            },

            {
                "role":
                    "user",

                "content":
                    case[
                        "user"
                    ],
            },

            {
                "role":
                    "user",

                "content":
                    (
                        "Exact semantic target context from "
                        "the validated routing stage.\n"
                        "This is descriptive context only and "
                        "does NOT grant authorization.\n"
                        "For grounded arguments, preserve the "
                        "listed values EXACTLY. Do not replace "
                        "them with placeholders, aliases, "
                        "configuration names, inferred names, "
                        "or rewritten values.\n\n"
                        + json.dumps(
                            semantic_context(
                                case
                            ),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    ),
            },
        ]

        raw = (
            backend.generate(
                messages,
                max_new_tokens=128,
            )
        )

        print(
            "RAW:",
            repr(
                raw
            ),
        )

        try:

            calls = (
                parse_tool_calls(
                    raw
                )
            )

        except Exception as exc:

            failed += 1

            reason = (
                f"{case['name']}: "
                f"parse failure: {exc}"
            )

            failures.append(
                reason
            )

            print(
                "RESULT: FAIL"
            )

            print(
                "REASON:",
                reason,
            )

            continue

        if len(
            calls
        ) != 1:

            failed += 1

            reason = (
                f"{case['name']}: "
                f"expected 1 tool call, "
                f"got {len(calls)}"
            )

            failures.append(
                reason
            )

            print(
                "RESULT: FAIL"
            )

            print(
                "REASON:",
                reason,
            )

            continue

        call = (
            calls[
                0
            ]
        )

        actual_tool = (
            call[
                "name"
            ]
        )

        actual_arguments = (
            call[
                "arguments"
            ]
        )

        expected_tool = (
            case[
                "expected_tool"
            ]
        )

        expected_arguments = (
            case[
                "expected_arguments"
            ]
        )

        tool_ok = (
            actual_tool
            == expected_tool
        )

        args_ok = (
            exact_arguments_match(
                actual_arguments,
                expected_arguments,
            )
        )

        if (
            tool_ok
            and args_ok
        ):

            passed += 1

            print(
                "RESULT: PASS"
            )

        else:

            failed += 1

            reason = (
                f"{case['name']}: "
                f"expected "
                f"{expected_tool}"
                f"{expected_arguments!r}, "
                f"got "
                f"{actual_tool}"
                f"{actual_arguments!r}"
            )

            failures.append(
                reason
            )

            print(
                "RESULT: FAIL"
            )

            print(
                "REASON:",
                reason,
            )

    print()
    print(
        "========================================"
    )

    print(
        "J3-N2 EVAL SUMMARY"
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
        len(
            CASES
        ),
    )

    print()

    if failures:

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
            "J3-N2 BLOOMZ-560M MODEL GATE: FAIL"
        )

        return 1

    print(
        "J3-N2 BLOOMZ-560M MODEL GATE: PASS"
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
