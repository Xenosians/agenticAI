from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
from numbers import Real

from datetime import (
    datetime,
    timezone,
)

from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from config import (
    ModelProfileSettings,
    Settings,
)

from learning.training.dpo_qlora import (
    EXPECTED_TRAINING_VERSIONS,
)

from learning.training.capability_curriculum import (
    build_agent_capability_records,
)

from subagents.core.definitions.loader import (
    load_agent_definition,
    load_agent_directory,
)
from subagents.core.definitions.types import AgentDefinition

from subagents.core.tooling.capabilities import (
    build_agent_capability_catalog,
)

from subagents.core.tooling.parser import (
    parse_tool_calls,
)

from subagents.core.tooling.prompt import (
    build_worker_system_prompt,
)

from subagents.llm.runtime.hf_prompt import (
    render_hf_causal_fallback_prompt,
)


JIRA_SFT_SCHEMA = (
    "jira-specialist-sft-corpus.v1"
)

JIRA_SFT_RECORD_SCHEMA = (
    "jira-specialist-sft-record.v1"
)

DEFAULT_BASE_MODEL_KEY = (
    "jira-func"
)

DEFAULT_TRAINED_MODEL_KEY = (
    "jira-func-trained"
)

# Registration target for a freshly trained checkpoint. It must differ from every
# profile an agent definition serves, so evaluation never changes production.
DEFAULT_CANDIDATE_MODEL_KEY = (
    "jira-func-candidate"
)


# Exact user requests reserved for the isolated Jira model gate.
# Seed SFT generation excludes these strings so the protocol gate
# remains a genuine held-out check rather than training replay.
JIRA_HELD_OUT_EVAL_REQUESTS = {
    "List Jira projects without filtering the collection",
    "Which Jira projects are available?",
    "Show every Jira project, with no search text",
    "Get Jira ticket KAN-3",
    "Search Jira tickets in project KAN",
    "Find Jira issues in OPS containing database timeout",
    "Show the history of Jira ticket KAN-3",
    "Show the comments on Jira ticket KAN-3",
    "Add comment Customer confirmed the VPN is working. to Jira ticket KAN-3",
    "Create Jira ticket in KAN with summary VPN login fails after reboot",
    "Create a Task in OPS with summary Rotate staging credentials",
    "Assign Jira ticket KAN-3 to alice@example.com",
    "Move Jira ticket KAN-3 to In Progress",
    "Get Jira project KAN",
    "Search Jira projects matching KAN",
    "Create Jira project LAB named Automation Lab using software-kanban",
    "Rename Jira project KAN to Platform Kanban",
    "Archive Jira project KAN",
    "Delete Jira project KAN",
    "List issues under OPS, not Jira projects",
    "List Jira projects matching OPS",
    "What discussion is on HELP-204?",
    "Set SEC-12 assignee to bob@example.com",
    "Set the priority of Jira ticket KAN-3 to High and add the label customer-impact",
    "On OPS-17 remove the label stale and set the due date to 2026-12-15",
    "Link Jira ticket SUP-12 to SUP-9: SUP-12 blocks SUP-9",
    "Record that HELP-204 is duplicated by HELP-210",
    "Which issues are linked to Jira ticket KAN-3?",
    "Show what ENG-88 blocks or is blocked by",
    "Which Jira tickets are assigned to alice@example.com?",
    "List the In Progress tickets assigned to bob@example.com",
    "Log 1h 30m of work on Jira ticket KAN-3 with the comment: Investigated VPN logs",
    "Record 45m spent on SEC-12",
}


# ============================================================
# HASH / SERIALIZATION HELPERS
# ============================================================


def _canonical_json(
    value: Any,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )


def _sha256_text(
    value: str,
) -> str:
    return (
        hashlib
        .sha256(
            value.encode(
                "utf-8"
            )
        )
        .hexdigest()
    )


def _sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            chunk = handle.read(
                1024
                * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def _utc_now(
) -> str:
    return (
        datetime
        .now(
            timezone.utc
        )
        .isoformat()
    )


def _run_stamp(
) -> str:
    return (
        datetime
        .now(
            timezone.utc
        )
        .strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
    )


def canonical_tool_call(
    tool_name: str,
    arguments: dict[
        str,
        Any,
    ],
) -> str:
    return _canonical_json(
        [
            {
                "name": tool_name,
                "arguments": arguments,
            }
        ]
    )


def semantic_context_for(
    tool_name: str,
    arguments: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    return {
        "allowed_tools": [
            tool_name
        ],
        # A list argument is an exact collection; the runtime binds its
        # items directly, not a list of lists.
        "allowed_arguments": {
            key: (
                list(value)
                if isinstance(value, list)
                else [value]
            )
            for (
                key,
                value,
            ) in arguments.items()
        },
        "forbidden_tools": [],
        "forbidden_arguments": {},
    }


def build_runtime_messages(
    *,
    system_prompt: str,
    user_request: str,
    semantic_context: dict[
        str,
        Any,
    ],
    current_request_last: bool = False,
    verified_account: dict[str, str] | None = None,
) -> list[
    dict[
        str,
        str,
    ]
]:
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_request,
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
                    semantic_context,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            ),
        },
    ]
    if current_request_last:
        # The live worker order: system prompt, target contract, then (for a
        # ticket continuation) the verified-account message, request last.
        ordered = [messages[0], messages[2]]
        if verified_account is not None:
            from agent.onboarding import VerifiedAccountEvidence, ticket_continuation_message

            ordered.append(ticket_continuation_message(VerifiedAccountEvidence(**verified_account)))
        return [*ordered, messages[1]]
    if verified_account is not None:
        raise ValueError("A verified-account continuation needs the runtime-task layout.")
    return messages


# ============================================================
# CORPUS SCHEMAS
# ============================================================


class JiraSftRecord(
    BaseModel
):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default=(
            JIRA_SFT_RECORD_SCHEMA
        ),
        alias="schema",
    )

    record_id: str
    split: str
    tool_name: str
    user_request: str
    semantic_context: dict[
        str,
        Any,
    ]
    target_response: str
    # Present only for an explicitly built runtime-task corpus. Verification
    # reconstructs these messages from trusted current capabilities.
    prompt_messages: list[dict[str, str]] | None = None
    # The verified account a ticket continuation receives after an approved
    # account creation. The runtime prompt then carries the same workflow
    # message production sends; the target names this account in its summary.
    verified_account: dict[str, str] | None = None
    tags: list[str] = Field(
        default_factory=list
    )


class JiraSftCorpusManifest(
    BaseModel
):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default=(
            JIRA_SFT_SCHEMA
        ),
        alias="schema",
    )

    created_at: str
    source: str
    agent_name: str
    base_model_key: str
    worker_prompt_profile: str
    record_count: int
    train_count: int
    validation_count: int
    per_tool_counts: dict[
        str,
        int,
    ]
    train_sha256: str
    validation_sha256: str
    agent_definition_sha256: str
    capability_catalog_sha256: str
    system_prompt_sha256: str
    prompt_layout: Literal["legacy", "runtime_task"] = "legacy"


class JiraValidationSummary(BaseModel):
    """Validation record of one Jira SFT run (see ``summarize_validation``)."""

    model_config = ConfigDict(extra="forbid")

    optimizer_steps: int
    max_steps: int
    early_stopped: bool
    evaluated_steps: list[int]
    baseline_eval_loss: float | None
    best_eval_loss: float
    best_checkpoint_step: int


class JiraSftTrainingManifest(
    BaseModel
):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default=(
            "jira-specialist-sft-training.v1"
        ),
        alias="schema",
    )

    created_at: str
    base_model_key: str
    base_model_path: str
    # Historical manifests did not bind the base checkpoint/tokenizer bytes.
    base_model_identity: dict[str, str] | None = None
    output_directory: str
    adapter_directory: str
    merged_model_directory: str
    corpus_directory: str
    corpus_train_sha256: str
    corpus_validation_sha256: str
    # Optional only for historical manifests. New runs bind the exact prompt
    # contract as well as the two supervised split identities.
    worker_prompt_profile: str | None = None
    system_prompt_sha256: str | None = None
    prompt_layout: Literal["legacy", "runtime_task"] | None = None
    max_steps: int
    learning_rate: float
    max_length: int
    lora_r: int
    lora_alpha: int
    # Optional only for historical manifests trained without validation.
    validation: JiraValidationSummary | None = None
    trainable_parameters: int
    total_parameters: int
    trainable_ratio: float
    adapter_sha256: str
    merged_model_sha256: str
    training_versions: dict[
        str,
        str,
    ]


# ============================================================
# DETERMINISTIC REVIEWED SEED CORPUS
# ============================================================


def _seed_examples(
) -> list[
    tuple[
        str,
        str,
        dict[str, Any],
        list[str],
    ]
]:
    """
    Human-readable deterministic seed examples.

    These examples contain only fully specified routed Jira tasks.
    They teach capability selection + exact argument extraction.
    They are not provider results and they never authorize execution.
    """

    examples: list[
        tuple[
            str,
            str,
            dict[str, Any],
            list[str],
        ]
    ] = []

    # Explicit unfiltered requests teach omission of optional arguments.
    # Held-out paraphrases above are excluded from all generated supervision.
    for request in (
        "List all Jira projects",
        "Show the complete Jira project collection",
        "Fetch Jira projects without a query",
        "Display the Jira project catalog",
        "Retrieve the full list of Jira projects",
        "Give me all projects from Jira",
        "Enumerate Jira projects with no filter",
        "Please show all available projects in Jira",
        "Read the Jira project listing",
        "Return the unfiltered Jira project collection",
        "Pull up each project defined in Jira",
        "I need the whole Jira project inventory",
        "Print out all projects on the Jira instance",
        "Give me an overview of every Jira project",
        "Show me each and every Jira project",
        "Return all Jira projects, please",
    ):
        examples.append((
            "jira_project_list",
            request,
            {},
            ["project", "read", "unfiltered", "empty-arguments"],
        ))

    ticket_keys = [
        "KAN-3",
        "OPS-17",
        "HELP-204",
        "ENG-88",
        "SEC-12",
        "APP-41",
    ]

    project_keys = [
        "KAN",
        "OPS",
        "HELP",
        "ENG",
        "SEC",
        "APP",
    ]

    # Exact issue retrieval.
    get_templates = [
        "Get Jira ticket {ticket}",
        "Show Jira issue {ticket}",
        "Open the Jira ticket metadata for {ticket}",
        "Retrieve issue {ticket} from Jira",
        "What is the current state of Jira ticket {ticket}?",
        "Fetch Jira issue {ticket}",
    ]

    for index, ticket in enumerate(
        ticket_keys
    ):
        for template in get_templates:
            examples.append(
                (
                    "ticket_get",
                    template.format(
                        ticket=ticket
                    ),
                    {
                        "ticket_key": ticket,
                    },
                    [
                        "issue",
                        "read",
                        "exact-id",
                    ],
                )
            )

    # Ticket search with project-as-scope hardening.
    search_templates = [
        "Search Jira tickets in project {project}",
        "List Jira issues under {project}",
        "Find tickets inside Jira project {project}",
        "Show issues from {project}, not Jira projects",
        "Which Jira tickets are in project {project}?",
        "Search issues scoped to {project}",
    ]

    for project in project_keys:
        for template in search_templates:
            examples.append(
                (
                    "ticket_search",
                    template.format(
                        project=project
                    ),
                    {
                        "project_key": project,
                    },
                    [
                        "issue",
                        "search",
                        "project-scope",
                    ],
                )
            )

    search_variants = [
        (
            "Find Jira issues in OPS containing database timeout",
            {
                "project_key": "OPS",
                "text": "database timeout",
            },
        ),
        (
            "Search SEC tickets with priority High",
            {
                "project_key": "SEC",
                "priority": "High",
            },
        ),
        (
            "Show HELP issues with status In Progress",
            {
                "project_key": "HELP",
                "status": "In Progress",
            },
        ),
        (
            "Find tickets containing certificate expired",
            {
                "text": "certificate expired",
            },
        ),
        (
            "Search APP issues with status Done",
            {
                "project_key": "APP",
                "status": "Done",
            },
        ),
        (
            "Find KAN tickets containing VPN",
            {
                "project_key": "KAN",
                "text": "VPN",
            },
        ),
        (
            "Search ENG tickets with priority Highest",
            {
                "project_key": "ENG",
                "priority": "Highest",
            },
        ),
        (
            "Find OPS issues with status To Do containing staging",
            {
                "project_key": "OPS",
                "status": "To Do",
                "text": "staging",
            },
        ),
        # Search words are not router-bound: the model copies them alone and
        # beside every filter, including a bound project. Plain searches stay
        # the majority so a status or priority is emitted only when stated, and
        # searches without a project stay few beside unfiltered project listing.
        ("Find INFRA issues about disk full", {"project_key": "INFRA", "text": "disk full"}),
        (
            "Look in project QA for issues containing SSO redirect loop",
            {"project_key": "QA", "text": "SSO redirect loop"},
        ),
        ("Find tickets in PAY regarding license expired", {"project_key": "PAY", "text": "license expired"}),
        (
            "In project HELP, find tickets with the words password reset",
            {"project_key": "HELP", "text": "password reset"},
        ),
        (
            "Search Jira issues for the text email bounce in project APP",
            {"project_key": "APP", "text": "email bounce"},
        ),
        ("Search DEV tickets containing API 502", {"project_key": "DEV", "text": "API 502"}),
        ("Which CS issues mention backup failed?", {"project_key": "CS", "text": "backup failed"}),
        ("Show issues in INFRA that mention out of memory", {"project_key": "INFRA", "text": "out of memory"}),
        ("Search ENG for issues about API 502", {"project_key": "ENG", "text": "API 502"}),
        ("Find SEC issues mentioning MFA prompt", {"project_key": "SEC", "text": "MFA prompt"}),
        ("Look for tickets in QA containing slow login", {"project_key": "QA", "text": "slow login"}),
        ("Are there OPS tickets about DNS failure?", {"project_key": "OPS", "text": "DNS failure"}),
        (
            "Search the KAN project for issues regarding printer offline",
            {"project_key": "KAN", "text": "printer offline"},
        ),
        ("Search Jira for tickets mentioning password reset", {"text": "password reset"}),
        ("Are there any issues about MFA prompt?", {"text": "MFA prompt"}),
        ("Show Blocked tickets that mention email bounce", {"status": "Blocked", "text": "email bounce"}),
        ("Find High priority issues about slow login", {"priority": "High", "text": "slow login"}),
        (
            "Search DEV tickets with status In Review containing disk full",
            {"project_key": "DEV", "status": "In Review", "text": "disk full"},
        ),
        (
            "Show In Progress issues in PAY that mention license expired",
            {"project_key": "PAY", "status": "In Progress", "text": "license expired"},
        ),
        (
            "List CS issues with priority Highest that mention backup failed",
            {"project_key": "CS", "priority": "Highest", "text": "backup failed"},
        ),
    ]

    for user_request, arguments in search_variants:
        examples.append(
            (
                "ticket_search",
                user_request,
                arguments,
                [
                    "issue",
                    "search",
                    "filters",
                ],
            )
        )

    # History and comments are intentionally contrasted.
    for ticket in ticket_keys:
        history_templates = [
            "Show the history of Jira ticket {ticket}",
            "What changed on Jira issue {ticket}?",
            "Retrieve the change history for {ticket}",
            "Show field changes on {ticket}",
        ]

        comments_templates = [
            "Show the comments on Jira ticket {ticket}",
            "What discussion is on {ticket}?",
            "Read comments from Jira issue {ticket}",
            "Show discussion entries attached to {ticket}",
        ]

        for template in history_templates:
            examples.append(
                (
                    "ticket_history",
                    template.format(
                        ticket=ticket
                    ),
                    {
                        "ticket_key": ticket,
                    },
                    [
                        "issue",
                        "history",
                        "read",
                    ],
                )
            )

        for template in comments_templates:
            examples.append(
                (
                    "ticket_comments",
                    template.format(
                        ticket=ticket
                    ),
                    {
                        "ticket_key": ticket,
                    },
                    [
                        "issue",
                        "comments",
                        "read",
                    ],
                )
            )

    comments = [
        "Customer confirmed the VPN is working.",
        "Waiting for the network team.",
        "Please retest after the maintenance window.",
        "Logs attached by the requester.",
        "Issue reproduced on staging only.",
        "Resolved after certificate rotation.",
    ]

    for index, ticket in enumerate(
        ticket_keys
    ):
        comment = comments[index]
        examples.extend(
            [
                (
                    "ticket_add_comment",
                    f"Add comment {comment} to Jira ticket {ticket}",
                    {
                        "ticket_key": ticket,
                        "comment": comment,
                    },
                    [
                        "issue",
                        "comment",
                        "mutation",
                        "exact-text",
                    ],
                ),
                (
                    "ticket_add_comment",
                    f"Post exactly this note on {ticket}: {comment}",
                    {
                        "ticket_key": ticket,
                        "comment": comment,
                    },
                    [
                        "issue",
                        "comment",
                        "mutation",
                        "exact-text",
                    ],
                ),
            ]
        )

    summaries = [
        "VPN login fails after reboot",
        "Rotate staging credentials",
        "Certificate expires next week",
        "Database timeout during deploy",
        "Laptop enrollment is blocked",
        "Service account permission review",
    ]

    ticket_types = [
        "Task",
        "Bug",
        "Story",
    ]

    for index, project in enumerate(
        project_keys
    ):
        summary = summaries[index]

        examples.extend(
            [
                (
                    "ticket_create",
                    f"Create Jira ticket in {project} with summary {summary}",
                    {
                        "project_key": project,
                        "summary": summary,
                    },
                    [
                        "issue",
                        "create",
                        "mutation",
                    ],
                ),
                (
                    "ticket_create",
                    f"Open a Jira issue in {project} titled {summary}",
                    {
                        "project_key": project,
                        "summary": summary,
                    },
                    [
                        "issue",
                        "create",
                        "mutation",
                    ],
                ),
            ]
        )

        ticket_type = ticket_types[
            index
            % len(
                ticket_types
            )
        ]

        examples.append(
            (
                "ticket_create",
                f"Create a {ticket_type} in {project} with summary {summary}",
                {
                    "project_key": project,
                    "summary": summary,
                    "ticket_type": ticket_type,
                },
                [
                    "issue",
                    "create",
                    "mutation",
                    "type",
                ],
            )
        )

    # More creates with and without an explicit type, so a bound ticket_type
    # is always carried into the call and never invented when absent.
    for user_request, arguments in (
        (
            "Raise a Bug in QA titled Checkout button does nothing",
            {"project_key": "QA", "summary": "Checkout button does nothing", "ticket_type": "Bug"},
        ),
        (
            "Write up a Story in DEV with summary Export reports as CSV",
            {"project_key": "DEV", "summary": "Export reports as CSV", "ticket_type": "Story"},
        ),
        (
            "File a Task in INFRA titled Patch the jump host kernel",
            {"project_key": "INFRA", "summary": "Patch the jump host kernel", "ticket_type": "Task"},
        ),
        (
            "Open a Bug in PAY with summary Refund total shows twice",
            {"project_key": "PAY", "summary": "Refund total shows twice", "ticket_type": "Bug"},
        ),
        (
            "Open a ticket in CS titled Chat widget fails to load",
            {"project_key": "CS", "summary": "Chat widget fails to load"},
        ),
        (
            "Create an issue in QA with summary Nightly suite skips smoke tests",
            {"project_key": "QA", "summary": "Nightly suite skips smoke tests"},
        ),
    ):
        examples.append(
            (
                "ticket_create",
                user_request,
                arguments,
                ["issue", "create", "mutation", *(["type"] if "ticket_type" in arguments else [])],
            )
        )

    assignees = [
        "alice@example.com",
        "bob@example.com",
        "charlie@example.com",
        "dina@example.com",
        "eric@example.com",
        "farah@example.com",
    ]

    statuses = [
        "In Progress",
        "Done",
        "To Do",
        "Blocked",
        "Open",
        "In Review",
    ]

    for index, ticket in enumerate(
        ticket_keys
    ):
        assignee = assignees[index]
        status = statuses[index]

        examples.extend(
            [
                (
                    "ticket_assign",
                    f"Assign Jira ticket {ticket} to {assignee}",
                    {
                        "ticket_key": ticket,
                        "assignee": assignee,
                    },
                    [
                        "issue",
                        "assign",
                        "mutation",
                    ],
                ),
                (
                    "ticket_assign",
                    f"Set {ticket} assignee to {assignee}",
                    {
                        "ticket_key": ticket,
                        "assignee": assignee,
                    },
                    [
                        "issue",
                        "assign",
                        "mutation",
                    ],
                ),
                (
                    "ticket_transition",
                    f"Move Jira ticket {ticket} to {status}",
                    {
                        "ticket_key": ticket,
                        "status": status,
                    },
                    [
                        "issue",
                        "transition",
                        "mutation",
                    ],
                ),
                (
                    "ticket_transition",
                    f"Change {ticket} status to {status}",
                    {
                        "ticket_key": ticket,
                        "status": status,
                    },
                    [
                        "issue",
                        "transition",
                        "mutation",
                    ],
                ),
            ]
        )

    # Field updates use only exact user-supplied values.
    update_examples = [
        (
            "Change the priority of Jira ticket OPS-17 to Highest",
            {"ticket_key": "OPS-17", "priority": "Highest"},
        ),
        (
            "Add the label vpn to HELP-204",
            {"ticket_key": "HELP-204", "add_labels": ["vpn"]},
        ),
        (
            "Set the due date of ENG-88 to 2026-11-30",
            {"ticket_key": "ENG-88", "due_date": "2026-11-30"},
        ),
        (
            "Remove the label triage from SEC-12 and make its priority Low",
            {"ticket_key": "SEC-12", "priority": "Low", "remove_labels": ["triage"]},
        ),
        (
            "On APP-41 add the labels backend and release-blocker",
            {"ticket_key": "APP-41", "add_labels": ["backend", "release-blocker"]},
        ),
        (
            "Lower KAN-3 to priority Medium",
            {"ticket_key": "KAN-3", "priority": "Medium"},
        ),
        # Label lists of one to three items, alone and beside every companion
        # field, so the list is always closed before the next key.
        ("Tag QA-19 with the label regression", {"ticket_key": "QA-19", "add_labels": ["regression"]}),
        ("Add labels frontend and mobile to INFRA-77", {"ticket_key": "INFRA-77", "add_labels": ["frontend", "mobile"]}),
        (
            "Put the labels audit, billing and onboarding on DEV-150",
            {"ticket_key": "DEV-150", "add_labels": ["audit", "billing", "onboarding"]},
        ),
        ("Drop the label needs-info from CS-8", {"ticket_key": "CS-8", "remove_labels": ["needs-info"]}),
        (
            "Remove the labels hotfix and performance from PAY-23",
            {"ticket_key": "PAY-23", "remove_labels": ["hotfix", "performance"]},
        ),
        (
            "On ENG-88 add the label network and remove the label triage",
            {"ticket_key": "ENG-88", "add_labels": ["network"], "remove_labels": ["triage"]},
        ),
        (
            "Set QA-19 to priority Highest and label it security-review",
            {"ticket_key": "QA-19", "priority": "Highest", "add_labels": ["security-review"]},
        ),
        (
            "Clear the label documentation from INFRA-77 and push the due date to 2027-01-08",
            {"ticket_key": "INFRA-77", "remove_labels": ["documentation"], "due_date": "2027-01-08"},
        ),
        (
            "For DEV-150 add the labels backend and hotfix, set priority High and make it due 2026-11-14",
            {"ticket_key": "DEV-150", "add_labels": ["backend", "hotfix"], "priority": "High", "due_date": "2026-11-14"},
        ),
        (
            "Lower CS-8 to priority Low and take off the label regression",
            {"ticket_key": "CS-8", "priority": "Low", "remove_labels": ["regression"]},
        ),
        (
            "Label PAY-23 as billing and mobile, and remove the label audit",
            {"ticket_key": "PAY-23", "add_labels": ["billing", "mobile"], "remove_labels": ["audit"]},
        ),
        (
            "Please add the label performance to APP-41 with priority Medium",
            {"ticket_key": "APP-41", "add_labels": ["performance"], "priority": "Medium"},
        ),
        (
            "HELP-204 needs the labels onboarding and documentation",
            {"ticket_key": "HELP-204", "add_labels": ["onboarding", "documentation"]},
        ),
        ("Untag SEC-12: remove the labels vpn and network", {"ticket_key": "SEC-12", "remove_labels": ["vpn", "network"]}),
        (
            "Give KAN-3 the label frontend and a due date of 2026-12-01",
            {"ticket_key": "KAN-3", "add_labels": ["frontend"], "due_date": "2026-12-01"},
        ),
        (
            "Remove the label release-blocker from OPS-17; it is now due 2026-10-31",
            {"ticket_key": "OPS-17", "remove_labels": ["release-blocker"], "due_date": "2026-10-31"},
        ),
        (
            "Set priority Lowest on QA-19 and drop the labels triage and needs-info",
            {"ticket_key": "QA-19", "priority": "Lowest", "remove_labels": ["triage", "needs-info"]},
        ),
        (
            "Mark INFRA-77 with the labels hotfix, network and security-review",
            {"ticket_key": "INFRA-77", "add_labels": ["hotfix", "network", "security-review"]},
        ),
        (
            "Take the label mobile off DEV-150 and add the label frontend instead",
            {"ticket_key": "DEV-150", "remove_labels": ["mobile"], "add_labels": ["frontend"]},
        ),
        (
            "Make CS-8 priority High, add the label backend, and set it due 2027-01-08",
            {"ticket_key": "CS-8", "priority": "High", "add_labels": ["backend"], "due_date": "2027-01-08"},
        ),
    ]

    for user_request, arguments in update_examples:
        examples.append(
            (
                "ticket_update",
                user_request,
                arguments,
                [
                    "issue",
                    "update",
                    "mutation",
                    "exact-values",
                ],
            )
        )

    # Links keep the user's subject first and the relationship words exact.
    link_examples = [
        ("OPS-17", "blocks", "ENG-88", "Mark OPS-17 as blocking ENG-88: OPS-17 blocks ENG-88"),
        ("HELP-204", "relates to", "KAN-3", "Link HELP-204 to KAN-3: HELP-204 relates to KAN-3"),
        ("SEC-12", "is blocked by", "APP-41", "Note that SEC-12 is blocked by APP-41"),
        ("ENG-88", "duplicates", "OPS-17", "ENG-88 duplicates OPS-17, please link them"),
        ("APP-41", "causes", "HELP-204", "Link APP-41 and HELP-204 so that APP-41 causes HELP-204"),
        ("KAN-3", "is caused by", "SEC-12", "Record that KAN-3 is caused by SEC-12"),
    ]

    for ticket, relationship, other, user_request in link_examples:
        examples.append(
            (
                "ticket_link",
                user_request,
                {
                    "ticket_key": ticket,
                    "other_ticket_key": other,
                    "relationship": relationship,
                },
                [
                    "issue",
                    "link",
                    "mutation",
                ],
            )
        )

    # Reading links is contrasted with creating them.
    links_templates = [
        "Which issues are linked to {ticket}?",
        "Show the issue links of Jira ticket {ticket}",
        "What is {ticket} linked to?",
    ]

    for ticket in ticket_keys[:4]:
        for template in links_templates:
            examples.append(
                (
                    "ticket_links",
                    template.format(
                        ticket=ticket
                    ),
                    {
                        "ticket_key": ticket,
                    },
                    [
                        "issue",
                        "links",
                        "read",
                    ],
                )
            )

    # Assigned-ticket reads name the user explicitly.
    assigned_examples = [
        ("Which Jira tickets are assigned to charlie@example.com?", {"assignee": "charlie@example.com"}),
        ("Show tickets assigned to dina@example.com", {"assignee": "dina@example.com"}),
        ("List the Done tickets assigned to eric@example.com", {"assignee": "eric@example.com", "status": "Done"}),
        ("What is farah@example.com working on in Jira?", {"assignee": "farah@example.com"}),
        ("Show open tickets assigned to alice@example.com with status Open", {"assignee": "alice@example.com", "status": "Open"}),
        ("Which issues does Jira user 5b10ac8d82e05b22cc7d4ef5 have assigned?", {"assignee": "5b10ac8d82e05b22cc7d4ef5"}),
    ]

    for user_request, arguments in assigned_examples:
        examples.append(
            (
                "ticket_assigned",
                user_request,
                arguments,
                [
                    "issue",
                    "assigned",
                    "read",
                ],
            )
        )

    # Worklogs keep the exact duration and comment.
    worklog_examples = [
        ("Log 2h of work on OPS-17", {"ticket_key": "OPS-17", "time_spent": "2h"}),
        ("Log 30m on HELP-204 with the comment: Called the requester", {"ticket_key": "HELP-204", "time_spent": "30m", "comment": "Called the requester"}),
        ("Add a worklog of 1d 2h to ENG-88", {"ticket_key": "ENG-88", "time_spent": "1d 2h"}),
        ("Record 15m spent on APP-41 with the comment: Restarted the service", {"ticket_key": "APP-41", "time_spent": "15m", "comment": "Restarted the service"}),
        ("Track 3h 15m of work on KAN-3", {"ticket_key": "KAN-3", "time_spent": "3h 15m"}),
        ("I spent 45m on SEC-12, log it with the comment: Reviewed firewall rules", {"ticket_key": "SEC-12", "time_spent": "45m", "comment": "Reviewed firewall rules"}),
    ]

    for user_request, arguments in worklog_examples:
        examples.append(
            (
                "ticket_worklog_add",
                user_request,
                arguments,
                [
                    "issue",
                    "worklog",
                    "mutation",
                    "exact-text",
                ],
            )
        )

    # Project reads are intentionally contrasted with ticket searches.
    project_names = [
        "Kanban Platform",
        "Operations Hub",
        "Help Center",
        "Engineering Core",
        "Security Operations",
        "Application Services",
    ]

    for index, project in enumerate(
        project_keys
    ):
        project_name = project_names[index]

        examples.extend(
            [
                (
                    "jira_project_get",
                    f"Get Jira project {project}",
                    {
                        "project_id_or_key": project,
                    },
                    [
                        "project",
                        "read",
                    ],
                ),
                (
                    "jira_project_get",
                    f"Show metadata for Jira project {project}",
                    {
                        "project_id_or_key": project,
                    },
                    [
                        "project",
                        "read",
                    ],
                ),
                (
                    "jira_project_list",
                    f"Search Jira projects matching {project}",
                    {
                        "query": project,
                    },
                    [
                        "project",
                        "search",
                    ],
                ),
                (
                    "jira_project_list",
                    f"List Jira projects matching {project}",
                    {
                        "query": project,
                    },
                    [
                        "project",
                        "search",
                    ],
                ),
                (
                    "jira_project_update",
                    f"Rename Jira project {project} to {project_name} v2",
                    {
                        "project_id_or_key": project,
                        "new_name": (
                            f"{project_name} v2"
                        ),
                    },
                    [
                        "project",
                        "rename",
                        "mutation",
                    ],
                ),
                (
                    "jira_project_archive",
                    f"Archive Jira project {project}",
                    {
                        "project_id_or_key": project,
                    },
                    [
                        "project",
                        "archive",
                        "mutation",
                        "high-risk",
                    ],
                ),
                (
                    "jira_project_delete",
                    f"Delete Jira project {project}",
                    {
                        "project_id_or_key": project,
                    },
                    [
                        "project",
                        "delete",
                        "mutation",
                        "high-risk",
                    ],
                ),
            ]
        )

    create_projects = [
        (
            "LAB",
            "Automation Lab",
            "software-kanban",
        ),
        (
            "SCRUM",
            "Delivery Scrum",
            "software-scrum",
        ),
        (
            "BUS",
            "Business Tasks",
            "business-task-tracking",
        ),
        (
            "SERV",
            "Service Desk",
            "service-it-management",
        ),
    ]

    for (
        project_key,
        project_name,
        template,
    ) in create_projects:
        examples.extend(
            [
                (
                    "jira_project_create",
                    (
                        f"Create Jira project {project_key} named "
                        f"{project_name} using {template}"
                    ),
                    {
                        "project_key": project_key,
                        "project_name": project_name,
                        "template": template,
                    },
                    [
                        "project",
                        "create",
                        "mutation",
                        "template",
                    ],
                ),
                (
                    "jira_project_create",
                    (
                        f"Make a Jira project with key {project_key}, "
                        f"name {project_name}, template {template}"
                    ),
                    {
                        "project_key": project_key,
                        "project_name": project_name,
                        "template": template,
                    },
                    [
                        "project",
                        "create",
                        "mutation",
                        "template",
                    ],
                ),
            ]
        )

    return examples


def _agent_identity_hash(
    agent,
) -> str:
    return _sha256_text(
        _canonical_json(
            {
                "name": agent.name,
                "description": agent.description,
                "model": agent.model,
                "tools": list(
                    agent.tools
                ),
                "max_steps": agent.max_steps,
                "system_prompt": agent.system_prompt,
            }
        )
    )


def _build_records(
    *,
    project_root: Path | None = None,
) -> list[
    JiraSftRecord
]:
    """
    Build a deterministic hybrid Jira supervision corpus.

    Existing callers may still call _build_records() with no arguments.
    Catalog-derived language examples augment the existing reviewed seed
    instead of replacing it.
    """

    if project_root is None:
        project_root = (
            Path(__file__)
            .resolve()
            .parents[2]
        )

    project_root = (
        project_root
        .expanduser()
        .resolve()
    )

    raw = _seed_examples()

    per_tool_index: dict[
        str,
        int,
    ] = {}

    records: list[
        JiraSftRecord
    ] = []

    seen_requests: set[
        str
    ] = set()

    for (
        tool_name,
        user_request,
        arguments,
        tags,
    ) in raw:
        normalized_request = (
            user_request.strip()
        )

        if (
            normalized_request
            in JIRA_HELD_OUT_EVAL_REQUESTS
        ):
            continue

        if normalized_request in seen_requests:
            raise ValueError(
                "Duplicate Jira SFT user request: "
                f"{normalized_request}"
            )

        seen_requests.add(
            normalized_request
        )

        index = per_tool_index.get(
            tool_name,
            0,
        )

        per_tool_index[
            tool_name
        ] = index + 1

        split = (
            "validation"
            if index % 5 == 0
            else "train"
        )

        record_id = (
            f"jira-sft-{tool_name}-{index:04d}-"
            + _sha256_text(
                normalized_request
            )[:12]
        )

        records.append(
            JiraSftRecord(
                record_id=record_id,
                split=split,
                tool_name=tool_name,
                user_request=normalized_request,
                semantic_context=(
                    semantic_context_for(
                        tool_name,
                        arguments,
                    )
                ),
                target_response=(
                    canonical_tool_call(
                        tool_name,
                        arguments,
                    )
                ),
                tags=[
                    *tags,
                    "reviewed-seed",
                ],
            )
        )

    generated = (
        build_agent_capability_records(
            project_root=(
                project_root
            ),
            agent_definition=(
                Path(
                    "subagents/agents/jira-specialist.md"
                )
            ),
            excluded_requests=(
                JIRA_HELD_OUT_EVAL_REQUESTS
            ),
        )
    )

    for generated_record in generated:
        normalized_request = (
            generated_record
            .user_request
            .strip()
        )

        if not normalized_request:
            continue

        if (
            normalized_request
            in JIRA_HELD_OUT_EVAL_REQUESTS
        ):
            continue

        if normalized_request in seen_requests:
            continue

        seen_requests.add(
            normalized_request
        )

        records.append(
            JiraSftRecord(
                record_id=(
                    "jira-sft-"
                    + generated_record.record_id
                ),
                split=(
                    generated_record.split
                ),
                tool_name=(
                    generated_record.tool_name
                ),
                user_request=(
                    normalized_request
                ),
                semantic_context=(
                    semantic_context_for(
                        generated_record.tool_name,
                        generated_record.arguments,
                    )
                ),
                target_response=(
                    generated_record.target_response
                ),
                tags=[
                    *generated_record.tags,
                    "capability-language-v2.1",
                ],
            )
        )

    for record in _continuation_records():
        if record.user_request in seen_requests:
            raise ValueError(f"Duplicate Jira SFT user request: {record.user_request}")
        seen_requests.add(record.user_request)
        records.append(record)

    return records


# Request shapes for the account -> Jira continuation. In production the Jira
# specialist receives the user's whole original request after the account step
# succeeds; the held-out continuation cases use none of these shapes.
CONTINUATION_REQUEST_TEMPLATES = (
    "Create a corporate account for given name {given}, family name {family}, "
    "department {department} and role {role}. After the account is successfully "
    "created, create a Jira ticket in project {project} for that created account, "
    "then notify me by email.",
    "Please create an account for {given} {family} ({role}, {department}) and, once "
    "it exists, open a Jira ticket in {project} for the new account.",
    "Onboard {given} {family} as {role} in {department}: create their directory "
    "account first, then raise a Jira issue in {project} about it.",
    "Set up a new account for {given} {family}, department {department}, role {role}. "
    "When that succeeds, file a ticket in Jira project {project} to {purpose}.",
)

# (given, family, department, role, project, verified user_id, verified email,
#  purpose or None, summary). The verified identity is copied exactly from the
# provider read-back. Ids take every shape account creation produces: derived
# from the names, with a numeric suffix when the base id is taken, and
# generated, digit-heavy family names for service and lab accounts. Plain
# examples rotate over the request templates in order, so each template sees
# every shape; the summary always names the account.
_CONTINUATION_EXAMPLES = (
    ("Maria", "Santos", "Finance", "Analyst", "KAN", "msantos", "msantos@example.com", None,
     "Onboarding follow-up for msantos (msantos@example.com)"),
    ("Kenji", "Tanaka", "Engineering", "Developer", "OPS", "ktanaka", "ktanaka@corp.example", None,
     "Onboarding follow-up for ktanaka (ktanaka@corp.example)"),
    ("Build", "Agent4471", "Platform", "Service Account", "APP", "bagent4471", "bagent4471@example.org", None,
     "Onboarding follow-up for bagent4471 (bagent4471@example.org)"),
    ("Aisha", "Bello", "HR", "Recruiter", "HELP", "abello2", "abello2@example.org", None,
     "Onboarding follow-up for abello2 (abello2@example.org)"),
    ("Lucas", "Moreau", "Sales", "Account Manager", "ENG", "lmoreau", "lmoreau@example.com", None,
     "Onboarding follow-up for lmoreau (lmoreau@example.com)"),
    ("Deploy", "Bot7c3e91", "IT", "Automation", "DEV", "dbot7c3e91", "dbot7c3e91@corp.example", None,
     "Onboarding follow-up for dbot7c3e91 (dbot7c3e91@corp.example)"),
    ("Scan", "Host0a8812", "Security", "Scanner", "SEC", "shost0a8812", "shost0a8812@example.com", None,
     "Onboarding follow-up for shost0a8812 (shost0a8812@example.com)"),
    ("Nina", "Kowalski", "Finance", "Accountant", "PAY", "nkowalski2", "nkowalski2@example.org", None,
     "Onboarding follow-up for nkowalski2 (nkowalski2@example.org)"),
    ("Priya", "Nair", "Security", "Engineer", "SEC", "pnair", "pnair@corp.example", None,
     "Onboarding follow-up for pnair (pnair@corp.example)"),
    ("Tomas", "Novak", "Operations", "Technician", "APP", "tnovak", "tnovak@example.org", None,
     "Onboarding follow-up for tnovak (tnovak@example.org)"),
    ("Sensor", "Edge9f41b07", "Facilities", "Monitor", "INFRA", "sedge9f41b07", "sedge9f41b07@corp.example",
     None, "Onboarding follow-up for sedge9f41b07 (sedge9f41b07@corp.example)"),
    ("Daniel", "Weber", "IT", "Support Specialist", "QA", "dweber3", "dweber3@corp.example", None,
     "Onboarding follow-up for dweber3 (dweber3@corp.example)"),
    ("Mail", "Proc5521", "IT", "Service Account", "OPS", "mproc5521", "mproc5521@example.com", None,
     "Onboarding follow-up for mproc5521 (mproc5521@example.com)"),
    ("Grace", "Okafor", "Legal", "Counsel", "INFRA", "gokafor", "gokafor@example.com", None,
     "Onboarding follow-up for gokafor (gokafor@example.com)"),
    ("Report", "Cron8e0d4a6", "Analytics", "Scheduler", "KAN", "rcron8e0d4a6", "rcron8e0d4a6@example.org", None,
     "Onboarding follow-up for rcron8e0d4a6 (rcron8e0d4a6@example.org)"),
    ("Badge", "Lab2b90c3", "Facilities", "Reader", "CS", "blab2b90c3", "blab2b90c3@example.com", None,
     "Onboarding follow-up for blab2b90c3 (blab2b90c3@example.com)"),
    ("Elif", "Sahin", "Marketing", "Writer", "HELP", "esahin3", "esahin3@corp.example", None,
     "Onboarding follow-up for esahin3 (esahin3@corp.example)"),
    ("Sofia", "Rossi", "Marketing", "Designer", "PAY", "srossi", "srossi@example.org", None,
     "Onboarding follow-up for srossi (srossi@example.org)"),
    ("Pedro", "Lima", "Sales", "Representative", "ENG", "plima4", "plima4@example.com", None,
     "Onboarding follow-up for plima4 (plima4@example.com)"),
    ("Audit", "Daemon4e17f82", "Compliance", "Collector", "QA", "adaemon4e17f82", "adaemon4e17f82@corp.example",
     None, "Onboarding follow-up for adaemon4e17f82 (adaemon4e17f82@corp.example)"),
    ("Omar", "Haddad", "Finance", "Controller", "DEV", "ohaddad", "ohaddad@example.com", None,
     "Onboarding follow-up for ohaddad (ohaddad@example.com)"),
    ("Chloe", "Martin", "HR", "Coordinator", "ENG", "cmartin", "cmartin@example.org", None,
     "Onboarding follow-up for cmartin (cmartin@example.org)"),
    ("Gateway", "Link90455", "Networking", "Proxy", "INFRA", "glink90455", "glink90455@example.org", None,
     "Onboarding follow-up for glink90455 (glink90455@example.org)"),
    ("Archive", "Store77012", "Records", "Service Account", "APP", "astore77012", "astore77012@example.com", None,
     "Onboarding follow-up for astore77012 (astore77012@example.com)"),
    ("Ingest", "Pipe3a5b821", "Data", "Loader", "OPS", "ipipe3a5b821", "ipipe3a5b821@corp.example", None,
     "Onboarding follow-up for ipipe3a5b821 (ipipe3a5b821@corp.example)"),
    ("Hugo", "Lindqvist", "Support", "Agent", "CS", "hlindqvist", "hlindqvist@example.org", None,
     "Onboarding follow-up for hlindqvist (hlindqvist@example.org)"),
    ("Victor", "Dubois", "Legal", "Paralegal", "SEC", "vdubois2", "vdubois2@example.com", None,
     "Onboarding follow-up for vdubois2 (vdubois2@example.com)"),
    ("Render", "Farm2048", "Design", "Batch Worker", "DEV", "rfarm2048", "rfarm2048@example.org", None,
     "Onboarding follow-up for rfarm2048 (rfarm2048@example.org)"),
    ("Amara", "Eze", "Operations", "Planner", "KAN", "aeze5", "aeze5@corp.example", None,
     "Onboarding follow-up for aeze5 (aeze5@corp.example)"),
    ("Ledger", "Queue6d93e20", "Finance", "Integration", "PAY", "lqueue6d93e20", "lqueue6d93e20@example.com", None,
     "Onboarding follow-up for lqueue6d93e20 (lqueue6d93e20@example.com)"),
    ("Hana", "Kim", "Engineering", "Tester", "CS", "hkim", "hkim@corp.example",
     "set up their laptop", "Set up laptop for hkim (hkim@corp.example)"),
    ("Ivan", "Petrov", "Operations", "Planner", "KAN", "ipetrov", "ipetrov@example.org",
     "grant VPN access", "Grant VPN access for ipetrov (ipetrov@example.org)"),
    ("Lena", "Fischer", "Support", "Agent", "OPS", "lfischer", "lfischer@example.com",
     "order a headset", "Order a headset for lfischer (lfischer@example.com)"),
    ("Mateo", "Garcia", "Security", "Analyst", "HELP", "mgarcia4", "mgarcia4@corp.example",
     "enrol them in security training", "Security training enrolment for mgarcia4 (mgarcia4@corp.example)"),
    ("Kiosk", "Vm61207", "Facilities", "Kiosk", "APP", "kvm61207", "kvm61207@example.com",
     "register the kiosk device", "Register kiosk device for kvm61207 (kvm61207@example.com)"),
    ("Backup", "Task20931", "IT", "Service Account", "INFRA", "btask20931", "btask20931@corp.example",
     "grant backup storage access", "Grant backup storage access for btask20931 (btask20931@corp.example)"),
    ("Print", "Relay3306", "Facilities", "Service Account", "QA", "prelay3306", "prelay3306@example.org",
     "connect it to the print server", "Print server connection for prelay3306 (prelay3306@example.org)"),
    ("Lucia", "Ferrari", "Sales", "Manager", "ENG", "lferrari2", "lferrari2@example.com",
     "set up their phone", "Set up phone for lferrari2 (lferrari2@example.com)"),
    ("Metrics", "Pod1c6d39", "Platform", "Exporter", "SEC", "mpod1c6d39", "mpod1c6d39@corp.example",
     "allow read access to the metrics database",
     "Metrics database read access for mpod1c6d39 (mpod1c6d39@corp.example)"),
    ("Rosa", "Medina", "HR", "Advisor", "DEV", "rmedina", "rmedina@example.org",
     "replace their badge", "Replace badge for rmedina (rmedina@example.org)"),
)


def _continuation_records() -> list[JiraSftRecord]:
    """Ticket continuations after a verified account, rendered as production sends them.

    Within each request template every fifth record is held for validation, so
    validation covers every wording and id shape too.
    """
    records = []
    plain = [template for template in CONTINUATION_REQUEST_TEMPLATES if "{purpose}" not in template]
    plain_index = 0
    per_template: dict[str, int] = {}
    for index, (given, family, department, role, project, user_id, email, purpose, summary) in enumerate(
        _CONTINUATION_EXAMPLES
    ):
        if purpose:
            template = CONTINUATION_REQUEST_TEMPLATES[-1]
        else:
            template = plain[plain_index % len(plain)]
            plain_index += 1
        position = per_template.get(template, 0)
        per_template[template] = position + 1
        request = template.format(
            given=given, family=family, department=department, role=role, project=project, purpose=purpose,
        )
        arguments = {"project_key": project, "summary": summary}
        records.append(JiraSftRecord(
            record_id=f"jira-sft-continuation-{index:04d}-{_sha256_text(request)[:12]}",
            split="validation" if position % 5 == 0 else "train",
            tool_name="ticket_create",
            user_request=request,
            semantic_context=semantic_context_for("ticket_create", arguments),
            target_response=canonical_tool_call("ticket_create", arguments),
            verified_account={"user_id": user_id, "email": email},
            tags=["issue", "create", "mutation", "workflow-continuation", "reviewed-seed"],
        ))
    for record in records:
        _check_continuation_record(record)
    return records


def _check_continuation_record(record: JiraSftRecord) -> None:
    """A continuation teaches only ticket_create, with a summary naming the verified account."""
    account = record.verified_account
    if account is None:
        return
    if (set(account) != {"user_id", "email"}
            or not all(isinstance(value, str) and value and value == value.strip() for value in account.values())):
        raise ValueError(f"Jira continuation {record.record_id} needs an exact user_id and email.")
    calls = parse_tool_calls(record.target_response)
    summary = calls[0]["arguments"].get("summary") if len(calls) == 1 else None
    if (record.tool_name != "ticket_create" or not isinstance(summary, str)
            or not (account["user_id"] in summary or account["email"] in summary)):
        raise ValueError(f"Jira continuation {record.record_id} must create a ticket naming its verified account.")


def _register_runtime_contract(prompt_layout: str) -> None:
    if prompt_layout == "runtime_task":
        # Signature registration is metadata-only: no provider/server starts.
        from mcp_server import register_capability_signatures
        register_capability_signatures()


def runtime_task_messages(
    agent: AgentDefinition, record: JiraSftRecord, prompt_profile: str,
) -> list[dict[str, str]]:
    """Reproduce one isolated local worker task, without prior conversation."""
    if record.semantic_context.get("allowed_tools") != [record.tool_name]:
        raise ValueError("Runtime-task corpus requires exactly its declared capability.")
    catalog = build_agent_capability_catalog(
        agent, include_arguments=True, allowed_tools=[record.tool_name],
    )
    if len(catalog) != 1 or catalog[0]["name"] != record.tool_name:
        raise ValueError("Runtime-task capability is not owned by the Jira specialist.")
    return build_runtime_messages(
        system_prompt=build_worker_system_prompt(
            agent, capability_catalog=catalog, prompt_profile=prompt_profile,
        ),
        user_request=record.user_request, semantic_context=record.semantic_context,
        current_request_last=True, verified_account=record.verified_account,
    )


def _runtime_task_record(
    agent: AgentDefinition, record: JiraSftRecord, prompt_profile: str,
) -> JiraSftRecord:
    catalog = build_agent_capability_catalog(
        agent, include_arguments=True, allowed_tools=[record.tool_name],
    )
    if len(catalog) != 1:
        raise ValueError("Runtime-task capability must have one trusted descriptor.")
    grounded = set(catalog[0]["grounded_arguments"])
    context = dict(record.semantic_context)
    # Derived proposal content is not a Hub-authorized target binding.
    for field in ("allowed_arguments", "forbidden_arguments"):
        context[field] = {
            name: value for name, value in context.get(field, {}).items()
            if name in grounded
        }
    record = record.model_copy(update={"semantic_context": context})
    return record.model_copy(update={
        "prompt_messages": runtime_task_messages(agent, record, prompt_profile),
    })


def build_jira_sft_corpus(
    *,
    project_root: Path,
    output_directory: Path,
    base_model_key: str = DEFAULT_BASE_MODEL_KEY,
    worker_prompt_profile: str | None = None,
    prompt_layout: Literal["legacy", "runtime_task"] = "legacy",
    force: bool = False,
) -> JiraSftCorpusManifest:
    if prompt_layout not in {"legacy", "runtime_task"}:
        raise ValueError("Unsupported Jira prompt layout.")
    _register_runtime_contract(prompt_layout)
    project_root = (
        project_root
        .expanduser()
        .resolve()
    )

    output_directory = (
        output_directory
        .expanduser()
        .resolve()
    )

    if output_directory.exists():
        if not force:
            raise FileExistsError(
                "Jira SFT corpus directory already exists: "
                f"{output_directory}. Pass force=True to rebuild."
            )

        shutil.rmtree(
            output_directory
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    settings = Settings()
    profile = settings.require_model_profile(
        base_model_key
    )
    prompt_profile = ModelProfileSettings(
        backend=profile.backend,
        worker_prompt_profile=(
            profile.worker_prompt_profile
            if worker_prompt_profile is None
            else worker_prompt_profile
        ),
    ).worker_prompt_profile

    agent = load_agent_definition(
        project_root
        / "subagents"
        / "agents"
        / "jira-specialist.md"
    )

    capabilities = build_agent_capability_catalog(
        agent,
        include_arguments=True,
    )

    system_prompt = build_worker_system_prompt(
        agent,
        capability_catalog=capabilities,
        prompt_profile=(
            prompt_profile
        ),
    )

    records = _build_records(
        project_root=(
            project_root
        )
    )
    if prompt_layout == "runtime_task":
        records = [_runtime_task_record(agent, record, prompt_profile) for record in records]
    else:
        # The legacy layout predates the live worker message order and cannot
        # carry a continuation's verified-account message.
        records = [record for record in records if record.verified_account is None]

    train = [
        record
        for record
        in records
        if record.split == "train"
    ]

    validation = [
        record
        for record
        in records
        if record.split == "validation"
    ]

    train_path = (
        output_directory
        / "train.jsonl"
    )

    validation_path = (
        output_directory
        / "validation.jsonl"
    )

    def write_jsonl(
        path: Path,
        items: list[
            JiraSftRecord
        ],
    ) -> None:
        with path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            for item in items:
                handle.write(
                    _canonical_json(
                        item.model_dump(
                            mode="json",
                            by_alias=True,
                        )
                    )
                    + "\n"
                )

    write_jsonl(
        train_path,
        train,
    )

    write_jsonl(
        validation_path,
        validation,
    )

    per_tool_counts: dict[
        str,
        int,
    ] = {}

    for record in records:
        per_tool_counts[
            record.tool_name
        ] = (
            per_tool_counts.get(
                record.tool_name,
                0,
            )
            + 1
        )

    manifest = JiraSftCorpusManifest(
        prompt_layout=prompt_layout,
        created_at=_utc_now(),
        source=(
            "catalog-derived-jira-capability-language-v2"
        ),
        agent_name=agent.name,
        base_model_key=base_model_key,
        worker_prompt_profile=(
            prompt_profile
        ),
        record_count=len(
            records
        ),
        train_count=len(
            train
        ),
        validation_count=len(
            validation
        ),
        per_tool_counts=(
            dict(
                sorted(
                    per_tool_counts.items()
                )
            )
        ),
        train_sha256=_sha256_file(
            train_path
        ),
        validation_sha256=_sha256_file(
            validation_path
        ),
        agent_definition_sha256=(
            _agent_identity_hash(
                agent
            )
        ),
        capability_catalog_sha256=(
            _sha256_text(
                _canonical_json(
                    capabilities
                )
            )
        ),
        system_prompt_sha256=(
            _sha256_text(
                system_prompt
            )
        ),
    )

    (
        output_directory
        / "manifest.json"
    ).write_text(
        json.dumps(
            manifest.model_dump(
                mode="json",
                by_alias=True,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return manifest


def _load_jsonl(
    path: Path,
) -> list[
    JiraSftRecord
]:
    records: list[
        JiraSftRecord
    ] = []

    for (
        line_number,
        line,
    ) in enumerate(
        path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            records.append(
                JiraSftRecord.model_validate(
                    json.loads(
                        line
                    )
                )
            )

        except Exception as exc:
            raise ValueError(
                f"Invalid Jira SFT record in {path} "
                f"line {line_number}: {exc}"
            ) from exc

    return records


def verify_jira_sft_corpus(
    *,
    project_root: Path,
    corpus_directory: Path,
    worker_prompt_profile: str | None = None,
) -> tuple[
    JiraSftCorpusManifest,
    list[JiraSftRecord],
    list[JiraSftRecord],
    str,
]:
    project_root = (
        project_root
        .expanduser()
        .resolve()
    )

    corpus_directory = (
        corpus_directory
        .expanduser()
        .resolve()
    )

    manifest_path = (
        corpus_directory
        / "manifest.json"
    )

    train_path = (
        corpus_directory
        / "train.jsonl"
    )

    validation_path = (
        corpus_directory
        / "validation.jsonl"
    )

    for path in [
        manifest_path,
        train_path,
        validation_path,
    ]:
        if not path.is_file():
            raise ValueError(
                "Jira SFT corpus is incomplete: "
                f"missing {path}"
            )

    manifest = JiraSftCorpusManifest.model_validate(
        json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )
    )
    _register_runtime_contract(manifest.prompt_layout)

    if _sha256_file(
        train_path
    ) != manifest.train_sha256:
        raise ValueError(
            "Jira SFT train split hash mismatch."
        )

    if _sha256_file(
        validation_path
    ) != manifest.validation_sha256:
        raise ValueError(
            "Jira SFT validation split hash mismatch."
        )

    settings = Settings()
    profile = settings.require_model_profile(
        manifest.base_model_key
    )
    expected_prompt_profile = ModelProfileSettings(
        backend=profile.backend,
        worker_prompt_profile=(
            profile.worker_prompt_profile
            if worker_prompt_profile is None
            else worker_prompt_profile
        ),
    ).worker_prompt_profile

    if (
        expected_prompt_profile
        != manifest.worker_prompt_profile
    ):
        raise ValueError(
            "Jira SFT corpus prompt profile no longer matches "
            "the configured base model profile or explicit candidate profile."
        )

    agent = load_agent_definition(
        project_root
        / "subagents"
        / "agents"
        / "jira-specialist.md"
    )

    capabilities = build_agent_capability_catalog(
        agent,
        include_arguments=True,
    )

    system_prompt = build_worker_system_prompt(
        agent,
        capability_catalog=capabilities,
        prompt_profile=(
            expected_prompt_profile
        ),
    )

    if (
        _agent_identity_hash(
            agent
        )
        != manifest.agent_definition_sha256
    ):
        raise ValueError(
            "Jira specialist definition changed after corpus build. "
            "Rebuild the corpus before training."
        )

    if (
        _sha256_text(
            _canonical_json(
                capabilities
            )
        )
        != manifest.capability_catalog_sha256
    ):
        raise ValueError(
            "Jira capability catalog changed after corpus build. "
            "Rebuild the corpus before training."
        )

    if (
        _sha256_text(
            system_prompt
        )
        != manifest.system_prompt_sha256
    ):
        raise ValueError(
            "Jira system prompt changed after corpus build. "
            "Rebuild the corpus before training."
        )

    train = _load_jsonl(
        train_path
    )
    validation = _load_jsonl(
        validation_path
    )

    if len(train) != manifest.train_count:
        raise ValueError(
            "Jira SFT train record count mismatch."
        )

    if len(validation) != manifest.validation_count:
        raise ValueError(
            "Jira SFT validation record count mismatch."
        )

    for record in [
        *train,
        *validation,
    ]:
        if manifest.prompt_layout == "runtime_task":
            expected_record = _runtime_task_record(agent, record, expected_prompt_profile)
            if (record.semantic_context != expected_record.semantic_context
                    or record.prompt_messages != expected_record.prompt_messages):
                raise ValueError("Jira runtime-task messages changed after corpus build.")
        elif record.prompt_messages is not None:
            raise ValueError("Legacy Jira corpus cannot override its verified prompt.")
        elif record.verified_account is not None:
            raise ValueError("A verified-account continuation needs the runtime-task layout.")
        _check_continuation_record(record)
        calls = parse_tool_calls(
            record.target_response
        )

        if len(calls) != 1:
            raise ValueError(
                "Jira SFT target must contain exactly one call: "
                f"{record.record_id}"
            )

        call = calls[0]

        if (
            call.get(
                "name"
            )
            != record.tool_name
        ):
            raise ValueError(
                "Jira SFT target tool mismatch: "
                f"{record.record_id}"
            )

        if record.tool_name not in (
            record.semantic_context.get(
                "allowed_tools",
                [],
            )
        ):
            raise ValueError(
                "Jira SFT semantic context tool mismatch: "
                f"{record.record_id}"
            )

    return (
        manifest,
        train,
        validation,
        system_prompt,
    )


# ============================================================
# TRAINING VERSION / ARTIFACT HELPERS
# ============================================================


def installed_training_versions(
) -> dict[
    str,
    str,
]:
    from importlib import metadata

    result: dict[
        str,
        str,
    ] = {}

    for package in [
        "torch",
        "transformers",
        "accelerate",
        "peft",
        "trl",
        "bitsandbytes",
        "datasets",
    ]:
        try:
            result[
                package
            ] = metadata.version(
                package
            )
        except metadata.PackageNotFoundError:
            result[
                package
            ] = "missing"

    return result


def assert_training_versions(
) -> dict[
    str,
    str,
]:
    versions = installed_training_versions()
    problems: list[str] = []

    for (
        package,
        expected,
    ) in EXPECTED_TRAINING_VERSIONS.items():
        observed = versions.get(
            package,
            "missing",
        )

        if observed != expected:
            problems.append(
                f"{package}: expected {expected}, found {observed}"
            )

    if problems:
        raise ValueError(
            "Training API version lock failed: "
            + "; ".join(
                problems
            )
        )

    return versions


def fingerprint_directory(
    directory: Path,
) -> str:
    resolved = (
        directory
        .expanduser()
        .resolve()
    )

    if not resolved.is_dir():
        raise ValueError(
            "Artifact directory does not exist: "
            f"{resolved}"
        )

    files = [
        path
        for path
        in sorted(
            resolved.rglob(
                "*"
            )
        )
        if path.is_file()
    ]

    if not files:
        raise ValueError(
            "Artifact directory is empty: "
            f"{resolved}"
        )

    payload = [
        {
            "path": str(
                path.relative_to(
                    resolved
                )
            ),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(
                path
            ),
        }
        for path in files
    ]

    return _sha256_text(
        _canonical_json(
            payload
        )
    )


# ============================================================
# TOKENIZATION / COLLATION
# ============================================================


def _render_prompt(
    tokenizer,
    messages: list[
        dict[
            str,
            str,
        ]
    ],
) -> str:
    chat_template = getattr(
        tokenizer,
        "chat_template",
        None,
    )

    if (
        isinstance(
            chat_template,
            str,
        )
        and chat_template.strip()
    ):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return render_hf_causal_fallback_prompt(
        messages
    )


def _tokenize_record(
    *,
    tokenizer,
    system_prompt: str,
    record: JiraSftRecord,
    max_length: int,
) -> dict[
    str,
    list[int],
]:
    messages = record.prompt_messages or build_runtime_messages(
        system_prompt=system_prompt,
        user_request=record.user_request,
        semantic_context=record.semantic_context,
        verified_account=record.verified_account,
    )

    from learning.training.hub_hybrid_qlora import (
        _encode_prompt_and_completion,
        _has_native_chat_template,
    )
    completion = record.target_response
    eos = tokenizer.eos_token or ""
    if not _has_native_chat_template(tokenizer) and eos and not completion.endswith(eos):
        completion += eos
    # Reuse the strict native/fallback boundary. Never discard the system
    # prefix or partial targets to make an oversized example trainable.
    try:
        input_ids, attention_mask, labels = _encode_prompt_and_completion(
            tokenizer=tokenizer, prompt_messages=messages,
            completion=completion, max_length=max_length,
        )
    except ValueError as exc:
        raise ValueError(f"Jira SFT record {record.record_id}: {exc}") from exc
    return {
        "input_ids": input_ids[0].tolist(),
        "attention_mask": attention_mask[0].tolist(),
        "labels": labels[0].tolist(),
    }


class JiraSftTokenDataset:
    def __init__(
        self,
        items: list[
            dict[
                str,
                list[int],
            ]
        ],
    ) -> None:
        self.items = items

    def __len__(
        self,
    ) -> int:
        return len(
            self.items
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[
        str,
        list[int],
    ]:
        return self.items[
            index
        ]


class JiraSftCollator:
    def __init__(
        self,
        *,
        pad_token_id: int,
    ) -> None:
        self.pad_token_id = (
            pad_token_id
        )

    def __call__(
        self,
        features: list[
            dict[
                str,
                list[int],
            ]
        ],
    ):
        import torch

        max_length = max(
            len(
                item[
                    "input_ids"
                ]
            )
            for item in features
        )

        batch_input_ids: list[
            list[int]
        ] = []
        batch_attention: list[
            list[int]
        ] = []
        batch_labels: list[
            list[int]
        ] = []

        for item in features:
            padding = (
                max_length
                - len(
                    item[
                        "input_ids"
                    ]
                )
            )

            batch_input_ids.append(
                item[
                    "input_ids"
                ]
                + [
                    self.pad_token_id
                ] * padding
            )

            batch_attention.append(
                item[
                    "attention_mask"
                ]
                + [
                    0
                ] * padding
            )

            batch_labels.append(
                item[
                    "labels"
                ]
                + [
                    -100
                ] * padding
            )

        return {
            "input_ids": torch.tensor(
                batch_input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                batch_attention,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                batch_labels,
                dtype=torch.long,
            ),
        }


# ============================================================
# TRAINER ARGUMENT COMPATIBILITY
# ============================================================


def build_jira_training_arguments(
    *,
    output_dir: Path,
    max_steps: int,
    learning_rate: float,
    gradient_accumulation_steps: int,
    seed: int,
    warmup_ratio: float = 0.03,
    guard_arguments: dict[str, Any] | None = None,
):
    """Build the exact Trainer configuration without loading weights.

    The locked Transformers runtime currently exposes ``warmup_steps``
    rather than accepting ``warmup_ratio`` in the TrainingArguments
    constructor. Keep the intended 3% warmup policy by converting it
    to an explicit
    step count here. Centralizing this construction also lets preflight
    instantiate the real arguments object before any checkpoint is loaded.
    ``guard_arguments`` (see ``jira_validation_guard_arguments``) adds
    validation, checkpointing and best-adapter selection.
    """
    guard_arguments = guard_arguments or {"save_strategy": "no"}

    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError(
            "max_steps must be a positive integer."
        )

    if not (0.0 <= warmup_ratio < 1.0):
        raise ValueError(
            "warmup_ratio must be in [0, 1)."
        )

    from transformers import (
        TrainingArguments,
    )

    warmup_steps = (
        0
        if warmup_ratio == 0.0
        else max(
            1,
            int(
                math.ceil(
                    max_steps
                    * warmup_ratio
                )
            ),
        )
    )

    return TrainingArguments(
        output_dir=str(
            output_dir
            .expanduser()
            .resolve()
        ),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=(
            gradient_accumulation_steps
        ),
        max_steps=max_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.0,
        logging_steps=1,
        report_to="none",
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },
        optim="paged_adamw_8bit",
        seed=seed,
        remove_unused_columns=False,
        **guard_arguments,
    )


# ============================================================
# VALIDATION GUARD: BEST CHECKPOINT, EARLY STOP, RESUME
# ============================================================


def jira_validation_guard_arguments(
    checkpoint_interval: int,
    *,
    baseline_evaluation: bool = True,
) -> dict[str, Any]:
    """TrainingArguments that keep the validation-best adapter.

    Validation loss is measured every ``checkpoint_interval`` optimizer steps
    and once before training (the baseline). Checkpoints are saved at the same
    interval, so an interrupted run can resume. Only the newest and the best
    checkpoints are kept. Training ends on the best adapter.
    """
    if type(checkpoint_interval) is not int or checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be a positive integer.")
    return {
        "eval_strategy": "steps",
        "eval_steps": checkpoint_interval,
        "save_strategy": "steps",
        "save_steps": checkpoint_interval,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "eval_on_start": baseline_evaluation,
        "per_device_eval_batch_size": 1,
        "prediction_loss_only": True,
        "restore_callback_states_from_checkpoint": True,
    }


def build_jira_trainer(
    *,
    model,
    training_args,
    train_items: list[dict[str, list[int]]],
    validation_items: list[dict[str, list[int]]],
    pad_token_id: int,
    early_stopping_patience: int,
):
    """Trainer with target-only loss, validation guard and early stopping."""
    if not train_items:
        raise ValueError("Jira SFT training split is empty.")
    if not validation_items:
        raise ValueError("Jira SFT requires a non-empty validation split.")
    if type(early_stopping_patience) is not int or early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be a positive integer.")

    from transformers import EarlyStoppingCallback, Trainer

    from learning.training.causal_loss import tail_only_trainer_class

    trainer_class = tail_only_trainer_class(Trainer)
    return trainer_class(
        model=model,
        args=training_args,
        train_dataset=JiraSftTokenDataset(train_items),
        eval_dataset=JiraSftTokenDataset(validation_items),
        data_collator=JiraSftCollator(pad_token_id=pad_token_id),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )


def summarize_validation(state, *, max_steps: int) -> JiraValidationSummary:
    """Validation record of a finished Trainer run, from its own state."""
    evaluations = [entry for entry in state.log_history if "eval_loss" in entry]
    if not evaluations or state.best_metric is None or state.best_model_checkpoint is None:
        raise ValueError("Jira SFT run recorded no validation checkpoint.")
    baseline = next((entry["eval_loss"] for entry in evaluations if entry["step"] == 0), None)
    best_step = int(Path(state.best_model_checkpoint).name.rsplit("-", 1)[1])
    return JiraValidationSummary(
        optimizer_steps=state.global_step,
        max_steps=max_steps,
        early_stopped=state.global_step < max_steps,
        evaluated_steps=[entry["step"] for entry in evaluations],
        baseline_eval_loss=baseline,
        best_eval_loss=float(state.best_metric),
        best_checkpoint_step=best_step,
    )


def resumable_checkpoint(trainer_directory: Path) -> str | None:
    """The newest saved checkpoint of an interrupted run, if any."""
    from transformers.trainer_utils import get_last_checkpoint

    directory = Path(trainer_directory)
    return get_last_checkpoint(str(directory)) if directory.is_dir() else None


def jira_run_identity(
    *,
    corpus_manifest: JiraSftCorpusManifest,
    base_model_identity: dict[str, str],
    max_steps: int,
    max_length: int,
    learning_rate: float = 2e-4,
    gradient_accumulation_steps: int = 8,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    seed: int = 42,
    checkpoint_interval: int,
    early_stopping_patience: int,
) -> dict[str, Any]:
    """Settings a resumed run must share with the run that saved its checkpoints."""
    return {
        "schema": "jira-specialist-sft-run.v2",
        "corpus_train_sha256": corpus_manifest.train_sha256,
        "corpus_validation_sha256": corpus_manifest.validation_sha256,
        "system_prompt_sha256": corpus_manifest.system_prompt_sha256,
        "prompt_layout": corpus_manifest.prompt_layout,
        "base_model_key": corpus_manifest.base_model_key,
        "base_model_identity": base_model_identity,
        "max_steps": max_steps,
        "max_length": max_length,
        "learning_rate": learning_rate,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "seed": seed,
        "checkpoint_interval": checkpoint_interval,
        "early_stopping_patience": early_stopping_patience,
    }


def _resume_run(run_directory: Path, identity: dict[str, Any]) -> Path:
    run_directory = run_directory.expanduser().resolve()
    config_path = run_directory / "run-config.json"
    if not config_path.is_file():
        raise ValueError(f"Resume run has no run-config.json: {run_directory}")
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(saved, dict):
        raise ValueError("Resume run-config.json must contain an object.")
    if saved.get("schema") != "jira-specialist-sft-run.v2" or not saved.get("base_model_identity"):
        raise ValueError(
            "Resume run has no recorded base-model/tokenizer content identity. "
            "Preserve this legacy run for reconciliation; its original base "
            "cannot be established from the current model files. Start a new "
            "run instead of rewriting its run-config.json."
        )
    if saved != identity:
        changed = sorted(key for key in set(saved) | set(identity) if saved.get(key) != identity.get(key))
        raise ValueError("Resume configuration does not match the interrupted run: " + ", ".join(changed))
    if (run_directory / "training-manifest.json").exists():
        raise ValueError("Resume run already has a completed training manifest.")
    return run_directory


def jira_base_model_identity(base_model_path: Path) -> dict[str, str]:
    """Inspect current local model and tokenizer bytes without loading weights.

    Runtime fingerprints are process snapshots by default. Training must bypass
    that cache so replacing bytes at the same path cannot reuse a saved identity.
    """
    from learning.evidence.execution_provenance import fingerprint_runtime_model_artifact

    resolved = base_model_path.expanduser().resolve()
    fingerprint = fingerprint_runtime_model_artifact(resolved, use_cache=False)
    return {
        "resolved_path": str(resolved),
        "content_sha256": fingerprint.content_sha256,
        "weights_sha256": fingerprint.weights_sha256,
        "tokenizer_sha256": fingerprint.tokenizer_sha256,
        "config_sha256": fingerprint.config_sha256,
    }


def validate_jira_training_options(
    *, max_steps: int, max_length: int, learning_rate: float,
    gradient_accumulation_steps: int, lora_r: int, lora_alpha: int,
    lora_dropout: float, seed: int, checkpoint_interval: int,
    early_stopping_patience: int,
) -> None:
    """Reject invalid recipes before corpus I/O, CUDA checks or model loading."""
    integers = {
        "max_steps": max_steps, "max_length": max_length,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lora_r": lora_r, "lora_alpha": lora_alpha,
        "checkpoint_interval": checkpoint_interval,
        "early_stopping_patience": early_stopping_patience,
    }
    for name, value in integers.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if max_steps > 2000:
        raise ValueError("Jira MVP SFT refuses max_steps > 2000.")
    if max_length < 256:
        raise ValueError("max_length must be at least 256.")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    for name, value in (("learning_rate", learning_rate), ("lora_dropout", lora_dropout)):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if learning_rate < 0:
        raise ValueError("learning_rate must be non-negative.")
    if not 0 <= lora_dropout < 1:
        raise ValueError("lora_dropout must be in [0, 1).")


def validate_jira_resume_checkpoint(
    run_directory: Path, *, max_steps: int, base_model_path: Path,
    early_stopping_patience: int,
) -> str:
    """Check required checkpoint files and plain metadata before hardware use.

    Never fall back to an older checkpoint after finding an incomplete newest
    one. Binary state is loaded later by Trainer; this preflight does not unpickle
    files or pretend that nonempty files prove their numerical integrity.
    """
    trainer_directory = (run_directory / "trainer-work").resolve()
    checkpoint = resumable_checkpoint(trainer_directory)
    if checkpoint is None:
        raise ValueError(f"Resume run has no saved checkpoint: {trainer_directory}")
    checkpoint_path = Path(checkpoint).resolve()
    required = (
        "adapter_config.json", "adapter_model.safetensors", "trainer_state.json",
        "optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin",
    )
    for name in required:
        path = checkpoint_path / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Resume checkpoint is incomplete: {checkpoint_path.name}/{name}")
    state = json.loads((checkpoint_path / "trainer_state.json").read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError("Resume trainer_state.json must contain an object.")
    step = state.get("global_step")
    if type(step) is not int or step < 1 or step > max_steps or checkpoint_path.name != f"checkpoint-{step}":
        raise ValueError("Resume checkpoint global_step does not match its directory or step budget.")
    if state.get("max_steps") != max_steps:
        raise ValueError("Resume checkpoint max_steps does not match its run configuration.")
    callbacks = state.get("stateful_callbacks")
    callback = callbacks.get("EarlyStoppingCallback") if isinstance(callbacks, dict) else None
    callback_args = callback.get("args") if isinstance(callback, dict) else None
    callback_state = callback.get("attributes") if isinstance(callback, dict) else None
    counter = callback_state.get("early_stopping_patience_counter") if isinstance(callback_state, dict) else None
    if (
        not isinstance(callback_args, dict)
        or callback_args.get("early_stopping_patience") != early_stopping_patience
        or callback_args.get("early_stopping_threshold") != 0.0
        or type(counter) is not int or not 0 <= counter <= early_stopping_patience
    ):
        raise ValueError("Resume checkpoint is missing matching early-stopping state.")
    if step == max_steps or counter == early_stopping_patience:
        raise ValueError(
            "Resume checkpoint already finished its optimization budget or early stop. "
            "Reconcile its unfinished artifact export instead of repeating optimizer steps."
        )
    best = state.get("best_model_checkpoint")
    metric = state.get("best_metric")
    if not isinstance(best, str) or not best:
        raise ValueError("Resume checkpoint has no recorded validation-best adapter.")
    if isinstance(metric, bool) or not isinstance(metric, Real) or not math.isfinite(metric):
        raise ValueError("Resume checkpoint has no finite validation-best loss.")
    best_path = Path(best).expanduser().resolve()
    best_step = state.get("best_global_step")
    if (
        best_path.parent != trainer_directory or type(best_step) is not int
        or not 1 <= best_step <= step or best_path.name != f"checkpoint-{best_step}"
    ):
        raise ValueError("Resume validation-best checkpoint does not belong to this run and step range.")
    for path in {checkpoint_path, best_path}:
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            artifact = path / name
            if not artifact.is_file() or artifact.stat().st_size == 0:
                raise ValueError(f"Resume checkpoint is incomplete: {path.name}/{name}")
        adapter_config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
        base_name = adapter_config.get("base_model_name_or_path") if isinstance(adapter_config, dict) else None
        if not isinstance(base_name, str) or Path(base_name).expanduser().resolve() != base_model_path:
            raise ValueError("Resume adapter base model does not match the bound local base model.")
    return str(checkpoint_path)


# ============================================================
# REAL BOUNDED SFT + QLORA TRAINING
# ============================================================


def train_jira_sft(
    *,
    project_root: Path,
    corpus_directory: Path,
    output_root: Path,
    allow_training: bool,
    max_steps: int,
    max_length: int = 1024,
    learning_rate: float = 2e-4,
    gradient_accumulation_steps: int = 8,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    seed: int = 42,
    worker_prompt_profile: str | None = None,
    checkpoint_interval: int = 10,
    early_stopping_patience: int = 3,
    resume_run_directory: Path | None = None,
) -> JiraSftTrainingManifest:
    if not allow_training:
        raise PermissionError(
            "Real Jira SFT requires explicit allow_training=True."
        )

    validate_jira_training_options(
        max_steps=max_steps, max_length=max_length, learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        seed=seed, checkpoint_interval=checkpoint_interval,
        early_stopping_patience=early_stopping_patience,
    )

    (
        corpus_manifest,
        train_records,
        validation_records,
        system_prompt,
    ) = verify_jira_sft_corpus(
        project_root=project_root,
        corpus_directory=corpus_directory,
        worker_prompt_profile=worker_prompt_profile,
    )

    if not train_records:
        raise ValueError("Jira SFT training split is empty.")
    if not validation_records:
        raise ValueError("Jira SFT requires a non-empty validation split.")

    settings = Settings()
    profile = settings.require_model_profile(
        corpus_manifest.base_model_key
    )

    if profile.backend != "hf-causal":
        raise ValueError(
            "Jira SFT currently requires an hf-causal base profile."
        )

    if profile.model_path is None:
        raise ValueError(
            "Jira SFT base profile has no model_path."
        )

    base_model_path = (
        profile.model_path
        .expanduser()
        .resolve()
    )

    if not base_model_path.is_dir():
        raise ValueError(
            "Jira SFT base model path does not exist: "
            f"{base_model_path}"
        )

    base_model_identity = jira_base_model_identity(base_model_path)
    run_identity = jira_run_identity(
        corpus_manifest=corpus_manifest, base_model_identity=base_model_identity,
        max_steps=max_steps, max_length=max_length,
        learning_rate=learning_rate, gradient_accumulation_steps=gradient_accumulation_steps,
        lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, seed=seed,
        checkpoint_interval=checkpoint_interval, early_stopping_patience=early_stopping_patience,
    )
    resumed_run = (
        _resume_run(resume_run_directory, run_identity)
        if resume_run_directory is not None else None
    )
    resume_checkpoint = (
        validate_jira_resume_checkpoint(
            resumed_run, max_steps=max_steps, base_model_path=base_model_path,
            early_stopping_patience=early_stopping_patience,
        )
        if resumed_run is not None else None
    )
    guard_arguments = jira_validation_guard_arguments(
        checkpoint_interval, baseline_evaluation=resumed_run is None,
    )

    versions = assert_training_versions()

    import torch

    if not torch.cuda.is_available():
        raise ValueError(
            "CUDA is required for Jira BLOOMZ QLoRA training."
        )

    if not torch.cuda.is_bf16_supported():
        raise ValueError(
            "The configured Jira training recipe requires BF16 support."
        )

    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(
            base_model_path
        ),
        local_files_only=True,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "BLOOMZ tokenizer has neither pad nor EOS token."
            )

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenized_train = [
        _tokenize_record(
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            record=record,
            max_length=max_length,
        )
        for record in train_records
    ]

    tokenized_validation = [
        _tokenize_record(
            tokenizer=tokenizer, system_prompt=system_prompt,
            record=record, max_length=max_length,
        )
        for record in validation_records
    ]

    if not tokenized_train:
        raise ValueError(
            "Jira SFT training split is empty."
        )

    dtype = torch.bfloat16

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(
            base_model_path
        ),
        local_files_only=True,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=dtype,
    )

    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    model = get_peft_model(
        model,
        peft_config,
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    run_directory = resumed_run or (
        output_root
        .expanduser()
        .resolve()
        / "runs"
        / _run_stamp()
    )

    adapter_directory = (
        run_directory
        / "adapter"
    )

    merged_directory = (
        run_directory
        / "merged"
    )

    trainer_directory = (
        run_directory
        / "trainer-work"
    )

    if resumed_run is None:
        run_directory.mkdir(
            parents=True,
            exist_ok=False,
        )
        (run_directory / "run-config.json").write_text(
            json.dumps(run_identity, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
    training_args = build_jira_training_arguments(
        output_dir=trainer_directory,
        max_steps=max_steps,
        learning_rate=learning_rate,
        gradient_accumulation_steps=(
            gradient_accumulation_steps
        ),
        seed=seed,
        guard_arguments=guard_arguments,
    )

    trainer = build_jira_trainer(
        model=model, training_args=training_args,
        train_items=tokenized_train, validation_items=tokenized_validation,
        pad_token_id=tokenizer.pad_token_id,
        early_stopping_patience=early_stopping_patience,
    )

    # Ends on the validation-best adapter (load_best_model_at_end).
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    validation_summary = summarize_validation(trainer.state, max_steps=max_steps)

    model.save_pretrained(
        adapter_directory,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(
        adapter_directory
    )

    adapter_sha256 = fingerprint_directory(
        adapter_directory
    )

    # --------------------------------------------------------
    # RELEASE TRAINING GPU STATE BEFORE CPU MERGE
    # --------------------------------------------------------

    del trainer
    del model

    gc.collect()
    torch.cuda.empty_cache()

    # --------------------------------------------------------
    # MERGE ADAPTER INTO A STANDALONE CHECKPOINT
    #
    # Runtime continues to consume an ordinary hf-causal model path.
    # This avoids introducing a special adapter loader into production
    # solely for Jira and keeps execution provenance checkpoint-based.
    # --------------------------------------------------------

    from peft import (
        PeftModel,
    )

    merge_base = AutoModelForCausalLM.from_pretrained(
        str(
            base_model_path
        ),
        local_files_only=True,
        dtype=torch.float16,
        device_map={
            "": "cpu"
        },
        low_cpu_mem_usage=True,
    )

    merge_model = PeftModel.from_pretrained(
        merge_base,
        str(
            adapter_directory
        ),
        is_trainable=False,
    )

    merged = merge_model.merge_and_unload(
        safe_merge=True
    )

    if hasattr(
        merged,
        "config",
    ):
        merged.config.use_cache = True

    merged.save_pretrained(
        merged_directory,
        safe_serialization=True,
        max_shard_size="256MB",
    )

    tokenizer.save_pretrained(
        merged_directory
    )

    del merged
    del merge_model
    del merge_base
    del tokenizer

    gc.collect()
    torch.cuda.empty_cache()

    merged_sha256 = fingerprint_directory(
        merged_directory
    )

    manifest = JiraSftTrainingManifest(
        created_at=_utc_now(),
        base_model_key=(
            corpus_manifest.base_model_key
        ),
        base_model_path=str(
            base_model_path
        ),
        base_model_identity=base_model_identity,
        output_directory=str(
            run_directory
        ),
        adapter_directory=str(
            adapter_directory
        ),
        merged_model_directory=str(
            merged_directory
        ),
        corpus_directory=str(
            corpus_directory
            .expanduser()
            .resolve()
        ),
        corpus_train_sha256=(
            corpus_manifest.train_sha256
        ),
        corpus_validation_sha256=(
            corpus_manifest.validation_sha256
        ),
        worker_prompt_profile=corpus_manifest.worker_prompt_profile,
        system_prompt_sha256=corpus_manifest.system_prompt_sha256,
        prompt_layout=corpus_manifest.prompt_layout,
        max_steps=max_steps,
        learning_rate=learning_rate,
        max_length=max_length,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        validation=validation_summary,
        trainable_parameters=(
            trainable_parameters
        ),
        total_parameters=(
            total_parameters
        ),
        trainable_ratio=(
            trainable_parameters
            / total_parameters
            if total_parameters
            else 0.0
        ),
        adapter_sha256=(
            adapter_sha256
        ),
        merged_model_sha256=(
            merged_sha256
        ),
        training_versions=versions,
    )

    manifest_path = (
        run_directory
        / "training-manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest.model_dump(
                mode="json",
                by_alias=True,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    latest_pointer = (
        output_root
        .expanduser()
        .resolve()
        / "latest-training.json"
    )

    latest_pointer.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    latest_pointer.write_text(
        json.dumps(
            {
                "schema": (
                    "jira-specialist-sft-latest.v1"
                ),
                "training_manifest": str(
                    manifest_path
                ),
                "merged_model_directory": str(
                    merged_directory
                ),
                "merged_model_sha256": (
                    merged_sha256
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return manifest


def register_trained_candidate_profile(
    *,
    env_path: Path,
    training_manifest_path: Path,
    model_key: str = DEFAULT_CANDIDATE_MODEL_KEY,
) -> dict[
    str,
    Any,
]:
    serving_agents = sorted(
        agent.name
        for agent in load_agent_directory(Settings().agents_dir)
        if agent.model == model_key
    )
    if serving_agents:
        raise ValueError(
            f"Model profile '{model_key}' is served by {', '.join(serving_agents)}; "
            "register the candidate under a separate key and promote it only after evaluation."
        )

    env_path = (
        env_path
        .expanduser()
        .resolve()
    )

    training_manifest_path = (
        training_manifest_path
        .expanduser()
        .resolve()
    )

    if not env_path.is_file():
        raise ValueError(
            f".env file does not exist: {env_path}"
        )

    manifest = JiraSftTrainingManifest.model_validate(
        json.loads(
            training_manifest_path.read_text(
                encoding="utf-8"
            )
        )
    )

    corpus_manifest, _train, _validation, _prompt = verify_jira_sft_corpus(
        project_root=Path(__file__).resolve().parents[2],
        corpus_directory=Path(manifest.corpus_directory),
        worker_prompt_profile=manifest.worker_prompt_profile,
    )

    if (
        corpus_manifest.base_model_key != manifest.base_model_key
        or corpus_manifest.train_sha256 != manifest.corpus_train_sha256
        or corpus_manifest.validation_sha256 != manifest.corpus_validation_sha256
    ):
        raise ValueError("Jira training manifest does not match its verified corpus.")

    if (
        manifest.system_prompt_sha256 is not None
        and manifest.system_prompt_sha256 != corpus_manifest.system_prompt_sha256
    ):
        raise ValueError("Jira training prompt hash does not match its verified corpus.")
    if manifest.prompt_layout is not None and manifest.prompt_layout != corpus_manifest.prompt_layout:
        raise ValueError("Jira training layout does not match its verified corpus.")

    merged_model_directory = Path(
        manifest.merged_model_directory
    ).expanduser().resolve()

    if not merged_model_directory.is_dir():
        raise ValueError(
            "Trained merged model directory does not exist: "
            f"{merged_model_directory}"
        )

    if fingerprint_directory(
        merged_model_directory
    ) != manifest.merged_model_sha256:
        raise ValueError(
            "Trained merged model fingerprint does not match the "
            "training manifest."
        )

    text = env_path.read_text(
        encoding="utf-8"
    )

    lines = text.splitlines()
    profile_index = None
    raw_value = None

    for index, line in enumerate(
        lines
    ):
        if line.startswith(
            "MODEL_PROFILES="
        ):
            profile_index = index
            raw_value = line.split(
                "=",
                1,
            )[1].strip()
            break

    if profile_index is None or raw_value is None:
        raise ValueError(
            "MODEL_PROFILES is missing from .env."
        )

    if (
        len(raw_value) >= 2
        and raw_value[0] == raw_value[-1]
        and raw_value[0] in {
            "'",
            '"',
        }
    ):
        raw_value = raw_value[
            1:-1
        ]

    profiles = json.loads(
        raw_value
    )

    if not isinstance(
        profiles,
        dict,
    ):
        raise ValueError(
            "MODEL_PROFILES must decode to an object."
        )

    profiles[
        model_key
    ] = {
        "backend": "hf-causal",
        "model_path": str(
            merged_model_directory
        ),
        "enabled": True,
        "quantization": "bnb4",
        "compute_dtype": "bfloat16",
        "model_dtype": "auto",
        "device_map": "auto",
        "worker_prompt_profile": corpus_manifest.worker_prompt_profile,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
    }

    encoded = json.dumps(
        profiles,
        ensure_ascii=False,
        separators=(
            ",",
            ":",
        ),
    )

    lines[
        profile_index
    ] = (
        "MODEL_PROFILES='"
        + encoded
        + "'"
    )

    backup = env_path.with_name(
        ".env.before-jira-sft-registration"
    )

    if not backup.exists():
        shutil.copy2(
            env_path,
            backup,
        )

    env_path.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )

    return profiles[
        model_key
    ]
