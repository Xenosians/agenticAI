from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil

from datetime import (
    datetime,
    timezone,
)

from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from config import (
    Settings,
)

from learning.training.dpo_qlora import (
    EXPECTED_TRAINING_VERSIONS,
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


# Exact user requests reserved for the isolated Jira model gate.
# Seed SFT generation excludes these strings so the protocol gate
# remains a genuine held-out check rather than training replay.
JIRA_HELD_OUT_EVAL_REQUESTS = {
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
        "allowed_arguments": {
            key: [
                value
            ]
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
) -> list[
    dict[
        str,
        str,
    ]
]:
    return [
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
    output_directory: str
    adapter_directory: str
    merged_model_directory: str
    corpus_directory: str
    corpus_train_sha256: str
    corpus_validation_sha256: str
    max_steps: int
    learning_rate: float
    max_length: int
    lora_r: int
    lora_alpha: int
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
) -> list[
    JiraSftRecord
]:
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

        # Every fifth example for each capability is held out.
        # This guarantees validation coverage across all Jira tools.
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

        semantic_context = (
            semantic_context_for(
                tool_name,
                arguments,
            )
        )

        records.append(
            JiraSftRecord(
                record_id=record_id,
                split=split,
                tool_name=tool_name,
                user_request=normalized_request,
                semantic_context=semantic_context,
                target_response=canonical_tool_call(
                    tool_name,
                    arguments,
                ),
                tags=tags,
            )
        )

    return records


def build_jira_sft_corpus(
    *,
    project_root: Path,
    output_directory: Path,
    base_model_key: str = DEFAULT_BASE_MODEL_KEY,
    force: bool = False,
) -> JiraSftCorpusManifest:
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
            profile.worker_prompt_profile
        ),
    )

    records = _build_records()

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
        created_at=_utc_now(),
        source=(
            "deterministic-reviewed-jira-capability-seed"
        ),
        agent_name=agent.name,
        base_model_key=base_model_key,
        worker_prompt_profile=(
            profile.worker_prompt_profile
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

    if (
        profile.worker_prompt_profile
        != manifest.worker_prompt_profile
    ):
        raise ValueError(
            "Jira SFT corpus prompt profile no longer matches "
            "the configured base model profile."
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
            profile.worker_prompt_profile
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
    messages = build_runtime_messages(
        system_prompt=system_prompt,
        user_request=record.user_request,
        semantic_context=record.semantic_context,
    )

    prompt = _render_prompt(
        tokenizer,
        messages,
    )

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
    )[
        "input_ids"
    ]

    target_text = (
        record.target_response
        + (
            tokenizer.eos_token
            or ""
        )
    )

    target_ids = tokenizer(
        target_text,
        add_special_tokens=False,
    )[
        "input_ids"
    ]

    if not target_ids:
        raise ValueError(
            "Jira SFT target tokenized to zero tokens: "
            f"{record.record_id}"
        )

    if len(target_ids) >= max_length:
        raise ValueError(
            "Jira SFT target exceeds max_length by itself: "
            f"{record.record_id}"
        )

    available_prompt = (
        max_length
        - len(target_ids)
    )

    # Preserve the END of the prompt because it contains the exact
    # semantic target context. Compact prompt mode keeps the complete
    # capability protocol within practical bounds for BLOOMZ.
    if len(prompt_ids) > available_prompt:
        prompt_ids = prompt_ids[
            -available_prompt:
        ]

    input_ids = [
        *prompt_ids,
        *target_ids,
    ]

    labels = [
        *(
            [-100]
            * len(
                prompt_ids
            )
        ),
        *target_ids,
    ]

    attention_mask = [
        1
    ] * len(
        input_ids
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
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
):
    """Build the exact Trainer configuration without loading weights.

    The locked Transformers runtime currently exposes ``warmup_steps``
    rather than accepting ``warmup_ratio`` in the TrainingArguments
    constructor. Keep the intended 3% warmup policy by converting it
    to an explicit
    step count here. Centralizing this construction also lets preflight
    instantiate the real arguments object before any checkpoint is loaded.
    """

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
        save_strategy="no",
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
    )


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
) -> JiraSftTrainingManifest:
    if not allow_training:
        raise PermissionError(
            "Real Jira SFT requires explicit allow_training=True."
        )

    if not isinstance(
        max_steps,
        int,
    ) or max_steps < 1:
        raise ValueError(
            "max_steps must be a positive integer."
        )

    if max_steps > 2000:
        raise ValueError(
            "Jira MVP SFT refuses max_steps > 2000."
        )

    if max_length < 256:
        raise ValueError(
            "max_length must be at least 256."
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

    (
        corpus_manifest,
        train_records,
        _validation_records,
        system_prompt,
    ) = verify_jira_sft_corpus(
        project_root=project_root,
        corpus_directory=corpus_directory,
    )

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

    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
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

    run_directory = (
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

    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    training_args = build_jira_training_arguments(
        output_dir=trainer_directory,
        max_steps=max_steps,
        learning_rate=learning_rate,
        gradient_accumulation_steps=(
            gradient_accumulation_steps
        ),
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=(
            JiraSftTokenDataset(
                tokenized_train
            )
        ),
        data_collator=(
            JiraSftCollator(
                pad_token_id=(
                    tokenizer.pad_token_id
                )
            )
        ),
    )

    trainer.train()

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
        max_shard_size="2GB",
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
        max_steps=max_steps,
        learning_rate=learning_rate,
        max_length=max_length,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
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
    model_key: str = DEFAULT_TRAINED_MODEL_KEY,
) -> dict[
    str,
    Any,
]:
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
        "worker_prompt_profile": "compact",
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
