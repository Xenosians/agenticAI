from __future__ import annotations

import hashlib
import json
import re
import shutil

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

from subagents.core.definitions.loader import load_agent_definition
from subagents.core.orchestration.intent_contract import (
    trusted_grounded_arguments,
    trusted_tool_effect,
)
from subagents.core.tooling.capabilities import build_agent_capability_catalog
from tools.registry import get_tool


CAPABILITY_CURRICULUM_SCHEMA = "capability-language-curriculum.v2"
CAPABILITY_RECORD_SCHEMA = "capability-language-record.v2"


class CapabilityLanguageRecord(BaseModel):
    """Training/evaluation material only; never execution authority."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    schema_name: str = Field(
        default=CAPABILITY_RECORD_SCHEMA,
        alias="schema",
    )
    record_id: str
    split: str
    agent_name: str
    tool_name: str
    user_request: str
    arguments: dict[str, Any]
    semantic_context: dict[str, Any]
    target_response: str
    effect: str
    risk: str
    requires_approval: bool
    variant: str
    tags: list[str] = Field(default_factory=list)


class CapabilityLanguageManifest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    schema_name: str = Field(
        default=CAPABILITY_CURRICULUM_SCHEMA,
        alias="schema",
    )
    created_at: str
    agent_name: str
    source_agent_definition: str
    record_count: int
    train_count: int
    validation_count: int
    tool_count: int
    covered_tools: list[str]
    per_tool_counts: dict[str, int]
    read_count: int
    mutation_count: int
    approval_required_count: int
    typo_variant_count: int
    context_variant_count: int
    capability_catalog_sha256: str
    records_sha256: str
    training_authorized: bool = False
    promotion_authorized: bool = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    return _canonical_json(
        [
            {
                "name": tool_name,
                "arguments": arguments,
            }
        ]
    )


def _first_sentence(description: str) -> str:
    text = " ".join(description.split())
    if not text:
        raise ValueError("Capability description is empty.")

    positive = re.split(
        r"\b(?:Do NOT|It does NOT|This does NOT|Never)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()

    if "." in positive:
        positive = positive.split(".", 1)[0].strip()

    return re.sub(r"\s+", " ", positive).rstrip(" .")


def _lower_initial(text: str) -> str:
    words = text.split()
    if words and words[0].isupper():
        words[0] = words[0].lower()
    return " ".join(words)


def _human_tool_phrase(tool_name: str) -> str:
    return tool_name.replace("_", " ").strip()


_SAMPLE_VALUES = {
    "ticket_key": "OPS-17",
    "project_key": "OPS",
    "project_id_or_key": "OPS",
    "project_name": "Operations Hub",
    "new_name": "Operations Hub v2",
    "summary": "Login fails after password reset",
    "comment": "Customer confirmed the issue is reproducible.",
    "assignee": "alice@example.com",
    "status": "In Progress",
    "ticket_type": "Bug",
    "priority": "High",
    "query": "database timeout",
    "text": "database timeout",
    "branch_name": "feature/capability-learning",
    "message": "feat: improve capability language handling",
    "commit_message": "feat: improve capability language handling",
    "path": "README.md",
    "cwd": ".",
    "service": "api",
    "subject": "workspace",
    "executable": "pwd",
}


def _fallback_string_value(argument_name: str) -> str:
    lower = argument_name.strip().lower()

    if lower in _SAMPLE_VALUES:
        return _SAMPLE_VALUES[lower]

    if "repository" in lower:
        return "ai"

    if "file" in lower or "path" in lower:
        return "README.md"

    if "name" in lower:
        return "example-name"

    if "id" in lower or "key" in lower:
        return "EXAMPLE-1"

    return "example-" + lower.replace("_", "-")


def _sample_value(
    *,
    argument_name: str,
    schema: dict[str, Any],
    ordinal: int,
) -> Any:
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return enum[ordinal % len(enum)]

    argument_type = str(schema.get("type", "str")).strip().lower()

    if argument_type in {"str", "string"}:
        return _fallback_string_value(argument_name)

    if argument_type in {
        "list[str]",
        "array[str]",
        "list",
        "array",
    }:
        lower = argument_name.strip().lower()

        if "path" in lower or "file" in lower:
            return ["README.md", "src/app.py"]

        if lower == "args":
            return ["--version"]

        return [_fallback_string_value(argument_name)]

    if argument_type in {"int", "integer"}:
        return 10

    if argument_type in {"float", "number"}:
        return 1.0

    if argument_type in {"bool", "boolean"}:
        return True

    raise ValueError(
        "Unsupported catalog argument type "
        f"{argument_type!r} for {argument_name!r}."
    )


def _policy_argument_sets(
    *,
    tool: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Read existing generic process policy for process-capability supervision.

    No second executable allowlist is created here.
    """

    resolver = tool.get("policy_resolver")

    if getattr(resolver, "__name__", "") != "evaluate_process_policy":
        return []

    try:
        from services.process_runner import (
            ALLOWED_EXECUTABLES,
            GIT_READ_ARGUMENTS,
        )
    except Exception:
        return []

    result: list[dict[str, Any]] = []

    for executable in sorted(ALLOWED_EXECUTABLES):
        policy = ALLOWED_EXECUTABLES[executable]
        argument_policy = policy.get("argument_policy")

        if argument_policy == "none":
            args: list[str] = []
        elif argument_policy == "single_directory_name":
            args = ["scratch"]
        elif argument_policy == "git_read":
            command = sorted(GIT_READ_ARGUMENTS)[0]
            args = list(command)
        else:
            continue

        result.append(
            {
                "executable": executable,
                "args": args,
            }
        )

    return result


def _argument_sets(
    *,
    tool_name: str,
    tool: dict[str, Any],
    capability: dict[str, Any],
) -> list[dict[str, Any]]:
    policy_sets = _policy_argument_sets(tool=tool)
    if policy_sets:
        return policy_sets

    schema = capability.get("argument_schema", {})
    if not isinstance(schema, dict):
        raise ValueError(
            f"Capability '{tool_name}' has invalid argument schema."
        )

    arguments: dict[str, Any] = {}

    for ordinal, (argument_name, argument_schema) in enumerate(
        schema.items()
    ):
        if not isinstance(argument_schema, dict):
            continue

        arguments[argument_name] = _sample_value(
            argument_name=argument_name,
            schema=argument_schema,
            ordinal=ordinal,
        )

    return [arguments]


def _render_value(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(str(item) for item in value) + "]"

    return str(value)


def _argument_clause(arguments: dict[str, Any]) -> str:
    if not arguments:
        return ""

    return "; ".join(
        f"{name}={_render_value(value)}"
        for name, value in arguments.items()
    )


def _typo_once(text: str) -> str:
    """Apply one typo to language only; exact argument values are appended later."""

    words = text.split()

    candidates = [
        (index, word)
        for index, word in enumerate(words)
        if len(re.sub(r"[^A-Za-z]", "", word)) >= 5
    ]

    if not candidates:
        return text

    index, word = max(
        candidates,
        key=lambda item: (len(item[1]), -item[0]),
    )

    letters = list(word)
    positions = [
        pos
        for pos, char in enumerate(letters)
        if char.isalpha()
    ]

    if len(positions) < 4:
        return text

    del letters[positions[len(positions) // 2]]
    words[index] = "".join(letters)

    return " ".join(words)


def _request_variants(
    *,
    tool_name: str,
    description: str,
    arguments: dict[str, Any],
) -> list[tuple[str, str]]:
    operation = _lower_initial(_first_sentence(description))
    human_name = _human_tool_phrase(tool_name)
    clause = _argument_clause(arguments)

    suffix = (
        " Use these exact values: " + clause + "."
        if clause
        else ""
    )

    base = [
        (
            "canonical",
            operation + "." + suffix,
        ),
        (
            "conversational",
            "Can you " + operation + "?" + suffix,
        ),
        (
            "terse",
            "Please handle " + human_name + "." + suffix,
        ),
        (
            "context",
            (
                "Context from the current task: "
                + clause
                + ". Now "
                + operation
                + "."
            )
            if clause
            else (
                "In the current task, "
                + operation
                + "."
            ),
        ),
        (
            "noisy-context",
            (
                "We were already working on this request. "
                "Ignore the casual wording and preserve the exact scope. "
                + operation
                + "."
                + suffix
            ),
        ),
        (
            "typo",
            "pls "
            + _typo_once(operation)
            + "."
            + suffix,
        ),
    ]

    return [
        (variant, " ".join(request.split()))
        for variant, request in base
    ]


def _semantic_context(
    *,
    tool_name: str,
    tool: dict[str, Any],
    arguments: dict[str, Any],
) -> dict[str, Any]:
    allowed_arguments: dict[str, list[str]] = {}

    grounded = trusted_grounded_arguments(
        tool_name=tool_name,
        tool=tool,
    )

    for argument_name in grounded:
        if argument_name not in arguments:
            continue

        value = arguments[argument_name]

        if isinstance(value, str):
            allowed_arguments[argument_name] = [value]
        elif (
            isinstance(value, list)
            and value
            and all(isinstance(item, str) for item in value)
        ):
            allowed_arguments[argument_name] = list(value)

    return {
        "allowed_tools": [tool_name],
        "allowed_arguments": allowed_arguments,
        "forbidden_tools": [],
        "forbidden_arguments": {},
    }


def _record_id(payload: dict[str, Any]) -> str:
    return (
        "caplang-"
        + _sha256_text(_canonical_json(payload))[:24]
    )


def build_agent_capability_records(
    *,
    project_root: Path,
    agent_definition: Path,
    excluded_requests: Iterable[str] = (),
) -> list[CapabilityLanguageRecord]:
    """
    Derive supervision from the exact current agent/tool contract.

    There is no runtime keyword router here. New agent capabilities are
    automatically included from trusted metadata.
    """

    root = project_root.expanduser().resolve()

    agent_path = (
        agent_definition
        if agent_definition.is_absolute()
        else root / agent_definition
    )

    agent = load_agent_definition(agent_path)

    capabilities = build_agent_capability_catalog(
        agent,
        include_arguments=True,
    )

    capability_by_name = {
        capability["name"]: capability
        for capability in capabilities
    }

    excluded = {
        item.strip()
        for item in excluded_requests
        if isinstance(item, str) and item.strip()
    }

    records: list[CapabilityLanguageRecord] = []
    seen_requests: set[str] = set()

    for tool_name in agent.tools:
        capability = capability_by_name.get(tool_name)

        if capability is None:
            raise ValueError(
                "Agent capability missing from trusted catalog: "
                + tool_name
            )

        tool = get_tool(tool_name)

        if tool is None:
            raise ValueError(
                "Unknown trusted capability: "
                + tool_name
            )

        risk = tool.get("risk")
        requires_approval = tool.get("requires_approval")

        if not isinstance(risk, str) or not risk.strip():
            raise ValueError(
                f"Capability '{tool_name}' has no trusted risk."
            )

        if not isinstance(requires_approval, bool):
            raise ValueError(
                f"Capability '{tool_name}' has no approval metadata."
            )

        effect = trusted_tool_effect(tool)

        argument_sets = _argument_sets(
            tool_name=tool_name,
            tool=tool,
            capability=capability,
        )

        if not argument_sets:
            argument_sets = [{}]

        for argument_set_index, arguments in enumerate(argument_sets):
            variants = _request_variants(
                tool_name=tool_name,
                description=capability["description"],
                arguments=arguments,
            )

            for variant_index, (variant, user_request) in enumerate(
                variants
            ):
                normalized_request = user_request.strip()

                if normalized_request in excluded:
                    continue

                if normalized_request in seen_requests:
                    continue

                seen_requests.add(normalized_request)

                split = (
                    "validation"
                    if variant_index == 0
                    else "train"
                )

                semantic_context = _semantic_context(
                    tool_name=tool_name,
                    tool=tool,
                    arguments=arguments,
                )

                tags = [
                    "catalog-derived",
                    effect,
                    (
                        "approval-required"
                        if requires_approval
                        else "no-approval"
                    ),
                    "risk-" + risk.strip().lower(),
                    "variant-" + variant,
                ]

                identity = {
                    "agent_name": agent.name,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "variant": variant,
                    "argument_set_index": argument_set_index,
                    "user_request": normalized_request,
                }

                records.append(
                    CapabilityLanguageRecord(
                        record_id=_record_id(identity),
                        split=split,
                        agent_name=agent.name,
                        tool_name=tool_name,
                        user_request=normalized_request,
                        arguments=arguments,
                        semantic_context=semantic_context,
                        target_response=canonical_tool_call(
                            tool_name,
                            arguments,
                        ),
                        effect=effect,
                        risk=risk.strip().lower(),
                        requires_approval=requires_approval,
                        variant=variant,
                        tags=tags,
                    )
                )

    covered = {
        record.tool_name
        for record in records
    }

    missing = [
        tool_name
        for tool_name in agent.tools
        if tool_name not in covered
    ]

    if missing:
        raise ValueError(
            "Capability curriculum does not cover every agent tool: "
            + ", ".join(missing)
        )

    return records


def build_hub_chosen_response(
    record: CapabilityLanguageRecord,
) -> str:
    """Build the existing Hub contract as a lesson target only."""

    intent = {
        "summary": record.user_request,
        "effect": record.effect,
        "allowed_tools": [record.tool_name],
        "forbidden_tools": [],
        "allowed_arguments":
            record.semantic_context["allowed_arguments"],
        "forbidden_arguments": {},
        "max_tool_calls": 1,
        "clarification_required": False,
    }

    payload = {
        "delegations": [
            {
                "agent": record.agent_name,
                "instructions": record.user_request,
                "intent": intent,
            }
        ]
    }

    return _canonical_json(payload)


def _hub_chapters_for(
    record: CapabilityLanguageRecord,
) -> list[str]:
    chapters = [
        "hub-02-resource-operation",
        "hub-04-delegation",
    ]

    if record.semantic_context["allowed_arguments"]:
        chapters.extend(
            [
                "hub-03-scope-context",
                "hub-05-grounded-bindings",
            ]
        )

    if record.effect == "mutation":
        chapters.append(
            "hub-06-governed-mutations"
        )

    return chapters


def build_hub_lesson_candidates(
    records: list[CapabilityLanguageRecord],
) -> dict[str, list[dict[str, Any]]]:
    by_chapter: dict[str, list[dict[str, Any]]] = {}

    for record in records:
        chosen = build_hub_chosen_response(record)

        for chapter in _hub_chapters_for(record):
            by_chapter.setdefault(
                chapter,
                [],
            ).append(
                {
                    "user_request": record.user_request,
                    "chosen_response": chosen,
                    "note": (
                        "Catalog-derived capability language lesson; "
                        f"agent={record.agent_name}; "
                        f"tool={record.tool_name}; "
                        f"variant={record.variant}; "
                        f"risk={record.risk}; "
                        "execution_authority=none"
                    ),
                }
            )

    return by_chapter


def _write_jsonl(
    path: Path,
    values: Iterable[BaseModel | dict[str, Any]],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        for value in values:
            if isinstance(value, BaseModel):
                payload = value.model_dump(
                    mode="json",
                    by_alias=True,
                )
            else:
                payload = value

            handle.write(
                _canonical_json(payload)
                + "\n"
            )


def materialize_agent_capability_curriculum(
    *,
    project_root: Path,
    agent_definition: Path,
    output_directory: Path,
    force: bool = False,
    excluded_requests: Iterable[str] = (),
) -> CapabilityLanguageManifest:
    root = project_root.expanduser().resolve()

    agent_path = (
        agent_definition
        if agent_definition.is_absolute()
        else root / agent_definition
    )

    output = output_directory.expanduser().resolve()

    if output.exists():
        if not force:
            raise FileExistsError(
                "Capability curriculum output already exists: "
                + str(output)
            )

        shutil.rmtree(output)

    output.mkdir(parents=True, exist_ok=False)

    agent = load_agent_definition(agent_path)

    capabilities = build_agent_capability_catalog(
        agent,
        include_arguments=True,
    )

    records = build_agent_capability_records(
        project_root=root,
        agent_definition=agent_path,
        excluded_requests=excluded_requests,
    )

    train = [
        item
        for item in records
        if item.split == "train"
    ]

    validation = [
        item
        for item in records
        if item.split == "validation"
    ]

    if not train or not validation:
        raise ValueError(
            "Capability curriculum requires train and validation records."
        )

    records_path = output / "records.jsonl"

    _write_jsonl(records_path, records)
    _write_jsonl(
        output / "worker-train.jsonl",
        train,
    )
    _write_jsonl(
        output / "worker-validation.jsonl",
        validation,
    )

    hub_root = output / "hub-lessons"
    hub_root.mkdir(parents=True, exist_ok=False)

    for chapter, items in sorted(
        build_hub_lesson_candidates(records).items()
    ):
        _write_jsonl(
            hub_root / f"{chapter}.jsonl",
            items,
        )

    per_tool = Counter(
        item.tool_name
        for item in records
    )

    manifest = CapabilityLanguageManifest(
        created_at=_utc_now(),
        agent_name=agent.name,
        source_agent_definition=str(agent_path),
        record_count=len(records),
        train_count=len(train),
        validation_count=len(validation),
        tool_count=len(agent.tools),
        covered_tools=sorted(per_tool),
        per_tool_counts=dict(sorted(per_tool.items())),
        read_count=sum(
            1
            for item in records
            if item.effect == "read"
        ),
        mutation_count=sum(
            1
            for item in records
            if item.effect == "mutation"
        ),
        approval_required_count=sum(
            1
            for item in records
            if item.requires_approval
        ),
        typo_variant_count=sum(
            1
            for item in records
            if item.variant == "typo"
        ),
        context_variant_count=sum(
            1
            for item in records
            if "context" in item.variant
        ),
        capability_catalog_sha256=_sha256_text(
            _canonical_json(capabilities)
        ),
        records_sha256=_sha256_file(records_path),
        training_authorized=False,
        promotion_authorized=False,
    )

    (output / "manifest.json").write_text(
        json.dumps(
            manifest.model_dump(
                mode="json",
                by_alias=True,
            ),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return manifest
