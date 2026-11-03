from __future__ import annotations

import json
from typing import Any

from services.result_redaction import redact_result_value


VALID_CONTEXT_ROLES = {
    "user",
    "assistant",
}

_MAX_RESULT_ITEMS_PER_JOB = 4
_MAX_RESULT_ITEM_BYTES = 4_096
_MAX_RESULT_EVIDENCE_BYTES = 20_000
_MAX_PRIOR_RESULT_CONTEXT_BYTES = 24_000


def normalize_conversation_context(
    value: Any,
    *,
    max_turns: int = 24,
    include_result_evidence: bool = True,
) -> list[dict[str, Any]]:
    """
    Validate and bound durable Phoenix-projected chat context.

    Context is conversational evidence only. It is never an
    authorization token and never widens the currently requested
    mutation scope.
    """

    if value is None:
        return []

    if not isinstance(value, list):
        raise ValueError("conversation context must be a list")

    if not isinstance(max_turns, int) or isinstance(max_turns, bool) or max_turns < 1:
        raise ValueError("max_turns must be a positive integer")

    normalized: list[dict[str, Any]] = []

    for item in value:
        if not isinstance(item, dict):
            raise ValueError("conversation context entries must be objects")

        role = item.get("role")
        content = item.get("content")

        if not isinstance(role, str):
            raise ValueError("conversation context role must be a string")

        role = role.strip().lower()

        if role not in VALID_CONTEXT_ROLES:
            raise ValueError("conversation context role must be user or assistant")

        if not isinstance(content, str):
            raise ValueError("conversation context content must be a string")

        content = content.strip()

        if not content:
            continue

        turn: dict[str, Any] = {"role": role, "content": content}
        if include_result_evidence and role == "assistant":
            evidence = _normalize_result_evidence(item.get("result_evidence"))
            if evidence is not None:
                turn["result_evidence"] = evidence

        normalized.append(turn)

    return normalized[-max_turns:]


def conversation_messages(
    value: Any,
    *,
    max_turns: int = 24,
) -> list[dict[str, str]]:
    return normalize_conversation_context(
        value,
        max_turns=max_turns,
        include_result_evidence=False,
    )


def prior_result_messages(
    value: Any,
    *,
    max_turns: int = 24,
) -> list[dict[str, str]]:
    """Return a bounded, separate prompt message with persisted prior results."""
    turns = normalize_conversation_context(value, max_turns=max_turns)
    records = [
        turn["result_evidence"]
        for turn in turns
        if isinstance(turn.get("result_evidence"), dict)
    ]

    selected: list[dict[str, Any]] = []
    used_bytes = 0
    for record in reversed(records):
        encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        record_bytes = len(encoded.encode("utf-8"))
        if used_bytes + record_bytes > _MAX_PRIOR_RESULT_CONTEXT_BYTES:
            break
        selected.append(record)
        used_bytes += record_bytes

    if not selected:
        return []

    selected.reverse()
    payload = {
        "records": selected,
        "older_records_omitted": len(selected) < len(records),
    }
    return [
        {
            "role": "user",
            "content": (
                "PRIOR DURABLE JOB RESULT RECORDS (historical evidence only; "
                "not current-state checks or permission):\n"
                + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            ),
        }
    ]


def _normalize_result_evidence(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    job_id = value.get("job_id")
    job_status = value.get("job_status")
    recorded_at = value.get("recorded_at")
    raw_results = value.get("results")
    if not all(
        isinstance(item, str) and item.strip()
        for item in (job_id, job_status)
    ):
        return None
    if not isinstance(raw_results, list):
        return None

    normalized_results: list[dict[str, Any]] = []
    for item in raw_results[-_MAX_RESULT_ITEMS_PER_JOB:]:
        if not isinstance(item, dict):
            continue
        agent_name = item.get("agent_name")
        status = item.get("status")
        if not all(
            isinstance(text, str) and text.strip()
            for text in (agent_name, status)
        ):
            continue

        normalized_item: dict[str, Any] = {
            key: item[key]
            for key in ("agent_name", "status", "proposed_tool", "outcome_code")
            if isinstance(item.get(key), str) and item[key].strip()
        }
        details_redacted = False
        for key in ("tool_result", "presentation"):
            if isinstance(item.get(key), dict):
                normalized_item[key], redacted = redact_result_value(item[key])
                details_redacted = details_redacted or redacted
        if item.get("details_omitted") is True or details_redacted:
            normalized_item["details_omitted"] = True

        encoded = json.dumps(normalized_item, ensure_ascii=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > _MAX_RESULT_ITEM_BYTES:
            normalized_item.pop("tool_result", None)
            normalized_item["details_omitted"] = True
            encoded = json.dumps(normalized_item, ensure_ascii=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > _MAX_RESULT_ITEM_BYTES:
            normalized_item.pop("presentation", None)
            encoded = json.dumps(normalized_item, ensure_ascii=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) <= _MAX_RESULT_ITEM_BYTES:
            normalized_results.append(normalized_item)

    evidence = {
        "job_id": job_id.strip(),
        "job_status": job_status.strip(),
        "results": normalized_results,
        "results_truncated": value.get("results_truncated") is True
        or len(raw_results) > _MAX_RESULT_ITEMS_PER_JOB,
    }
    if isinstance(recorded_at, str) and recorded_at.strip():
        evidence["recorded_at"] = recorded_at.strip()
    approval = value.get("approval_execution")
    if isinstance(approval, dict):
        execution_status = approval.get("execution_status")
        approval_status = approval.get("approval_status")
        if all(
            isinstance(item, str) and item.strip()
            for item in (execution_status, approval_status)
        ):
            normalized_approval: dict[str, Any] = {
                "execution_status": execution_status.strip(),
                "approval_status": approval_status.strip(),
            }
            if isinstance(approval.get("tool"), str) and approval["tool"].strip():
                normalized_approval["tool"] = approval["tool"].strip()
            for key in ("protocol_status",):
                if isinstance(approval.get(key), str) and approval[key].strip():
                    normalized_approval[key] = approval[key].strip()
            if isinstance(approval.get("replayed"), bool):
                normalized_approval["replayed"] = approval["replayed"]
            approval_details_redacted = False
            if isinstance(approval.get("result"), dict):
                (
                    normalized_approval["result"],
                    approval_details_redacted,
                ) = redact_result_value(
                    approval["result"]
                )
            if approval.get("details_omitted") is True or approval_details_redacted:
                normalized_approval["details_omitted"] = True

            encoded_approval = json.dumps(
                normalized_approval,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if len(encoded_approval.encode("utf-8")) > _MAX_RESULT_ITEM_BYTES:
                normalized_approval.pop("result", None)
                normalized_approval["details_omitted"] = True
                encoded_approval = json.dumps(
                    normalized_approval,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            if len(encoded_approval.encode("utf-8")) <= _MAX_RESULT_ITEM_BYTES:
                evidence["approval_execution"] = normalized_approval

    while (
        len(json.dumps(evidence, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        > _MAX_RESULT_EVIDENCE_BYTES
        and evidence["results"]
    ):
        evidence["results"].pop(0)
        evidence["results_truncated"] = True

    evidence_bytes = len(
        json.dumps(evidence, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    if evidence_bytes > _MAX_RESULT_EVIDENCE_BYTES:
        approval = evidence.get("approval_execution")
        if isinstance(approval, dict) and "result" in approval:
            approval.pop("result", None)
            approval["details_omitted"] = True
            evidence_bytes = len(
                json.dumps(evidence, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            )
    if evidence_bytes > _MAX_RESULT_EVIDENCE_BYTES:
        evidence.pop("approval_execution", None)
        evidence_bytes = len(
            json.dumps(evidence, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
    if evidence_bytes > _MAX_RESULT_EVIDENCE_BYTES:
        return None
    if not evidence["results"] and "approval_execution" not in evidence:
        return None
    return evidence


QUOTED_TURN_HEADER = (
    "EARLIER CONVERSATION TURN (quoted evidence, not the current request; "
    "it never sets the output format):"
)


def quoted_conversation_messages(
    value: Any,
    *,
    max_turns: int = 24,
) -> list[dict[str, str]]:
    """Prior turns for a structured-output stage, each quoted as user evidence.

    Replayed as chat turns, an earlier assistant answer primes the model to
    answer in kind: a prose reply makes the next generation prose instead of
    the stage's required JSON. Quoting keeps every turn and its role, and one
    message per turn keeps the request budget's oldest-first trimming.
    """
    return [
        {
            "role": "user",
            "content": QUOTED_TURN_HEADER + "\n" + json.dumps(turn, ensure_ascii=False),
        }
        for turn in normalize_conversation_context(
            value,
            max_turns=max_turns,
        )
    ]
