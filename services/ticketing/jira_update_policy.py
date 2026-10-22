from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx

from .jira import JiraTicketService
from .jira_issue_policy import (
    MAX_PROVIDER_ID_LENGTH,
    MAX_PROVIDER_NAME_LENGTH,
    base_snapshot_arguments,
    jira_mutation_result,
    jira_rejection_result,
    provider_string,
    read_jira_issue_snapshot,
    revalidate_base_snapshot,
    trusted_get_json,
    validate_issue_key,
)
from .validation import (
    updated_labels,
    validate_update_request,
)


OPERATION = "update_ticket"

UPDATE_SNAPSHOT_ARGUMENTS = (
    "expected_issue_id",
    "expected_issue_key",
    "expected_issue_summary",
    "expected_project_id",
    "expected_project_key",
    "expected_project_name",
)


def _edit_metadata(
    service: JiraTicketService,
    issue_key: str,
) -> dict[str, Any]:
    payload = trusted_get_json(
        service,
        f"/rest/api/3/issue/{quote(issue_key, safe='')}/editmeta",
        operation="issue edit metadata lookup",
    )

    fields = payload.get("fields") if isinstance(payload, dict) else None
    if not isinstance(fields, dict):
        raise RuntimeError("Jira returned invalid issue edit metadata.")

    return fields


def _require_editable(
    edit_fields: dict[str, Any],
    field_id: str,
    operation: str,
    label: str,
) -> dict[str, Any]:
    meta = edit_fields.get(field_id)
    operations = meta.get("operations") if isinstance(meta, dict) else None

    if not isinstance(operations, list) or operation not in operations:
        raise ValueError(
            f"{label} cannot be changed on this issue: the field is not on "
            f"its edit screen or does not allow '{operation}'."
        )

    return meta


def _resolve_priority(
    service: JiraTicketService,
    requested: str,
    *,
    edit_meta: dict[str, Any],
) -> dict[str, str]:
    payload = trusted_get_json(
        service,
        "/rest/api/3/priority",
        operation="priority lookup",
    )

    if not isinstance(payload, list):
        raise RuntimeError("Jira returned invalid priority metadata.")

    allowed = edit_meta.get("allowedValues")
    allowed_ids = (
        {
            str(item.get("id"))
            for item in allowed
            if isinstance(item, dict) and item.get("id") is not None
        }
        if isinstance(allowed, list) and allowed
        else None
    )

    options: list[str] = []
    matches: dict[str, str] = {}

    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            priority_id = provider_string(
                item.get("id"),
                field_name="priority ID",
                max_length=MAX_PROVIDER_ID_LENGTH,
            )
            name = provider_string(
                item.get("name"),
                field_name="priority name",
                max_length=MAX_PROVIDER_NAME_LENGTH,
            )
        except RuntimeError:
            continue

        if allowed_ids is not None and priority_id not in allowed_ids:
            continue

        options.append(name)
        if name.casefold() == requested.casefold():
            matches[priority_id] = name

    if not matches:
        raise ValueError(
            f"Unknown priority '{requested}' for this issue. Valid "
            f"priorities: {', '.join(options) or 'none configured'}."
        )

    if len(matches) > 1:
        raise ValueError(
            f"The priority '{requested}' matches more than one Jira priority."
        )

    priority_id, name = next(iter(matches.items()))
    return {"id": priority_id, "name": name}


def _model_arguments(
    *,
    ticket_key: str,
    priority: object,
    add_labels: object,
    remove_labels: object,
    due_date: object,
) -> dict[str, Any]:
    arguments: dict[str, Any] = {"ticket_key": ticket_key}

    for name, value in (
        ("priority", priority),
        ("add_labels", add_labels),
        ("remove_labels", remove_labels),
        ("due_date", due_date),
    ):
        if value is not None:
            arguments[name] = value

    return arguments


def prepare_jira_ticket_update(
    service: JiraTicketService,
    *,
    ticket_key: str,
    priority: str | None = None,
    add_labels: list[str] | None = None,
    remove_labels: list[str] | None = None,
    due_date: str | None = None,
) -> dict[str, Any]:
    """Read-only preparation that freezes current values and the target priority."""

    try:
        requested_key = validate_issue_key(ticket_key)
        requested = validate_update_request(
            priority=priority,
            add_labels=add_labels,
            remove_labels=remove_labels,
            due_date=due_date,
        )

        snapshot = read_jira_issue_snapshot(
            service,
            requested_key,
            include_editable=True,
            operation="update issue snapshot lookup",
        )

        if snapshot["issue_key"] != requested_key:
            raise ValueError(
                "Jira resolved the requested issue reference to a different "
                "issue key. Trusted policy will not rewrite it."
            )

        edit_fields = _edit_metadata(service, requested_key)

        trusted: dict[str, Any] = {
            **base_snapshot_arguments(snapshot),
            "expected_priority_id": snapshot["priority_id"],
            "expected_priority_name": snapshot["priority_name"],
            "expected_labels": snapshot["labels"],
            "expected_due_date": snapshot["due_date"],
        }

        if requested["priority"] is not None:
            priority_meta = _require_editable(edit_fields, "priority", "set", "Priority")
            target = _resolve_priority(
                service,
                requested["priority"],
                edit_meta=priority_meta,
            )
            trusted["expected_target_priority_id"] = target["id"]
            trusted["expected_target_priority_name"] = target["name"]

        if requested["add_labels"]:
            _require_editable(edit_fields, "labels", "add", "Labels")

        if requested["remove_labels"]:
            _require_editable(edit_fields, "labels", "remove", "Labels")

        if requested["due_date"] is not None:
            _require_editable(edit_fields, "duedate", "set", "Due date")

        return {
            "ok": True,
            "status": "ready",
            "risk": "medium",
            "requires_approval": True,
            "execution_arguments": {
                **_model_arguments(
                    ticket_key=requested_key,
                    priority=priority,
                    add_labels=add_labels,
                    remove_labels=remove_labels,
                    due_date=due_date,
                ),
                **trusted,
            },
            "error": None,
        }

    except (PermissionError, LookupError, ValueError) as exc:
        return {
            "ok": False,
            "status": "denied",
            "risk": "medium",
            "requires_approval": True,
            "error": str(exc),
        }

    except (RuntimeError, httpx.HTTPError) as exc:
        return {
            "ok": False,
            "status": "error",
            "risk": "medium",
            "requires_approval": True,
            "error": str(exc),
        }


def _planned_changes(
    current: dict[str, Any],
    requested: dict[str, Any],
    *,
    target_priority_id: str | None,
    target_priority_name: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Request body, old -> new changes, and the expected read-back state."""

    fields: dict[str, Any] = {}
    changes: list[dict[str, Any]] = []
    expected = dict(current)

    if requested["priority"] is not None and current["priority_id"] != target_priority_id:
        fields["priority"] = {"id": target_priority_id}
        changes.append(
            {
                "field": "priority",
                "from_value": current["priority_name"],
                "to_value": target_priority_name,
            }
        )
        expected["priority_id"] = target_priority_id
        expected["priority_name"] = target_priority_name

    additions = [
        label for label in requested["add_labels"]
        if label not in current["labels"]
    ]
    removals = [
        label for label in requested["remove_labels"]
        if label in current["labels"]
    ]

    if additions or removals:
        final_labels = updated_labels(
            current["labels"],
            add_labels=additions,
            remove_labels=removals,
        )
        changes.append(
            {
                "field": "labels",
                "from_value": ", ".join(current["labels"]) or None,
                "to_value": ", ".join(final_labels) or None,
            }
        )
        expected["labels"] = final_labels

    if requested["due_date"] is not None and current["due_date"] != requested["due_date"]:
        fields["duedate"] = requested["due_date"]
        changes.append(
            {
                "field": "duedate",
                "from_value": current["due_date"],
                "to_value": requested["due_date"],
            }
        )
        expected["due_date"] = requested["due_date"]

    body: dict[str, Any] = {}
    if fields:
        body["fields"] = fields
    if additions or removals:
        # Label operations never replace the whole set, so labels added by
        # someone else after approval are preserved.
        body["update"] = {
            "labels": (
                [{"add": label} for label in additions]
                + [{"remove": label} for label in removals]
            )
        }

    return body, changes, expected


def execute_jira_ticket_update(
    service: JiraTicketService,
    *,
    ticket_key: str,
    priority: str | None = None,
    add_labels: list[str] | None = None,
    remove_labels: list[str] | None = None,
    due_date: str | None = None,
    expected_issue_id: str | None = None,
    expected_issue_key: str | None = None,
    expected_issue_summary: str | None = None,
    expected_project_id: str | None = None,
    expected_project_key: str | None = None,
    expected_project_name: str | None = None,
    expected_priority_id: str | None = None,
    expected_priority_name: str | None = None,
    expected_labels: list[str] | None = None,
    expected_due_date: str | None = None,
    expected_target_priority_id: str | None = None,
    expected_target_priority_name: str | None = None,
) -> dict[str, Any]:
    """Execute one approved update: revalidate, PUT once, verify by read-back."""

    try:
        requested_key = validate_issue_key(ticket_key)
        requested = validate_update_request(
            priority=priority,
            add_labels=add_labels,
            remove_labels=remove_labels,
            due_date=due_date,
        )

        base = {
            "expected_issue_id": expected_issue_id,
            "expected_issue_key": expected_issue_key,
            "expected_issue_summary": expected_issue_summary,
            "expected_project_id": expected_project_id,
            "expected_project_key": expected_project_key,
            "expected_project_name": expected_project_name,
        }

        if (
            any(value is None for value in base.values())
            or not isinstance(expected_labels, list)
            or not all(isinstance(label, str) and label for label in expected_labels)
        ):
            raise ValueError(
                "Approved Jira update requires a complete trusted provider snapshot."
            )

        if expected_issue_key != requested_key:
            raise ValueError(
                "Approved Jira issue snapshot does not match the persisted ticket key."
            )

        if requested["priority"] is not None and (
            not isinstance(expected_target_priority_id, str)
            or not isinstance(expected_target_priority_name, str)
            or expected_target_priority_name.casefold() != requested["priority"].casefold()
        ):
            raise ValueError(
                "Approved Jira update does not carry a trusted target priority "
                "matching the requested priority."
            )

        current = revalidate_base_snapshot(
            service,
            operation="approved update issue snapshot revalidation",
            include_editable=True,
            expected_extra={
                "priority_id": expected_priority_id,
                "priority_name": expected_priority_name,
                "labels": sorted(expected_labels),
                "due_date": expected_due_date,
            },
            **base,
        )

    except (PermissionError, LookupError, ValueError) as exc:
        return jira_mutation_result(
            operation=OPERATION,
            ticket_key=ticket_key,
            status="denied",
            error=str(exc),
        )

    except (RuntimeError, httpx.HTTPError) as exc:
        return jira_mutation_result(
            operation=OPERATION,
            ticket_key=ticket_key,
            status="error",
            error=str(exc),
        )

    body, changes, expected_after = _planned_changes(
        current,
        requested,
        target_priority_id=expected_target_priority_id,
        target_priority_name=expected_target_priority_name,
    )

    if not changes:
        return jira_mutation_result(
            operation=OPERATION,
            ticket_key=requested_key,
            ok=True,
            status="success",
            verification_ok=True,
            changes=[],
            message="The ticket already has the requested values; nothing changed.",
        )

    try:
        response = service.client.put(
            f"/rest/api/3/issue/{quote(requested_key, safe='')}",
            json=body,
        )
    except (httpx.TimeoutException, httpx.TransportError):
        return jira_mutation_result(
            operation=OPERATION,
            ticket_key=requested_key,
            status="outcome_unknown",
            mutation_performed=None,
            changes=changes,
            error=(
                "The Jira update request encountered an ambiguous transport "
                "failure. Automatic retry is disabled."
            ),
        )

    rejection = jira_rejection_result(
        response,
        operation=OPERATION,
        ticket_key=requested_key,
        action="ticket update",
        changes=changes,
    )
    if rejection is not None:
        return rejection

    try:
        after = read_jira_issue_snapshot(
            service,
            requested_key,
            include_editable=True,
            operation="update read-back verification",
        )
    except Exception:
        after = None

    if after != expected_after:
        return jira_mutation_result(
            operation=OPERATION,
            ticket_key=requested_key,
            status="outcome_unknown",
            mutation_performed=True,
            changes=changes,
            error=(
                "Jira accepted the update, but trusted read-back verification "
                "could not confirm exactly the approved values."
            ),
        )

    return jira_mutation_result(
        operation=OPERATION,
        ticket_key=requested_key,
        ok=True,
        status="success",
        changed=True,
        mutation_performed=True,
        verification_ok=True,
        changes=changes,
        message="Ticket updated and verified.",
    )
