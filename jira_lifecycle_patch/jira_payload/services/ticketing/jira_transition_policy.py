from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx

from .jira import JiraTicketService
from .jira_issue_policy import (
    MAX_PROVIDER_ID_LENGTH,
    MAX_PROVIDER_NAME_LENGTH,
    MAX_STATUS_INPUT_LENGTH,
    base_snapshot_arguments,
    exact_user_string,
    expected_base_snapshot,
    provider_string,
    read_jira_issue_snapshot,
    trusted_get_json,
    validate_issue_key,
)


def read_jira_transitions(
    service: JiraTicketService,
    issue_key: str,
    *,
    operation: str,
) -> list[dict[str, str]]:
    payload = trusted_get_json(
        service,
        (
            "/rest/api/3/issue/"
            f"{quote(issue_key, safe='')}/transitions"
        ),
        operation=operation,
    )

    if not isinstance(payload, dict):
        raise RuntimeError(
            "Jira returned invalid transition data."
        )

    raw_transitions = payload.get("transitions")
    if not isinstance(raw_transitions, list):
        raise RuntimeError(
            "Jira returned an invalid transition list."
        )

    transitions: list[dict[str, str]] = []

    for raw_transition in raw_transitions:
        if not isinstance(raw_transition, dict):
            continue

        target = raw_transition.get("to")
        if not isinstance(target, dict):
            continue

        try:
            transition = {
                "transition_id": provider_string(
                    raw_transition.get("id"),
                    field_name="transition ID",
                    max_length=MAX_PROVIDER_ID_LENGTH,
                ),
                "transition_name": provider_string(
                    raw_transition.get("name"),
                    field_name="transition name",
                    max_length=MAX_PROVIDER_NAME_LENGTH,
                ),
                "target_status_id": provider_string(
                    target.get("id"),
                    field_name="target status ID",
                    max_length=MAX_PROVIDER_ID_LENGTH,
                ),
                "target_status_name": provider_string(
                    target.get("name"),
                    field_name="target status name",
                    max_length=MAX_PROVIDER_NAME_LENGTH,
                ),
            }
        except RuntimeError:
            continue

        transitions.append(transition)

    return transitions


def resolve_transition(
    transitions: list[dict[str, str]],
    requested_status: str,
) -> dict[str, str]:
    requested = requested_status.casefold()

    matches: dict[str, dict[str, str]] = {}
    for transition in transitions:
        names = {
            transition["transition_name"].casefold(),
            transition["target_status_name"].casefold(),
        }
        if requested in names:
            matches[transition["transition_id"]] = transition

    if not matches:
        raise LookupError(
            "The requested Jira status is not available as a transition for "
            "this issue."
        )

    if len(matches) != 1:
        raise ValueError(
            "The requested Jira status is ambiguous across multiple provider "
            "transitions. Specify a uniquely resolvable destination."
        )

    return next(iter(matches.values()))


def prepare_jira_ticket_transition(
    service: JiraTicketService,
    *,
    ticket_key: str,
    status: str,
) -> dict[str, Any]:
    """Bind the exact source issue state and exact provider transition."""

    try:
        requested_key = validate_issue_key(ticket_key)
        requested_status = exact_user_string(
            status,
            field_name="status",
            max_length=MAX_STATUS_INPUT_LENGTH,
        )

        snapshot = read_jira_issue_snapshot(
            service,
            requested_key,
            include_status=True,
            operation="transition issue snapshot lookup",
        )

        if snapshot["issue_key"] != requested_key:
            raise ValueError(
                "Jira resolved the requested issue reference to a different "
                "issue key. Trusted policy will not rewrite it."
            )

        execution_arguments: dict[str, Any] = {
            "ticket_key": requested_key,
            "status": requested_status,
            **base_snapshot_arguments(snapshot),
            "expected_source_status_id": snapshot["status_id"],
            "expected_source_status_name": snapshot["status_name"],
        }

        if snapshot["status_name"].casefold() == requested_status.casefold():
            execution_arguments.update(
                {
                    "expected_transition_noop": True,
                    "expected_transition_id": None,
                    "expected_transition_name": None,
                    "expected_target_status_id": snapshot["status_id"],
                    "expected_target_status_name": snapshot["status_name"],
                }
            )
        else:
            transition = resolve_transition(
                read_jira_transitions(
                    service,
                    requested_key,
                    operation="transition discovery",
                ),
                requested_status,
            )

            execution_arguments.update(
                {
                    "expected_transition_noop": False,
                    "expected_transition_id": transition["transition_id"],
                    "expected_transition_name": transition["transition_name"],
                    "expected_target_status_id": transition["target_status_id"],
                    "expected_target_status_name": transition["target_status_name"],
                }
            )

        return {
            "ok": True,
            "status": "ready",
            "risk": "medium",
            "requires_approval": True,
            "execution_arguments": execution_arguments,
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


def revalidate_jira_ticket_transition_snapshot(
    service: JiraTicketService,
    *,
    expected_issue_id: str,
    expected_issue_key: str,
    expected_issue_summary: str,
    expected_project_id: str,
    expected_project_key: str,
    expected_project_name: str,
    expected_source_status_id: str,
    expected_source_status_name: str,
    expected_transition_noop: bool,
    expected_transition_id: str | None,
    expected_transition_name: str | None,
    expected_target_status_id: str,
    expected_target_status_name: str,
) -> dict[str, Any]:
    if not isinstance(expected_transition_noop, bool):
        raise ValueError(
            "expected_transition_noop must be a boolean."
        )

    expected = expected_base_snapshot(
        expected_issue_id=expected_issue_id,
        expected_issue_key=expected_issue_key,
        expected_issue_summary=expected_issue_summary,
        expected_project_id=expected_project_id,
        expected_project_key=expected_project_key,
        expected_project_name=expected_project_name,
    )

    source_status_id = exact_user_string(
        expected_source_status_id,
        field_name="expected_source_status_id",
        max_length=MAX_PROVIDER_ID_LENGTH,
    )
    source_status_name = exact_user_string(
        expected_source_status_name,
        field_name="expected_source_status_name",
        max_length=MAX_PROVIDER_NAME_LENGTH,
    )
    target_status_id = exact_user_string(
        expected_target_status_id,
        field_name="expected_target_status_id",
        max_length=MAX_PROVIDER_ID_LENGTH,
    )
    target_status_name = exact_user_string(
        expected_target_status_name,
        field_name="expected_target_status_name",
        max_length=MAX_PROVIDER_NAME_LENGTH,
    )

    current = read_jira_issue_snapshot(
        service,
        expected["issue_key"],
        include_status=True,
        operation="approved transition issue snapshot revalidation",
    )

    expected_current = {
        **expected,
        "status_id": source_status_id,
        "status_name": source_status_name,
    }

    if current != expected_current:
        raise PermissionError(
            "The Jira issue or workflow source state changed after approval. "
            "The approved transition will not execute against changed state."
        )

    if expected_transition_noop:
        if expected_transition_id is not None or expected_transition_name is not None:
            raise ValueError(
                "A no-op transition snapshot must not contain a provider "
                "transition ID or name."
            )
        if target_status_id != source_status_id or target_status_name != source_status_name:
            raise ValueError(
                "A no-op transition target must match the approved source status."
            )
        return {
            "issue": current,
            "transition": None,
        }

    transition_id = exact_user_string(
        expected_transition_id,
        field_name="expected_transition_id",
        max_length=MAX_PROVIDER_ID_LENGTH,
    )
    transition_name = exact_user_string(
        expected_transition_name,
        field_name="expected_transition_name",
        max_length=MAX_PROVIDER_NAME_LENGTH,
    )

    transitions = read_jira_transitions(
        service,
        expected["issue_key"],
        operation="approved transition revalidation",
    )

    matches = [
        transition
        for transition in transitions
        if transition["transition_id"] == transition_id
    ]

    if len(matches) != 1:
        raise PermissionError(
            "The approved Jira transition is no longer uniquely available."
        )

    transition = matches[0]
    expected_transition = {
        "transition_id": transition_id,
        "transition_name": transition_name,
        "target_status_id": target_status_id,
        "target_status_name": target_status_name,
    }

    if transition != expected_transition:
        raise PermissionError(
            "The approved Jira transition metadata changed after approval."
        )

    return {
        "issue": current,
        "transition": transition,
    }
