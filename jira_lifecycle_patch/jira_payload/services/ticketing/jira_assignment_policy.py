from __future__ import annotations

from typing import Any

import httpx

from .jira import JiraTicketService
from .jira_issue_policy import (
    MAX_ASSIGNEE_INPUT_LENGTH,
    MAX_PROVIDER_ID_LENGTH,
    MAX_PROVIDER_NAME_LENGTH,
    base_snapshot_arguments,
    exact_user_string,
    expected_base_snapshot,
    provider_string,
    read_jira_issue_snapshot,
    trusted_get_json,
    validate_issue_key,
)


def _candidate_snapshot(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    try:
        account_id = provider_string(
            value.get("accountId"),
            field_name="assignable account ID",
            max_length=MAX_PROVIDER_ID_LENGTH,
        )
    except RuntimeError:
        return None

    display_name = value.get("displayName")
    if not isinstance(display_name, str) or not display_name.strip():
        display_name = None

    email_address = value.get("emailAddress")
    if not isinstance(email_address, str) or not email_address.strip():
        email_address = None

    active = value.get("active")
    if not isinstance(active, bool):
        active = True

    return {
        "account_id": account_id,
        "display_name": display_name,
        "email_address": email_address,
        "active": active,
    }


def _candidate_matches(
    candidate: dict[str, Any],
    requested: str,
) -> bool:
    if candidate["account_id"] == requested:
        return True

    folded = requested.casefold()

    for field_name in (
        "display_name",
        "email_address",
    ):
        value = candidate.get(field_name)
        if isinstance(value, str) and value.casefold() == folded:
            return True

    return False


def resolve_assignable_user(
    service: JiraTicketService,
    *,
    issue_key: str,
    assignee: str,
) -> dict[str, Any]:
    """Resolve one semantic assignee to exactly one Jira account ID."""

    payload = trusted_get_json(
        service,
        "/rest/api/3/user/assignable/search",
        operation="assignable-user resolution",
        params={
            "issueKey": issue_key,
            "query": assignee,
            "maxResults": 100,
        },
    )

    if not isinstance(payload, list):
        raise RuntimeError(
            "Jira returned invalid assignable-user data."
        )

    matches: dict[str, dict[str, Any]] = {}

    for raw_candidate in payload:
        candidate = _candidate_snapshot(raw_candidate)
        if candidate is None:
            continue
        if not _candidate_matches(candidate, assignee):
            continue
        matches[candidate["account_id"]] = candidate

    if not matches:
        # ``query`` is the human-facing search parameter. If the caller
        # supplied a provider-native account ID, Jira may require its exact
        # ``accountId`` filter instead of returning it from text search.
        account_payload = trusted_get_json(
            service,
            "/rest/api/3/user/assignable/search",
            operation="assignable-user account-ID resolution",
            params={
                "issueKey": issue_key,
                "accountId": assignee,
                "maxResults": 100,
            },
        )

        if not isinstance(account_payload, list):
            raise RuntimeError(
                "Jira returned invalid assignable-user data."
            )

        for raw_candidate in account_payload:
            candidate = _candidate_snapshot(raw_candidate)
            if candidate is None:
                continue
            if candidate["account_id"] == assignee:
                matches[candidate["account_id"]] = candidate

    if not matches:
        raise LookupError(
            "No uniquely matching assignable Jira user was found for the "
            "supplied assignee identifier."
        )

    if len(matches) != 1:
        raise ValueError(
            "The supplied assignee identifier is ambiguous across multiple "
            "assignable Jira accounts."
        )

    candidate = next(iter(matches.values()))

    if candidate["active"] is not True:
        raise PermissionError(
            "The resolved Jira assignee account is inactive."
        )

    return candidate


def _resolve_exact_account_id(
    service: JiraTicketService,
    *,
    issue_key: str,
    account_id: str,
) -> dict[str, Any]:
    payload = trusted_get_json(
        service,
        "/rest/api/3/user/assignable/search",
        operation="approved assignable-user revalidation",
        params={
            "issueKey": issue_key,
            "accountId": account_id,
            "maxResults": 100,
        },
    )

    if not isinstance(payload, list):
        raise RuntimeError(
            "Jira returned invalid assignable-user data."
        )

    candidates: dict[str, dict[str, Any]] = {}
    for raw_candidate in payload:
        candidate = _candidate_snapshot(raw_candidate)
        if candidate is None:
            continue
        if candidate["account_id"] == account_id:
            candidates[account_id] = candidate

    candidate = candidates.get(account_id)
    if candidate is None:
        raise PermissionError(
            "The approved Jira assignee is no longer assignable to this issue."
        )

    if candidate["active"] is not True:
        raise PermissionError(
            "The approved Jira assignee is no longer active."
        )

    return candidate


def prepare_jira_ticket_assignment(
    service: JiraTicketService,
    *,
    ticket_key: str,
    assignee: str,
) -> dict[str, Any]:
    """Read-only preparation that binds issue and target-account identity."""

    try:
        requested_key = validate_issue_key(ticket_key)
        requested_assignee = exact_user_string(
            assignee,
            field_name="assignee",
            max_length=MAX_ASSIGNEE_INPUT_LENGTH,
        )

        snapshot = read_jira_issue_snapshot(
            service,
            requested_key,
            include_assignee=True,
            operation="assignment issue snapshot lookup",
        )

        if snapshot["issue_key"] != requested_key:
            raise ValueError(
                "Jira resolved the requested issue reference to a different "
                "issue key. Trusted policy will not rewrite it."
            )

        target = resolve_assignable_user(
            service,
            issue_key=requested_key,
            assignee=requested_assignee,
        )

        execution_arguments = {
            "ticket_key": requested_key,
            "assignee": requested_assignee,
            **base_snapshot_arguments(snapshot),
            "expected_previous_assignee_present": snapshot["assignee_present"],
            "expected_previous_assignee_account_id": snapshot["assignee_account_id"],
            "expected_previous_assignee_display_name": snapshot["assignee_display_name"],
            "expected_assignee_account_id": target["account_id"],
            "expected_assignee_display_name": target["display_name"],
        }

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


def revalidate_jira_ticket_assignment_snapshot(
    service: JiraTicketService,
    *,
    expected_issue_id: str,
    expected_issue_key: str,
    expected_issue_summary: str,
    expected_project_id: str,
    expected_project_key: str,
    expected_project_name: str,
    expected_previous_assignee_present: bool,
    expected_previous_assignee_account_id: str | None,
    expected_previous_assignee_display_name: str | None,
    expected_assignee_account_id: str,
    expected_assignee_display_name: str | None,
) -> dict[str, Any]:
    if not isinstance(expected_previous_assignee_present, bool):
        raise ValueError(
            "expected_previous_assignee_present must be a boolean."
        )

    expected = expected_base_snapshot(
        expected_issue_id=expected_issue_id,
        expected_issue_key=expected_issue_key,
        expected_issue_summary=expected_issue_summary,
        expected_project_id=expected_project_id,
        expected_project_key=expected_project_key,
        expected_project_name=expected_project_name,
    )

    if expected_previous_assignee_present:
        previous_account_id = exact_user_string(
            expected_previous_assignee_account_id,
            field_name="expected_previous_assignee_account_id",
            max_length=MAX_PROVIDER_ID_LENGTH,
        )
    else:
        if expected_previous_assignee_account_id is not None:
            raise ValueError(
                "Unassigned snapshot must not contain a previous account ID."
            )
        previous_account_id = None

    if expected_previous_assignee_display_name is not None:
        previous_display_name = exact_user_string(
            expected_previous_assignee_display_name,
            field_name="expected_previous_assignee_display_name",
            max_length=MAX_PROVIDER_NAME_LENGTH,
        )
    else:
        previous_display_name = None

    target_account_id = exact_user_string(
        expected_assignee_account_id,
        field_name="expected_assignee_account_id",
        max_length=MAX_PROVIDER_ID_LENGTH,
    )

    if expected_assignee_display_name is not None:
        target_display_name = exact_user_string(
            expected_assignee_display_name,
            field_name="expected_assignee_display_name",
            max_length=MAX_PROVIDER_NAME_LENGTH,
        )
    else:
        target_display_name = None

    current = read_jira_issue_snapshot(
        service,
        expected["issue_key"],
        include_assignee=True,
        operation="approved assignment issue snapshot revalidation",
    )

    expected_current = {
        **expected,
        "assignee_present": expected_previous_assignee_present,
        "assignee_account_id": previous_account_id,
        "assignee_display_name": previous_display_name,
    }

    if current != expected_current:
        raise PermissionError(
            "The Jira issue or assignee state changed after approval. The "
            "approved assignment will not execute against changed state."
        )

    target = _resolve_exact_account_id(
        service,
        issue_key=expected["issue_key"],
        account_id=target_account_id,
    )

    if (
        target_display_name is not None
        and target.get("display_name") != target_display_name
    ):
        raise PermissionError(
            "The approved Jira assignee identity changed after approval."
        )

    return {
        "issue": current,
        "target": target,
    }
