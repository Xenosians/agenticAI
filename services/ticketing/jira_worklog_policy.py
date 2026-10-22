from __future__ import annotations

from typing import Any, Callable
from urllib.parse import quote

import httpx

from .jira import JiraTicketService
from .jira_issue_policy import (
    MAX_PROVIDER_ID_LENGTH,
    base_snapshot_arguments,
    exact_user_string,
    jira_mutation_result,
    jira_rejection_result,
    provider_string,
    read_jira_issue_snapshot,
    require_issue_permission,
    revalidate_base_snapshot,
    trusted_get_json,
    validate_issue_key,
)
from .validation import (
    MAX_WORKLOG_COMMENT_LENGTH,
    duration_seconds,
    parse_duration,
)


OPERATION = "add_worklog"


def _time_tracking_seconds(
    service: JiraTicketService,
    time_spent: str,
    *,
    operation: str,
) -> int:
    """Seconds Jira will record for ``time_spent`` under the site configuration."""

    amounts = parse_duration(time_spent)

    payload = trusted_get_json(
        service,
        "/rest/api/3/configuration",
        operation=operation,
    )

    if not isinstance(payload, dict):
        raise RuntimeError("Jira returned invalid site configuration.")

    if payload.get("timeTrackingEnabled") is not True:
        raise PermissionError("Time tracking is disabled on this Jira site.")

    config = payload.get("timeTrackingConfiguration")
    hours_per_day = config.get("workingHoursPerDay") if isinstance(config, dict) else None
    days_per_week = config.get("workingDaysPerWeek") if isinstance(config, dict) else None

    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
        for value in (hours_per_day, days_per_week)
    ):
        raise RuntimeError("Jira returned invalid time tracking configuration.")

    return duration_seconds(
        amounts,
        hours_per_day=float(hours_per_day),
        days_per_week=float(days_per_week),
    )


def _exact_comment(comment: object) -> str | None:
    if comment is None:
        return None

    return exact_user_string(
        comment,
        field_name="comment",
        max_length=MAX_WORKLOG_COMMENT_LENGTH,
        allow_outer_whitespace=True,
    )


def prepare_jira_worklog_add(
    service: JiraTicketService,
    *,
    ticket_key: str,
    time_spent: str,
    comment: str | None = None,
) -> dict[str, Any]:
    """Read-only preparation: issue snapshot, permission and recorded seconds."""

    try:
        requested_key = validate_issue_key(ticket_key)
        exact_comment = _exact_comment(comment)

        snapshot = read_jira_issue_snapshot(
            service,
            requested_key,
            operation="worklog issue snapshot lookup",
        )

        if snapshot["issue_key"] != requested_key:
            raise ValueError(
                "Jira resolved the requested issue reference to a different "
                "issue key. Trusted policy will not rewrite it."
            )

        seconds = _time_tracking_seconds(
            service,
            time_spent,
            operation="time tracking configuration lookup",
        )

        require_issue_permission(
            service,
            requested_key,
            "WORK_ON_ISSUES",
            operation="worklog permission check",
        )

        execution_arguments: dict[str, Any] = {
            "ticket_key": requested_key,
            "time_spent": time_spent,
        }
        if exact_comment is not None:
            execution_arguments["comment"] = exact_comment

        return {
            "ok": True,
            "status": "ready",
            "risk": "medium",
            "requires_approval": True,
            "execution_arguments": {
                **execution_arguments,
                **base_snapshot_arguments(snapshot),
                "expected_time_spent_seconds": seconds,
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


def _read_worklog(
    service: JiraTicketService,
    *,
    issue_key: str,
    worklog_id: str,
    comment_text: Callable[[object], str],
) -> dict[str, Any]:
    payload = trusted_get_json(
        service,
        (
            f"/rest/api/3/issue/{quote(issue_key, safe='')}"
            f"/worklog/{quote(worklog_id, safe='')}"
        ),
        operation="worklog read-back verification",
    )

    if not isinstance(payload, dict):
        raise RuntimeError("Jira returned invalid worklog data.")

    raw_comment = payload.get("comment")

    return {
        "worklog_id": provider_string(
            payload.get("id"),
            field_name="worklog ID",
            max_length=MAX_PROVIDER_ID_LENGTH,
        ),
        "issue_id": provider_string(
            payload.get("issueId"),
            field_name="worklog issue ID",
            max_length=MAX_PROVIDER_ID_LENGTH,
        ),
        "time_spent_seconds": payload.get("timeSpentSeconds"),
        "comment": None if raw_comment is None else comment_text(raw_comment),
    }


def execute_jira_worklog_add(
    service: JiraTicketService,
    *,
    ticket_key: str,
    time_spent: str,
    comment: str | None = None,
    expected_issue_id: str | None = None,
    expected_issue_key: str | None = None,
    expected_issue_summary: str | None = None,
    expected_project_id: str | None = None,
    expected_project_key: str | None = None,
    expected_project_name: str | None = None,
    expected_time_spent_seconds: int | None = None,
    comment_document: Callable[[str], dict],
    comment_text: Callable[[object], str],
) -> dict[str, Any]:
    """Log one approved worklog once and verify it by reading it back."""

    try:
        requested_key = validate_issue_key(ticket_key)
        exact_comment = _exact_comment(comment)

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
            or not isinstance(expected_time_spent_seconds, int)
            or isinstance(expected_time_spent_seconds, bool)
        ):
            raise ValueError(
                "Approved Jira worklog requires a complete trusted provider snapshot."
            )

        if expected_issue_key != requested_key:
            raise ValueError(
                "Approved Jira issue snapshot does not match the persisted ticket key."
            )

        revalidate_base_snapshot(
            service,
            operation="approved worklog issue snapshot revalidation",
            **base,
        )

        seconds = _time_tracking_seconds(
            service,
            time_spent,
            operation="approved time tracking revalidation",
        )
        if seconds != expected_time_spent_seconds:
            raise PermissionError(
                "The Jira time tracking configuration changed after approval. "
                "The approved worklog will not be recorded."
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

    body: dict[str, Any] = {"timeSpent": time_spent}
    if exact_comment is not None:
        body["comment"] = comment_document(exact_comment)

    try:
        # The remaining estimate is left as is: the approval covers the
        # worklog only, not a second field change.
        response = service.client.post(
            f"/rest/api/3/issue/{quote(requested_key, safe='')}/worklog",
            params={"adjustEstimate": "leave"},
            json=body,
        )
    except (httpx.TimeoutException, httpx.TransportError):
        return jira_mutation_result(
            operation=OPERATION,
            ticket_key=requested_key,
            status="outcome_unknown",
            mutation_performed=None,
            new_value=time_spent,
            error=(
                "The Jira worklog request encountered an ambiguous transport "
                "failure. Automatic retry is disabled."
            ),
        )

    rejection = jira_rejection_result(
        response,
        operation=OPERATION,
        ticket_key=requested_key,
        action="worklog",
        new_value=time_spent,
    )
    if rejection is not None:
        return rejection

    try:
        payload = response.json()
        worklog_id = provider_string(
            payload.get("id") if isinstance(payload, dict) else None,
            field_name="worklog ID",
            max_length=MAX_PROVIDER_ID_LENGTH,
        )
    except (ValueError, RuntimeError):
        return jira_mutation_result(
            operation=OPERATION,
            ticket_key=requested_key,
            status="outcome_unknown",
            mutation_performed=True,
            new_value=time_spent,
            error=(
                "Jira reported success but did not identify the created worklog "
                "for verification."
            ),
        )

    expected = {
        "worklog_id": worklog_id,
        "issue_id": expected_issue_id,
        "time_spent_seconds": expected_time_spent_seconds,
        "comment": exact_comment,
    }

    try:
        current = _read_worklog(
            service,
            issue_key=requested_key,
            worklog_id=worklog_id,
            comment_text=comment_text,
        )
    except Exception:
        current = None

    if current != expected:
        return jira_mutation_result(
            operation=OPERATION,
            ticket_key=requested_key,
            status="outcome_unknown",
            mutation_performed=True,
            worklog_id=worklog_id,
            new_value=time_spent,
            error=(
                "Jira created the worklog, but trusted read-back could not "
                "confirm exactly the approved time and comment."
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
        worklog_id=worklog_id,
        new_value=time_spent,
        message=f"Logged {time_spent} on {requested_key} and verified the worklog.",
    )
