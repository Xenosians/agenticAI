from __future__ import annotations

from typing import Any

import httpx

from .jira import JiraTicketService
from .jira_issue_policy import (
    MAX_PROVIDER_ID_LENGTH,
    MAX_PROVIDER_NAME_LENGTH,
    MAX_PROVIDER_TEXT_LENGTH,
    MAX_TICKET_KEY_LENGTH,
    base_snapshot_arguments,
    exact_user_string,
    expected_base_snapshot,
    read_jira_issue_snapshot,
    validate_issue_key,
)


MAX_TICKET_COMMENT_LENGTH = 5000


def prepare_jira_ticket_comment(
    service: JiraTicketService,
    *,
    ticket_key: str,
    comment: str,
) -> dict[str, Any]:
    """Read-only trusted preparation for ``ticket_add_comment``."""

    try:
        requested_key = validate_issue_key(ticket_key)
        requested_comment = exact_user_string(
            comment,
            field_name="comment",
            max_length=MAX_TICKET_COMMENT_LENGTH,
            allow_outer_whitespace=True,
        )

        snapshot = read_jira_issue_snapshot(
            service,
            requested_key,
            operation="comment issue snapshot lookup",
        )

        if snapshot["issue_key"] != requested_key:
            raise ValueError(
                "Jira resolved the requested issue reference to a different "
                "issue key. Trusted policy will not silently rewrite the "
                "requested resource identifier."
            )

        return {
            "ok": True,
            "status": "ready",
            "risk": "medium",
            "requires_approval": True,
            "execution_arguments": {
                "ticket_key": requested_key,
                "comment": requested_comment,
                **base_snapshot_arguments(snapshot),
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


def revalidate_jira_ticket_comment_snapshot(
    service: JiraTicketService,
    *,
    expected_issue_id: str,
    expected_issue_key: str,
    expected_issue_summary: str,
    expected_project_id: str,
    expected_project_key: str,
    expected_project_name: str,
) -> dict[str, Any]:
    expected = expected_base_snapshot(
        expected_issue_id=expected_issue_id,
        expected_issue_key=expected_issue_key,
        expected_issue_summary=expected_issue_summary,
        expected_project_id=expected_project_id,
        expected_project_key=expected_project_key,
        expected_project_name=expected_project_name,
    )

    current = read_jira_issue_snapshot(
        service,
        expected["issue_key"],
        operation="approved comment issue snapshot revalidation",
    )

    if current != expected:
        raise PermissionError(
            "The Jira issue snapshot changed after approval. The approved "
            "comment mutation will not execute against changed provider state."
        )

    return current
