from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx

from .jira import JiraTicketService, JIRA_TICKET_KEY_PATTERN


MAX_TICKET_KEY_LENGTH = 100
MAX_PROVIDER_TEXT_LENGTH = 5000
MAX_PROVIDER_NAME_LENGTH = 255
MAX_PROVIDER_ID_LENGTH = 200
MAX_ASSIGNEE_INPUT_LENGTH = 200
MAX_STATUS_INPUT_LENGTH = 100


def exact_user_string(
    value: object,
    *,
    field_name: str,
    max_length: int,
    allow_outer_whitespace: bool = False,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")

    if not value.strip():
        raise ValueError(f"{field_name} must not be empty.")

    if not allow_outer_whitespace and value != value.strip():
        raise ValueError(
            f"{field_name} must not contain leading or trailing whitespace."
        )

    if len(value) > max_length:
        raise ValueError(
            f"{field_name} exceeds the maximum length of "
            f"{max_length} characters."
        )

    return value


def provider_string(
    value: object,
    *,
    field_name: str,
    max_length: int,
) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        value = str(value)

    if not isinstance(value, str):
        raise RuntimeError(f"Jira returned invalid {field_name}.")

    if not value or value != value.strip() or len(value) > max_length:
        raise RuntimeError(f"Jira returned invalid {field_name}.")

    return value


def optional_provider_string(
    value: object,
    *,
    field_name: str,
    max_length: int,
) -> str | None:
    if value is None:
        return None

    return provider_string(
        value,
        field_name=field_name,
        max_length=max_length,
    )


def trusted_get_json(
    service: JiraTicketService,
    path: str,
    *,
    operation: str,
    params: dict[str, Any] | None = None,
) -> Any:
    """Execute one trusted read-only Jira request."""

    try:
        response = service.client.get(path, params=params)
    except (
        httpx.TimeoutException,
        httpx.TransportError,
    ) as exc:
        raise RuntimeError(
            f"Jira {operation} request failed."
        ) from exc

    if response.status_code == 404:
        raise LookupError(
            "The requested Jira resource was not found."
        )

    if response.status_code in {401, 403}:
        raise PermissionError(
            "Jira authentication or authorization failed."
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"Jira returned HTTP {response.status_code} during {operation}."
        )

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Jira returned invalid JSON during {operation}."
        ) from exc


def validate_issue_key(ticket_key: object) -> str:
    requested_key = exact_user_string(
        ticket_key,
        field_name="ticket_key",
        max_length=MAX_TICKET_KEY_LENGTH,
    )

    if JIRA_TICKET_KEY_PATTERN.fullmatch(requested_key) is None:
        raise ValueError(
            "ticket_key is not a valid Jira issue key."
        )

    return requested_key


def read_jira_issue_snapshot(
    service: JiraTicketService,
    issue_key: str,
    *,
    include_assignee: bool = False,
    include_status: bool = False,
    operation: str,
) -> dict[str, Any]:
    fields = [
        "summary",
        "project",
    ]

    if include_assignee:
        fields.append("assignee")

    if include_status:
        fields.append("status")

    payload = trusted_get_json(
        service,
        (
            "/rest/api/3/issue/"
            f"{quote(issue_key, safe='')}"
        ),
        operation=operation,
        params={
            "fields": ",".join(fields),
        },
    )

    if not isinstance(payload, dict):
        raise RuntimeError(
            "Jira returned invalid issue metadata."
        )

    raw_fields = payload.get("fields")
    if not isinstance(raw_fields, dict):
        raise RuntimeError(
            "Jira issue metadata did not contain fields."
        )

    project = raw_fields.get("project")
    if not isinstance(project, dict):
        raise RuntimeError(
            "Jira issue metadata did not contain valid project metadata."
        )

    snapshot: dict[str, Any] = {
        "issue_id": provider_string(
            payload.get("id"),
            field_name="issue ID",
            max_length=MAX_PROVIDER_ID_LENGTH,
        ),
        "issue_key": provider_string(
            payload.get("key"),
            field_name="issue key",
            max_length=MAX_TICKET_KEY_LENGTH,
        ),
        "issue_summary": provider_string(
            raw_fields.get("summary"),
            field_name="issue summary",
            max_length=MAX_PROVIDER_TEXT_LENGTH,
        ),
        "project_id": provider_string(
            project.get("id"),
            field_name="project ID",
            max_length=MAX_PROVIDER_ID_LENGTH,
        ),
        "project_key": provider_string(
            project.get("key"),
            field_name="project key",
            max_length=MAX_TICKET_KEY_LENGTH,
        ),
        "project_name": provider_string(
            project.get("name"),
            field_name="project name",
            max_length=MAX_PROVIDER_NAME_LENGTH,
        ),
    }

    if include_assignee:
        assignee = raw_fields.get("assignee")

        if assignee is None:
            snapshot.update(
                {
                    "assignee_present": False,
                    "assignee_account_id": None,
                    "assignee_display_name": None,
                }
            )
        elif isinstance(assignee, dict):
            snapshot.update(
                {
                    "assignee_present": True,
                    "assignee_account_id": provider_string(
                        assignee.get("accountId"),
                        field_name="assignee account ID",
                        max_length=MAX_PROVIDER_ID_LENGTH,
                    ),
                    "assignee_display_name": optional_provider_string(
                        assignee.get("displayName"),
                        field_name="assignee display name",
                        max_length=MAX_PROVIDER_NAME_LENGTH,
                    ),
                }
            )
        else:
            raise RuntimeError(
                "Jira returned invalid assignee metadata."
            )

    if include_status:
        status = raw_fields.get("status")
        if not isinstance(status, dict):
            raise RuntimeError(
                "Jira returned invalid issue status metadata."
            )

        snapshot.update(
            {
                "status_id": provider_string(
                    status.get("id"),
                    field_name="status ID",
                    max_length=MAX_PROVIDER_ID_LENGTH,
                ),
                "status_name": provider_string(
                    status.get("name"),
                    field_name="status name",
                    max_length=MAX_PROVIDER_NAME_LENGTH,
                ),
            }
        )

    return snapshot


def base_snapshot_arguments(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "expected_issue_id": snapshot["issue_id"],
        "expected_issue_key": snapshot["issue_key"],
        "expected_issue_summary": snapshot["issue_summary"],
        "expected_project_id": snapshot["project_id"],
        "expected_project_key": snapshot["project_key"],
        "expected_project_name": snapshot["project_name"],
    }


def expected_base_snapshot(
    *,
    expected_issue_id: object,
    expected_issue_key: object,
    expected_issue_summary: object,
    expected_project_id: object,
    expected_project_key: object,
    expected_project_name: object,
) -> dict[str, str]:
    return {
        "issue_id": exact_user_string(
            expected_issue_id,
            field_name="expected_issue_id",
            max_length=MAX_PROVIDER_ID_LENGTH,
        ),
        "issue_key": exact_user_string(
            expected_issue_key,
            field_name="expected_issue_key",
            max_length=MAX_TICKET_KEY_LENGTH,
        ),
        "issue_summary": exact_user_string(
            expected_issue_summary,
            field_name="expected_issue_summary",
            max_length=MAX_PROVIDER_TEXT_LENGTH,
        ),
        "project_id": exact_user_string(
            expected_project_id,
            field_name="expected_project_id",
            max_length=MAX_PROVIDER_ID_LENGTH,
        ),
        "project_key": exact_user_string(
            expected_project_key,
            field_name="expected_project_key",
            max_length=MAX_TICKET_KEY_LENGTH,
        ),
        "project_name": exact_user_string(
            expected_project_name,
            field_name="expected_project_name",
            max_length=MAX_PROVIDER_NAME_LENGTH,
        ),
    }
