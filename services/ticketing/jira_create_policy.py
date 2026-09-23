from __future__ import annotations

from typing import (
    Any,
)

from urllib.parse import (
    quote,
)

import httpx

from .jira import (
    JiraTicketService,
)


MAX_PROJECT_KEY_LENGTH = 100
MAX_SUMMARY_LENGTH = 500
MAX_TICKET_TYPE_LENGTH = 100

DEFAULT_TICKET_TYPE = (
    "Task"
)


def _exact_string(
    value: object,
    *,
    field_name: str,
    max_length: int,
) -> str:
    """
    Validate one exact semantic value.

    Trusted policy must not silently trim, rewrite, canonicalize,
    or otherwise change model/user-controlled semantic input.
    """

    if not isinstance(
        value,
        str,
    ):
        raise ValueError(
            f"{field_name} must be a string."
        )

    if not value:
        raise ValueError(
            f"{field_name} must not be empty."
        )

    if (
        value
        != value.strip()
    ):
        raise ValueError(
            f"{field_name} must not contain "
            "leading or trailing whitespace."
        )

    if (
        len(
            value
        )
        > max_length
    ):
        raise ValueError(
            f"{field_name} exceeds the maximum "
            f"length of {max_length} characters."
        )

    return value


def _read_json(
    service: JiraTicketService,
    path: str,
    *,
    operation: str,
    params: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None,
) -> Any:
    """
    Trusted read-only Jira request.

    This helper never executes POST/PUT/DELETE.
    """

    try:
        response = (
            service
            .client
            .get(
                path,
                params=params,
            )
        )

    except (
        httpx.TimeoutException,
        httpx.TransportError,
    ) as exc:

        raise RuntimeError(
            "Jira transport failed during "
            f"{operation}."
        ) from exc

    if response.status_code in {
        401,
        403,
    }:
        raise PermissionError(
            "Jira authentication or authorization "
            f"failed during {operation}."
        )

    if response.status_code == 404:
        raise LookupError(
            "The requested Jira resource was not found "
            f"during {operation}."
        )

    if response.status_code in {
        400,
        422,
    }:
        raise ValueError(
            "Jira rejected the trusted read-only "
            f"{operation} request."
        )

    if response.status_code == 429:
        raise RuntimeError(
            "Jira rate limited the trusted "
            f"{operation} request."
        )

    if response.status_code >= 500:
        raise RuntimeError(
            "Jira returned HTTP "
            f"{response.status_code} during "
            f"{operation}."
        )

    if not (
        200
        <= response.status_code
        < 300
    ):
        raise RuntimeError(
            "Jira returned unexpected HTTP "
            f"{response.status_code} during "
            f"{operation}."
        )

    try:
        return (
            response.json()
        )

    except ValueError as exc:
        raise RuntimeError(
            "Jira returned invalid JSON during "
            f"{operation}."
        ) from exc


def _project_snapshot(
    payload: Any,
) -> dict[
    str,
    str,
]:
    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            "Jira returned invalid project metadata."
        )

    raw_id = (
        payload.get(
            "id"
        )
    )

    if (
        isinstance(
            raw_id,
            int,
        )
        and not isinstance(
            raw_id,
            bool,
        )
    ):
        project_id = str(
            raw_id
        )

    elif isinstance(
        raw_id,
        str,
    ):
        project_id = (
            raw_id
        )

    else:
        raise RuntimeError(
            "Jira project metadata did not contain "
            "a valid project ID."
        )

    project_id = (
        _exact_string(
            project_id,
            field_name=(
                "trusted Jira project ID"
            ),
            max_length=100,
        )
    )

    if not (
        project_id
        .isdigit()
    ):
        raise RuntimeError(
            "Trusted Jira project ID must be numeric."
        )

    project_key = (
        _exact_string(
            payload.get(
                "key"
            ),
            field_name=(
                "trusted Jira project key"
            ),
            max_length=(
                MAX_PROJECT_KEY_LENGTH
            ),
        )
    )

    project_name = (
        _exact_string(
            payload.get(
                "name"
            ),
            field_name=(
                "trusted Jira project name"
            ),
            max_length=255,
        )
    )

    return {
        "id":
            project_id,

        "key":
            project_key,

        "name":
            project_name,
    }


def _resolve_creatable_issue_type(
    service: JiraTicketService,
    *,
    project_id: str,
    requested_name: str,
) -> dict[
    str,
    str,
]:
    """
    Resolve exactly one creatable non-subtask issue type.

    Uses Jira's create-metadata resource rather than merely
    discovering globally visible issue types.
    """

    payload = (
        _read_json(
            service,
            (
                "/rest/api/3/issue/createmeta/"
                + quote(
                    project_id,
                    safe="",
                )
                + "/issuetypes"
            ),
            operation=(
                "issue-create metadata discovery"
            ),
            params={
                "startAt":
                    0,

                "maxResults":
                    50,
            },
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            "Jira returned invalid create-metadata."
        )

    issue_types = (
        payload.get(
            "issueTypes"
        )
    )

    if not isinstance(
        issue_types,
        list,
    ):
        raise RuntimeError(
            "Jira create-metadata did not contain "
            "an issueTypes collection."
        )

    total = (
        payload.get(
            "total"
        )
    )

    if (
        isinstance(
            total,
            int,
        )
        and not isinstance(
            total,
            bool,
        )
        and total
        > len(
            issue_types
        )
    ):
        raise RuntimeError(
            "Jira returned more creatable issue types "
            "than the bounded policy request retrieved. "
            "Trusted policy will not choose from an "
            "incomplete collection."
        )

    matches = []

    for item in issue_types:

        if not isinstance(
            item,
            dict,
        ):
            continue

        name = (
            item.get(
                "name"
            )
        )

        if (
            name
            != requested_name
        ):
            continue

        # Creating a subtask requires a parent and is not part
        # of the current ticket_create capability.
        if (
            item.get(
                "subtask"
            )
            is True
        ):
            continue

        issue_type_id = (
            item.get(
                "id"
            )
        )

        if not isinstance(
            issue_type_id,
            str,
        ):
            continue

        if (
            not issue_type_id
            or issue_type_id
            != issue_type_id.strip()
        ):
            continue

        matches.append(
            {
                "id":
                    issue_type_id,

                "name":
                    requested_name,
            }
        )

    if not matches:
        raise ValueError(
            "The requested Jira issue type is not "
            "available for issue creation in the "
            "selected project."
        )

    if (
        len(
            matches
        )
        != 1
    ):
        raise RuntimeError(
            "Jira returned an ambiguous creatable "
            "issue-type mapping."
        )

    return (
        matches[
            0
        ]
    )


def prepare_jira_ticket_create(
    service: JiraTicketService,
    *,
    project_key: str,
    summary: str,
    ticket_type: (
        str
        | None
    ) = None,
) -> dict[
    str,
    Any,
]:
    """
    READ-ONLY trusted preparation for ticket_create.

    Model/user-controlled values:
        project_key
        summary
        ticket_type, when explicitly supplied

    Trusted provider-owned values:
        expected_project_id
        expected_project_key
        expected_project_name
        expected_ticket_type_id
        expected_ticket_type_name

    No mutation occurs here.
    """

    try:
        requested_project_key = (
            _exact_string(
                project_key,
                field_name=(
                    "project_key"
                ),
                max_length=(
                    MAX_PROJECT_KEY_LENGTH
                ),
            )
        )

        requested_summary = (
            _exact_string(
                summary,
                field_name=(
                    "summary"
                ),
                max_length=(
                    MAX_SUMMARY_LENGTH
                ),
            )
        )

        if ticket_type is None:
            requested_ticket_type = (
                DEFAULT_TICKET_TYPE
            )

        else:
            requested_ticket_type = (
                _exact_string(
                    ticket_type,
                    field_name=(
                        "ticket_type"
                    ),
                    max_length=(
                        MAX_TICKET_TYPE_LENGTH
                    ),
                )
            )

        project_payload = (
            _read_json(
                service,
                (
                    "/rest/api/3/project/"
                    + quote(
                        requested_project_key,
                        safe="",
                    )
                ),
                operation=(
                    "project snapshot lookup"
                ),
            )
        )

        project = (
            _project_snapshot(
                project_payload
            )
        )

        # Resource identifiers are not silently canonicalized.
        #
        # Example:
        #     requested "kan"
        #     provider resolves "KAN"
        #
        # Mutation preparation fails instead of rewriting it.
        if (
            project[
                "key"
            ]
            != requested_project_key
        ):
            raise ValueError(
                "Jira resolved the requested project "
                "reference to a different project key. "
                "Trusted policy will not silently rewrite "
                "the requested resource identifier."
            )

        issue_type = (
            _resolve_creatable_issue_type(
                service,
                project_id=(
                    project[
                        "id"
                    ]
                ),
                requested_name=(
                    requested_ticket_type
                ),
            )
        )

        execution_arguments = {
            # Preserve user/model arguments exactly.
            "project_key":
                requested_project_key,

            "summary":
                requested_summary,

            # Trusted provider snapshot.
            "expected_project_id":
                project[
                    "id"
                ],

            "expected_project_key":
                project[
                    "key"
                ],

            "expected_project_name":
                project[
                    "name"
                ],

            "expected_ticket_type_id":
                issue_type[
                    "id"
                ],

            "expected_ticket_type_name":
                issue_type[
                    "name"
                ],
        }

        # Do not synthesize an optional model-facing argument
        # when the user did not supply one.
        if (
            ticket_type
            is not None
        ):
            execution_arguments[
                "ticket_type"
            ] = (
                requested_ticket_type
            )

        return {
            "ok":
                True,

            "status":
                "ready",

            "risk":
                "medium",

            "requires_approval":
                True,

            "execution_arguments":
                execution_arguments,

            "error":
                None,
        }

    except (
        PermissionError,
        LookupError,
        ValueError,
    ) as exc:

        return {
            "ok":
                False,

            "status":
                "denied",

            "risk":
                "medium",

            "requires_approval":
                True,

            "error":
                str(
                    exc
                ),
        }

    except (
        RuntimeError,
        httpx.HTTPError,
    ) as exc:

        return {
            "ok":
                False,

            "status":
                "error",

            "risk":
                "medium",

            "requires_approval":
                True,

            "error":
                str(
                    exc
                ),
        }



def revalidate_jira_ticket_create_snapshot(
    service: JiraTicketService,
    *,
    expected_project_id: str,
    expected_project_key: str,
    expected_project_name: str,
    expected_ticket_type_id: str,
    expected_ticket_type_name: str,
) -> dict[
    str,
    Any,
]:
    """
    Revalidate the provider-owned snapshot immediately before
    an approved Jira issue creation.

    This method is READ ONLY.

    It prevents an approval prepared against one project/type state
    from silently executing after provider state has changed.
    """

    resolved_project_id = (
        _exact_string(
            expected_project_id,
            field_name=(
                "expected_project_id"
            ),
            max_length=100,
        )
    )

    if not (
        resolved_project_id
        .isdigit()
    ):
        raise ValueError(
            "Trusted Jira project ID must be numeric."
        )

    resolved_project_key = (
        _exact_string(
            expected_project_key,
            field_name=(
                "expected_project_key"
            ),
            max_length=(
                MAX_PROJECT_KEY_LENGTH
            ),
        )
    )

    resolved_project_name = (
        _exact_string(
            expected_project_name,
            field_name=(
                "expected_project_name"
            ),
            max_length=255,
        )
    )

    resolved_type_id = (
        _exact_string(
            expected_ticket_type_id,
            field_name=(
                "expected_ticket_type_id"
            ),
            max_length=100,
        )
    )

    resolved_type_name = (
        _exact_string(
            expected_ticket_type_name,
            field_name=(
                "expected_ticket_type_name"
            ),
            max_length=(
                MAX_TICKET_TYPE_LENGTH
            ),
        )
    )

    project_payload = (
        _read_json(
            service,
            (
                "/rest/api/3/project/"
                + quote(
                    resolved_project_id,
                    safe="",
                )
            ),
            operation=(
                "approved project snapshot revalidation"
            ),
        )
    )

    current_project = (
        _project_snapshot(
            project_payload
        )
    )

    expected_project = {
        "id":
            resolved_project_id,

        "key":
            resolved_project_key,

        "name":
            resolved_project_name,
    }

    if (
        current_project
        != expected_project
    ):
        raise ValueError(
            "Jira project state changed after approval. "
            "The approved ticket-create snapshot is stale."
        )

    current_issue_type = (
        _resolve_creatable_issue_type(
            service,
            project_id=(
                resolved_project_id
            ),
            requested_name=(
                resolved_type_name
            ),
        )
    )

    expected_issue_type = {
        "id":
            resolved_type_id,

        "name":
            resolved_type_name,
    }

    if (
        current_issue_type
        != expected_issue_type
    ):
        raise ValueError(
            "Jira issue-type state changed after approval. "
            "The approved ticket-create snapshot is stale."
        )

    return {
        "project":
            current_project,

        "issue_type":
            current_issue_type,
    }
