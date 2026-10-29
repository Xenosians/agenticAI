from __future__ import annotations

from services.lua_adapters.http_clients import client as lua_http_client

import re

from typing import (
    Any,
)

from urllib.parse import (
    quote,
    urlsplit,
)

import httpx

from services.jira.authentication import AUTH_EVENT_HOOKS

from .base import (
    TicketService,
)

from .types import (
    TicketAssignedResult,
    TicketComment,
    TicketCommentsResult,
    TicketFieldChange,
    TicketHistoryEntry,
    TicketHistoryResult,
    TicketLink,
    TicketLinksResult,
    TicketLookupResult,
    TicketRecord,
    TicketSearchQuery,
    TicketSearchResult,
)

from .validation import (
    explicit_user_reference,
)


JIRA_TICKET_KEY_PATTERN = re.compile(
    r"^[A-Za-z][A-Za-z0-9_]*-\d+$"
)

JIRA_PROJECT_KEY_PATTERN = re.compile(
    r"^[A-Za-z][A-Za-z0-9_]*$"
)


JIRA_FIELD_NAMES = (
    "summary",
    "status",
    "issuetype",
    "priority",
    "assignee",
    "reporter",
    "project",
    "created",
    "updated",
)


JIRA_FIELDS = (
    ",".join(
        JIRA_FIELD_NAMES
    )
)


class JiraTicketService(
    TicketService
):

    def __init__(
        self,
        *,
        base_url: str,
        email: str,
        api_token: str,
        timeout_seconds: float = 10.0,
        transport: (
            httpx.BaseTransport
            | None
        ) = None,
    ) -> None:

        self.base_url = (
            self._validate_base_url(
                base_url
            )
        )

        self.email = (
            self._require_non_empty(
                email,
                "Jira email",
            )
        )

        self.api_token = (
            self._require_non_empty(
                api_token,
                "Jira API token",
            )
        )

        if (
            not isinstance(
                timeout_seconds,
                (
                    int,
                    float,
                ),
            )
            or isinstance(
                timeout_seconds,
                bool,
            )
            or timeout_seconds <= 0
        ):
            raise ValueError(
                "Jira timeout must be "
                "greater than zero."
            )

        self.client = (
            lua_http_client(
                base_url=(
                    self.base_url
                ),

                auth=(
                    self.email,
                    self.api_token,
                ),

                headers={
                    "Accept":
                        "application/json",
                },

                timeout=(
                    float(
                        timeout_seconds
                    )
                ),

                transport=transport,

                event_hooks=AUTH_EVENT_HOOKS,
            )
        )

    @staticmethod
    def _require_non_empty(
        value: str,
        name: str,
    ) -> str:

        if not isinstance(
            value,
            str,
        ):
            raise ValueError(
                f"{name} must "
                "be a string."
            )

        normalized = (
            value.strip()
        )

        if not normalized:
            raise ValueError(
                f"{name} must not "
                "be empty."
            )

        return normalized

    @classmethod
    def _validate_base_url(
        cls,
        value: str,
    ) -> str:

        normalized = (
            cls._require_non_empty(
                value,
                "Jira base URL",
            )
            .rstrip("/")
        )

        parsed = (
            urlsplit(
                normalized
            )
        )

        if (
            parsed.scheme
            not in {
                "http",
                "https",
            }
        ):
            raise ValueError(
                "Jira base URL must use "
                "http or https."
            )

        if not parsed.hostname:
            raise ValueError(
                "Jira base URL must contain "
                "a hostname."
            )

        if (
            parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError(
                "Jira base URL must not "
                "contain credentials."
            )

        if parsed.query:
            raise ValueError(
                "Jira base URL must not "
                "contain a query string."
            )

        if parsed.fragment:
            raise ValueError(
                "Jira base URL must not "
                "contain a fragment."
            )

        if parsed.path not in {
            "",
            "/",
        }:
            raise ValueError(
                "Jira base URL must be "
                "an origin without a path."
            )

        try:
            parsed.port

        except ValueError as exc:
            raise ValueError(
                "Jira base URL contains "
                "an invalid port."
            ) from exc

        return normalized

    @staticmethod
    def _nested_name(
        value: Any,
        field_name: str = "name",
    ) -> (
        str | None
    ):

        if not isinstance(
            value,
            dict,
        ):
            return None

        nested_value = (
            value.get(
                field_name
            )
        )

        if not isinstance(
            nested_value,
            str,
        ):
            return None

        nested_value = (
            nested_value.strip()
        )

        return (
            nested_value
            if nested_value
            else None
        )

    @staticmethod
    def _string_field(
        value: Any,
    ) -> (
        str | None
    ):

        if not isinstance(
            value,
            str,
        ):
            return None

        normalized = (
            value.strip()
        )

        return (
            normalized
            if normalized
            else None
        )

    @staticmethod
    def _normalize_limit(
        limit: int,
    ) -> tuple[
        int | None,
        str | None,
    ]:

        if (
            not isinstance(
                limit,
                int,
            )
            or isinstance(
                limit,
                bool,
            )
            or limit < 1
            or limit > 50
        ):
            return (
                None,
                (
                    "limit must be an integer "
                    "between 1 and 50."
                ),
            )

        return (
            limit,
            None,
        )

    @classmethod
    def _normalize_ticket_key(
        cls,
        ticket_key: str,
    ) -> tuple[
        str | None,
        str | None,
    ]:

        if not isinstance(
            ticket_key,
            str,
        ):
            return (
                None,
                (
                    "ticket_key must "
                    "be a string."
                ),
            )

        normalized = (
            ticket_key
            .strip()
            .upper()
        )

        if not normalized:
            return (
                None,
                (
                    "ticket_key must not "
                    "be empty."
                ),
            )

        if (
            JIRA_TICKET_KEY_PATTERN
            .fullmatch(
                normalized
            )
            is None
        ):
            return (
                None,
                (
                    "ticket_key is not "
                    "a valid Jira issue key."
                ),
            )

        return (
            normalized,
            None,
        )

    @classmethod
    def _normalize_project_key(
        cls,
        project_key: str,
    ) -> tuple[
        str | None,
        str | None,
    ]:

        if not isinstance(
            project_key,
            str,
        ):
            return (
                None,
                (
                    "project_key must "
                    "be a string."
                ),
            )

        normalized = (
            project_key
            .strip()
            .upper()
        )

        if not normalized:
            return (
                None,
                (
                    "project_key must not "
                    "be empty."
                ),
            )

        if (
            JIRA_PROJECT_KEY_PATTERN
            .fullmatch(
                normalized
            )
            is None
        ):
            return (
                None,
                (
                    "project_key is not "
                    "a valid Jira project key."
                ),
            )

        return (
            normalized,
            None,
        )

    @staticmethod
    def _jql_string(
        value: str,
    ) -> str:

        escaped = (
            value
            .replace(
                "\\",
                "\\\\",
            )
            .replace(
                '"',
                '\\"',
            )
        )

        return (
            f'"{escaped}"'
        )

    @classmethod
    def _build_search_jql(
        cls,
        query: TicketSearchQuery,
    ) -> tuple[
        str | None,
        str | None,
    ]:

        clauses = []

        if (
            query.project_key
            is not None
        ):
            (
                project_key,
                project_error,
            ) = (
                cls._normalize_project_key(
                    query.project_key
                )
            )

            if project_key is None:
                return (
                    None,
                    project_error,
                )

            clauses.append(
                (
                    "project = "
                    f"{cls._jql_string(project_key)}"
                )
            )

        if query.status is not None:
            clauses.append(
                (
                    "status = "
                    f"{cls._jql_string(query.status)}"
                )
            )

        if query.priority is not None:
            clauses.append(
                (
                    "priority = "
                    f"{cls._jql_string(query.priority)}"
                )
            )

        if query.text is not None:
            text_literal = (
                cls._jql_string(
                    query.text
                )
            )

            clauses.append(
                (
                    "("
                    f"summary ~ {text_literal}"
                    " OR "
                    f"description ~ {text_literal}"
                    ")"
                )
            )

        if clauses:
            jql = (
                " AND ".join(
                    clauses
                )
                + " "
            )

        else:
            jql = ""

        jql += (
            "ORDER BY updated DESC"
        )

        return (
            jql,
            None,
        )

    @classmethod
    def _decode_ticket(
        cls,
        payload: Any,
    ) -> (
        TicketRecord | None
    ):

        if not isinstance(
            payload,
            dict,
        ):
            return None

        key = (
            cls._string_field(
                payload.get(
                    "key"
                )
            )
        )

        fields = (
            payload.get(
                "fields"
            )
        )

        if (
            key is None
            or not isinstance(
                fields,
                dict,
            )
        ):
            return None

        summary = (
            cls._string_field(
                fields.get(
                    "summary"
                )
            )
        )

        status = (
            cls._nested_name(
                fields.get(
                    "status"
                )
            )
        )

        if (
            summary is None
            or status is None
        ):
            return None

        assignee = (
            cls._nested_name(
                fields.get(
                    "assignee"
                ),
                "displayName",
            )
        )

        reporter = (
            cls._nested_name(
                fields.get(
                    "reporter"
                ),
                "displayName",
            )
        )

        project = (
            fields.get(
                "project"
            )
        )

        project_key = None
        project_name = None

        if isinstance(
            project,
            dict,
        ):
            project_key = (
                cls._string_field(
                    project.get(
                        "key"
                    )
                )
            )

            project_name = (
                cls._string_field(
                    project.get(
                        "name"
                    )
                )
            )

        return (
            TicketRecord(
                provider="jira",

                key=key,

                summary=summary,

                status=status,

                ticket_type=(
                    cls._nested_name(
                        fields.get(
                            "issuetype"
                        )
                    )
                ),

                priority=(
                    cls._nested_name(
                        fields.get(
                            "priority"
                        )
                    )
                ),

                assignee=assignee,

                reporter=reporter,

                project_key=(
                    project_key
                ),

                project_name=(
                    project_name
                ),

                created_at=(
                    cls._string_field(
                        fields.get(
                            "created"
                        )
                    )
                ),

                updated_at=(
                    cls._string_field(
                        fields.get(
                            "updated"
                        )
                    )
                ),
            )
        )

    @classmethod
    def _decode_history_entry(
        cls,
        payload: Any,
    ) -> (
        TicketHistoryEntry | None
    ):

        if not isinstance(
            payload,
            dict,
        ):
            return None

        raw_items = (
            payload.get(
                "items"
            )
        )

        if not isinstance(
            raw_items,
            list,
        ):
            return None

        changes = []

        for raw_item in raw_items:
            if not isinstance(
                raw_item,
                dict,
            ):
                continue

            field_name = (
                cls._string_field(
                    raw_item.get(
                        "field"
                    )
                )
            )

            if field_name is None:
                continue

            changes.append(
                TicketFieldChange(
                    field=field_name,

                    from_value=(
                        cls._string_field(
                            raw_item.get(
                                "fromString"
                            )
                        )
                    ),

                    to_value=(
                        cls._string_field(
                            raw_item.get(
                                "toString"
                            )
                        )
                    ),
                )
            )

        return (
            TicketHistoryEntry(
                id=(
                    cls._string_field(
                        payload.get(
                            "id"
                        )
                    )
                ),

                author=(
                    cls._nested_name(
                        payload.get(
                            "author"
                        ),
                        "displayName",
                    )
                ),

                created_at=(
                    cls._string_field(
                        payload.get(
                            "created"
                        )
                    )
                ),

                changes=changes,
            )
        )

    @classmethod
    def _adf_to_text(
        cls,
        value: Any,
    ) -> str:
        """
        Convert Jira Atlassian Document Format into bounded,
        presentation-safe plain text.

        Structured rich-text semantics remain provider-side.
        """

        if isinstance(
            value,
            str,
        ):
            return (
                value.strip()
            )

        if isinstance(
            value,
            list,
        ):
            pieces = [
                cls._adf_to_text(
                    item
                )

                for item
                in value
            ]

            return (
                "\n".join(
                    piece

                    for piece
                    in pieces

                    if piece
                )
            )

        if not isinstance(
            value,
            dict,
        ):
            return ""

        node_type = (
            value.get(
                "type"
            )
        )

        if node_type == "text":
            text = (
                value.get(
                    "text"
                )
            )

            if isinstance(
                text,
                str,
            ):
                return (
                    text.strip()
                )

            return ""

        content = (
            value.get(
                "content"
            )
        )

        if not isinstance(
            content,
            list,
        ):
            return ""

        pieces = [
            cls._adf_to_text(
                item
            )

            for item
            in content
        ]

        pieces = [
            piece

            for piece
            in pieces

            if piece
        ]

        if not pieces:
            return ""

        if node_type in {
            "doc",
            "paragraph",
            "heading",
            "blockquote",
            "codeBlock",
            "listItem",
            "bulletList",
            "orderedList",
        }:
            return (
                "\n".join(
                    pieces
                )
            )

        return (
            " ".join(
                pieces
            )
        )

    @classmethod
    def _decode_comment(
        cls,
        payload: Any,
    ) -> (
        TicketComment | None
    ):

        if not isinstance(
            payload,
            dict,
        ):
            return None

        comment_id = (
            cls._string_field(
                payload.get(
                    "id"
                )
            )
        )

        if comment_id is None:
            return None

        body = (
            cls._adf_to_text(
                payload.get(
                    "body"
                )
            )
        )

        if not body:
            body = (
                "[Comment contains no "
                "plain-text content]"
            )

        return (
            TicketComment(
                id=comment_id,

                author=(
                    cls._nested_name(
                        payload.get(
                            "author"
                        ),
                        "displayName",
                    )
                ),

                body=body,

                created_at=(
                    cls._string_field(
                        payload.get(
                            "created"
                        )
                    )
                ),

                updated_at=(
                    cls._string_field(
                        payload.get(
                            "updated"
                        )
                    )
                ),
            )
        )

    def get_ticket(
        self,
        ticket_key: str,
    ) -> TicketLookupResult:

        (
            normalized_key,
            key_error,
        ) = (
            self._normalize_ticket_key(
                ticket_key
            )
        )

        if normalized_key is None:
            return (
                TicketLookupResult(
                    ok=False,
                    status="denied",
                    error=key_error,
                )
            )

        encoded_key = (
            quote(
                normalized_key,
                safe="",
            )
        )

        try:
            response = (
                self.client.get(
                    (
                        "/rest/api/3/"
                        f"issue/{encoded_key}"
                    ),

                    params={
                        "fields":
                            JIRA_FIELDS,
                    },
                )
            )

        except httpx.HTTPError:
            return (
                TicketLookupResult(
                    ok=False,
                    status="error",
                    error=(
                        "Jira request failed."
                    ),
                )
            )

        if (
            response.status_code
            == 404
        ):
            return (
                TicketLookupResult(
                    ok=False,

                    status="not_found",

                    error=(
                        f"Ticket '{normalized_key}' "
                        "was not found."
                    ),
                )
            )

        if (
            response.status_code
            in {
                401,
                403,
            }
        ):
            return (
                TicketLookupResult(
                    ok=False,

                    status="error",

                    error=(
                        "Jira authentication or "
                        "authorization failed."
                    ),
                )
            )

        if (
            response.status_code
            < 200
            or response.status_code
            >= 300
        ):
            return (
                TicketLookupResult(
                    ok=False,

                    status="error",

                    error=(
                        "Jira returned HTTP "
                        f"{response.status_code}."
                    ),
                )
            )

        try:
            payload = (
                response.json()
            )

        except ValueError:
            return (
                TicketLookupResult(
                    ok=False,

                    status="error",

                    error=(
                        "Jira returned an invalid "
                        "JSON response."
                    ),
                )
            )

        ticket = (
            self._decode_ticket(
                payload
            )
        )

        if ticket is None:
            return (
                TicketLookupResult(
                    ok=False,

                    status="error",

                    error=(
                        "Jira returned an invalid "
                        "issue response."
                    ),
                )
            )

        return (
            TicketLookupResult(
                ok=True,
                status="success",
                ticket=ticket,
                error=None,
            )
        )

    def search_tickets(
        self,
        query: TicketSearchQuery,
    ) -> TicketSearchResult:

        if not isinstance(
            query,
            TicketSearchQuery,
        ):
            return (
                TicketSearchResult(
                    ok=False,
                    status="denied",
                    error=(
                        "Invalid ticket search query."
                    ),
                )
            )

        (
            jql,
            jql_error,
        ) = (
            self._build_search_jql(
                query
            )
        )

        if jql is None:
            return (
                TicketSearchResult(
                    ok=False,
                    status="denied",
                    error=jql_error,
                )
            )

        return (
            self._search_jql(
                jql,
                limit=query.limit,
            )
        )

    def _search_jql(
        self,
        jql: str,
        *,
        limit: int,
    ) -> TicketSearchResult:
        """Run one trusted, bounded JQL search and decode its tickets."""

        request_body = {
            "jql":
                jql,

            "maxResults":
                limit,

            "fields":
                list(
                    JIRA_FIELD_NAMES
                ),
        }

        try:
            response = (
                self.client.post(
                    "/rest/api/3/search/jql",

                    json=(
                        request_body
                    ),
                )
            )

        except httpx.HTTPError:
            return (
                TicketSearchResult(
                    ok=False,
                    status="error",
                    error=(
                        "Jira search request failed."
                    ),
                )
            )

        if (
            response.status_code
            in {
                401,
                403,
            }
        ):
            return (
                TicketSearchResult(
                    ok=False,

                    status="error",

                    error=(
                        "Jira authentication or "
                        "authorization failed."
                    ),
                )
            )

        if (
            response.status_code
            == 400
        ):
            return (
                TicketSearchResult(
                    ok=False,

                    status="denied",

                    error=(
                        "Jira rejected the supplied "
                        "ticket search filters."
                    ),
                )
            )

        if (
            response.status_code
            < 200
            or response.status_code
            >= 300
        ):
            return (
                TicketSearchResult(
                    ok=False,

                    status="error",

                    error=(
                        "Jira returned HTTP "
                        f"{response.status_code} "
                        "for ticket search."
                    ),
                )
            )

        try:
            payload = (
                response.json()
            )

        except ValueError:
            return (
                TicketSearchResult(
                    ok=False,

                    status="error",

                    error=(
                        "Jira returned an invalid "
                        "JSON search response."
                    ),
                )
            )

        if not isinstance(
            payload,
            dict,
        ):
            return (
                TicketSearchResult(
                    ok=False,

                    status="error",

                    error=(
                        "Jira returned an invalid "
                        "ticket search response."
                    ),
                )
            )

        raw_issues = (
            payload.get(
                "issues"
            )
        )

        if not isinstance(
            raw_issues,
            list,
        ):
            return (
                TicketSearchResult(
                    ok=False,

                    status="error",

                    error=(
                        "Jira returned an invalid "
                        "ticket search issue list."
                    ),
                )
            )

        tickets = []

        for raw_issue in raw_issues:
            ticket = (
                self._decode_ticket(
                    raw_issue
                )
            )

            if ticket is None:
                return (
                    TicketSearchResult(
                        ok=False,

                        status="error",

                        error=(
                            "Jira returned an invalid "
                            "issue inside search results."
                        ),
                    )
                )

            tickets.append(
                ticket
            )

        next_page_token = (
            payload.get(
                "nextPageToken"
            )
        )

        is_last = (
            payload.get(
                "isLast"
            )
        )

        truncated = (
            (
                isinstance(
                    next_page_token,
                    str,
                )
                and bool(
                    next_page_token.strip()
                )
            )
            or is_last is False
        )

        return (
            TicketSearchResult(
                ok=True,
                status="success",
                tickets=tickets,
                count=len(
                    tickets
                ),
                truncated=truncated,
                error=None,
            )
        )

    def get_ticket_history(
        self,
        ticket_key: str,
        *,
        limit: int = 20,
    ) -> TicketHistoryResult:

        (
            normalized_key,
            key_error,
        ) = (
            self._normalize_ticket_key(
                ticket_key
            )
        )

        if normalized_key is None:
            return (
                TicketHistoryResult(
                    ok=False,
                    status="denied",
                    error=key_error,
                )
            )

        (
            normalized_limit,
            limit_error,
        ) = (
            self._normalize_limit(
                limit
            )
        )

        if normalized_limit is None:
            return (
                TicketHistoryResult(
                    ok=False,
                    status="denied",
                    error=limit_error,
                )
            )

        encoded_key = (
            quote(
                normalized_key,
                safe="",
            )
        )

        try:
            response = (
                self.client.get(
                    (
                        "/rest/api/3/issue/"
                        f"{encoded_key}/changelog"
                    ),

                    params={
                        "startAt":
                            0,

                        "maxResults":
                            normalized_limit,
                    },
                )
            )

        except httpx.HTTPError:
            return (
                TicketHistoryResult(
                    ok=False,

                    status="error",

                    ticket_key=(
                        normalized_key
                    ),

                    error=(
                        "Jira history request failed."
                    ),
                )
            )

        if (
            response.status_code
            == 404
        ):
            return (
                TicketHistoryResult(
                    ok=False,

                    status="not_found",

                    ticket_key=(
                        normalized_key
                    ),

                    error=(
                        f"Ticket '{normalized_key}' "
                        "was not found."
                    ),
                )
            )

        if (
            response.status_code
            in {
                401,
                403,
            }
        ):
            return (
                TicketHistoryResult(
                    ok=False,

                    status="error",

                    ticket_key=(
                        normalized_key
                    ),

                    error=(
                        "Jira authentication or "
                        "authorization failed."
                    ),
                )
            )

        if not (
            200
            <= response.status_code
            < 300
        ):
            return (
                TicketHistoryResult(
                    ok=False,

                    status="error",

                    ticket_key=(
                        normalized_key
                    ),

                    error=(
                        "Jira returned HTTP "
                        f"{response.status_code} "
                        "for ticket history."
                    ),
                )
            )

        try:
            payload = (
                response.json()
            )

        except ValueError:
            return (
                TicketHistoryResult(
                    ok=False,

                    status="error",

                    ticket_key=(
                        normalized_key
                    ),

                    error=(
                        "Jira returned an invalid "
                        "history JSON response."
                    ),
                )
            )

        if not isinstance(
            payload,
            dict,
        ):
            return (
                TicketHistoryResult(
                    ok=False,

                    status="error",

                    ticket_key=(
                        normalized_key
                    ),

                    error=(
                        "Jira returned an invalid "
                        "ticket history response."
                    ),
                )
            )

        raw_values = (
            payload.get(
                "values"
            )
        )

        if not isinstance(
            raw_values,
            list,
        ):
            return (
                TicketHistoryResult(
                    ok=False,

                    status="error",

                    ticket_key=(
                        normalized_key
                    ),

                    error=(
                        "Jira returned an invalid "
                        "history list."
                    ),
                )
            )

        history = []

        for raw_value in raw_values:
            entry = (
                self._decode_history_entry(
                    raw_value
                )
            )

            if entry is None:
                continue

            history.append(
                entry
            )

        total = (
            payload.get(
                "total"
            )
        )

        is_last = (
            payload.get(
                "isLast"
            )
        )

        truncated = (
            is_last is False
            or (
                isinstance(
                    total,
                    int,
                )
                and total
                > len(
                    history
                )
            )
        )

        return (
            TicketHistoryResult(
                ok=True,
                status="success",
                provider="jira",
                ticket_key=(
                    normalized_key
                ),
                history=history,
                count=len(
                    history
                ),
                truncated=truncated,
                error=None,
            )
        )

    def get_ticket_comments(
        self,
        ticket_key: str,
        *,
        limit: int = 20,
    ) -> TicketCommentsResult:

        (
            normalized_key,
            key_error,
        ) = (
            self._normalize_ticket_key(
                ticket_key
            )
        )

        if normalized_key is None:
            return (
                TicketCommentsResult(
                    ok=False,
                    status="denied",
                    error=key_error,
                )
            )

        (
            normalized_limit,
            limit_error,
        ) = (
            self._normalize_limit(
                limit
            )
        )

        if normalized_limit is None:
            return (
                TicketCommentsResult(
                    ok=False,
                    status="denied",
                    error=limit_error,
                )
            )

        encoded_key = (
            quote(
                normalized_key,
                safe="",
            )
        )

        try:
            response = (
                self.client.get(
                    (
                        "/rest/api/3/issue/"
                        f"{encoded_key}/comment"
                    ),

                    params={
                        "startAt":
                            0,

                        "maxResults":
                            normalized_limit,
                    },
                )
            )

        except httpx.HTTPError:
            return (
                TicketCommentsResult(
                    ok=False,

                    status="error",

                    ticket_key=(
                        normalized_key
                    ),

                    error=(
                        "Jira comments request failed."
                    ),
                )
            )

        if (
            response.status_code
            == 404
        ):
            return (
                TicketCommentsResult(
                    ok=False,

                    status="not_found",

                    ticket_key=(
                        normalized_key
                    ),

                    error=(
                        f"Ticket '{normalized_key}' "
                        "was not found."
                    ),
                )
            )

        if (
            response.status_code
            in {
                401,
                403,
            }
        ):
            return (
                TicketCommentsResult(
                    ok=False,

                    status="error",

                    ticket_key=(
                        normalized_key
                    ),

                    error=(
                        "Jira authentication or "
                        "authorization failed."
                    ),
                )
            )

        if not (
            200
            <= response.status_code
            < 300
        ):
            return (
                TicketCommentsResult(
                    ok=False,

                    status="error",

                    ticket_key=(
                        normalized_key
                    ),

                    error=(
                        "Jira returned HTTP "
                        f"{response.status_code} "
                        "for ticket comments."
                    ),
                )
            )

        try:
            payload = (
                response.json()
            )

        except ValueError:
            return (
                TicketCommentsResult(
                    ok=False,

                    status="error",

                    ticket_key=(
                        normalized_key
                    ),

                    error=(
                        "Jira returned an invalid "
                        "comments JSON response."
                    ),
                )
            )

        if not isinstance(
            payload,
            dict,
        ):
            return (
                TicketCommentsResult(
                    ok=False,

                    status="error",

                    ticket_key=(
                        normalized_key
                    ),

                    error=(
                        "Jira returned an invalid "
                        "comments response."
                    ),
                )
            )

        raw_comments = (
            payload.get(
                "comments"
            )
        )

        if not isinstance(
            raw_comments,
            list,
        ):
            return (
                TicketCommentsResult(
                    ok=False,

                    status="error",

                    ticket_key=(
                        normalized_key
                    ),

                    error=(
                        "Jira returned an invalid "
                        "comments list."
                    ),
                )
            )

        comments = []

        for raw_comment in raw_comments:
            comment = (
                self._decode_comment(
                    raw_comment
                )
            )

            if comment is None:
                continue

            comments.append(
                comment
            )

        total = (
            payload.get(
                "total"
            )
        )

        truncated = (
            isinstance(
                total,
                int,
            )
            and total
            > len(
                comments
            )
        )

        return (
            TicketCommentsResult(
                ok=True,
                status="success",
                provider="jira",
                ticket_key=(
                    normalized_key
                ),
                comments=comments,
                count=len(
                    comments
                ),
                truncated=truncated,
                error=None,
            )
        )

    @classmethod
    def _decode_issue_link(
        cls,
        payload: Any,
    ) -> (
        TicketLink | None
    ):
        if not isinstance(
            payload,
            dict,
        ):
            return None

        link_type = payload.get("type")
        if not isinstance(link_type, dict):
            return None

        type_name = cls._string_field(link_type.get("name"))

        # Jira reports each link from the requested issue's side:
        # ``outwardIssue`` reads "<issue> <type.outward> <other>" and
        # ``inwardIssue`` reads "<issue> <type.inward> <other>".
        for direction in ("outward", "inward"):
            other = payload.get(f"{direction}Issue")
            if not isinstance(other, dict):
                continue

            other_key = cls._string_field(other.get("key"))
            relationship = cls._string_field(link_type.get(direction))

            if (
                type_name is None
                or other_key is None
                or relationship is None
            ):
                return None

            other_fields = other.get("fields")
            if not isinstance(other_fields, dict):
                other_fields = {}

            return TicketLink(
                id=cls._string_field(payload.get("id")),
                link_type=type_name,
                direction=direction,
                relationship=relationship,
                other_key=other_key,
                other_summary=cls._string_field(other_fields.get("summary")),
                other_status=cls._nested_name(other_fields.get("status")),
            )

        return None

    def get_ticket_links(
        self,
        ticket_key: str,
    ) -> TicketLinksResult:

        normalized_key, key_error = self._normalize_ticket_key(ticket_key)

        if normalized_key is None:
            return TicketLinksResult(
                ok=False,
                status="denied",
                error=key_error,
            )

        try:
            response = self.client.get(
                f"/rest/api/3/issue/{quote(normalized_key, safe='')}",
                params={"fields": "issuelinks"},
            )
        except httpx.HTTPError:
            return TicketLinksResult(
                ok=False,
                status="error",
                ticket_key=normalized_key,
                error="Jira issue links request failed.",
            )

        if response.status_code == 404:
            return TicketLinksResult(
                ok=False,
                status="not_found",
                ticket_key=normalized_key,
                error=f"Ticket '{normalized_key}' was not found.",
            )

        if response.status_code in {401, 403}:
            return TicketLinksResult(
                ok=False,
                status="error",
                ticket_key=normalized_key,
                error="Jira authentication or authorization failed.",
            )

        if not 200 <= response.status_code < 300:
            return TicketLinksResult(
                ok=False,
                status="error",
                ticket_key=normalized_key,
                error=(
                    f"Jira returned HTTP {response.status_code} "
                    "for ticket links."
                ),
            )

        try:
            payload = response.json()
        except ValueError:
            payload = None

        fields = (
            payload.get("fields")
            if isinstance(payload, dict)
            else None
        )
        raw_links = (
            fields.get("issuelinks")
            if isinstance(fields, dict)
            else None
        )

        if not isinstance(raw_links, list):
            return TicketLinksResult(
                ok=False,
                status="error",
                ticket_key=normalized_key,
                error="Jira returned an invalid issue links response.",
            )

        links = []
        for raw_link in raw_links:
            link = self._decode_issue_link(raw_link)
            if link is not None:
                links.append(link)

        return TicketLinksResult(
            ok=True,
            status="success",
            provider="jira",
            ticket_key=(
                self._string_field(payload.get("key"))
                or normalized_key
            ),
            links=links,
            count=len(links),
        )

    def _user_identity(
        self,
        payload: Any,
    ) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None

        account_id = self._string_field(payload.get("accountId"))
        if account_id is None:
            return None

        return {
            "account_id": account_id,
            "display_name": self._string_field(payload.get("displayName")),
            "email_address": self._string_field(payload.get("emailAddress")),
        }

    def _resolve_user(
        self,
        reference: str,
    ) -> tuple[
        dict[str, Any] | None,
        str,
        str | None,
    ]:
        """
        Resolve one explicit user reference to exactly one Jira account.

        Accepts an account ID, or an exact (case-insensitive) email address
        or display name. Returns ``(identity, status, error)``.
        """

        try:
            if not any(char.isspace() or char == "@" for char in reference):
                response = self.client.get(
                    "/rest/api/3/user",
                    params={"accountId": reference},
                )
                if response.status_code == 200:
                    identity = self._user_identity(response.json())
                    if identity is not None and identity["account_id"] == reference:
                        return identity, "success", None
                elif response.status_code in {401, 403}:
                    return (
                        None,
                        "error",
                        "Jira authentication or authorization failed.",
                    )

            response = self.client.get(
                "/rest/api/3/user/search",
                params={"query": reference, "maxResults": 50},
            )
        except (httpx.HTTPError, ValueError):
            return None, "error", "Jira user lookup failed."

        if response.status_code in {401, 403}:
            return None, "error", "Jira authentication or authorization failed."

        if response.status_code != 200:
            return (
                None,
                "error",
                f"Jira returned HTTP {response.status_code} for user lookup.",
            )

        try:
            candidates = response.json()
        except ValueError:
            candidates = None

        if not isinstance(candidates, list):
            return None, "error", "Jira returned an invalid user lookup response."

        folded = reference.casefold()
        matches: dict[str, dict[str, Any]] = {}

        for candidate in candidates:
            identity = self._user_identity(candidate)
            if identity is None:
                continue
            if any(
                isinstance(identity[field], str)
                and identity[field].casefold() == folded
                for field in ("email_address", "display_name")
            ):
                matches[identity["account_id"]] = identity

        if not matches:
            return (
                None,
                "not_found",
                (
                    "No Jira user exactly matches the supplied account ID, "
                    "email address or display name."
                ),
            )

        if len(matches) > 1:
            return (
                None,
                "denied",
                (
                    f"The supplied user matches {len(matches)} Jira accounts. "
                    "Identify the user by account ID or exact email address."
                ),
            )

        return next(iter(matches.values())), "success", None

    def get_assigned_tickets(
        self,
        assignee: str,
        *,
        status: str | None = None,
        limit: int = 10,
    ) -> TicketAssignedResult:

        reference, reference_error = explicit_user_reference(assignee)
        if reference is None:
            return TicketAssignedResult(
                ok=False,
                status="denied",
                error=reference_error,
            )

        try:
            query = TicketSearchQuery(status=status, limit=limit)
        except ValueError:
            return TicketAssignedResult(
                ok=False,
                status="denied",
                error=(
                    "status must be a string of at most 100 characters and "
                    "limit an integer between 1 and 25."
                ),
            )

        identity, lookup_status, lookup_error = self._resolve_user(reference)
        if identity is None:
            return TicketAssignedResult(
                ok=False,
                status=lookup_status,
                error=lookup_error,
            )

        jql = f"assignee = {self._jql_string(identity['account_id'])} "
        if query.status is not None:
            jql += f"AND status = {self._jql_string(query.status)} "
        jql += "ORDER BY updated DESC"

        search = self._search_jql(jql, limit=query.limit)

        return TicketAssignedResult(
            ok=search.ok,
            status=search.status,
            provider="jira",
            assignee_account_id=identity["account_id"],
            assignee_display_name=identity["display_name"],
            tickets=search.tickets,
            count=search.count,
            truncated=search.truncated,
            error=search.error,
        )

    def close(
        self,
    ) -> None:
        self.client.close()
