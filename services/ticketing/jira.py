from __future__ import annotations

import re

from typing import (
    Any,
)

from urllib.parse import (
    quote,
    urlsplit,
)

import httpx

from .base import (
    TicketService,
)

from .types import (
    TicketLookupResult,
    TicketRecord,
)


JIRA_TICKET_KEY_PATTERN = re.compile(
    r"^[A-Za-z][A-Za-z0-9_]*-\d+$"
)


JIRA_FIELDS = (
    "summary,"
    "status,"
    "issuetype,"
    "priority,"
    "assignee,"
    "reporter,"
    "project,"
    "created,"
    "updated"
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
            httpx.Client(
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

                    error=(
                        key_error
                    ),
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

    def close(
        self,
    ) -> None:

        self.client.close()
