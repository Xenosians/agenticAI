from __future__ import annotations

import re

from abc import (
    ABC,
    abstractmethod,
)

from datetime import (
    datetime,
    timezone,
)

from urllib.parse import (
    quote,
)

import httpx

from pydantic import (
    BaseModel,
)

from .base import (
    TicketService,
)

from .jira import (
    JiraTicketService,
)

from .mock import (
    MockTicketService,
)

from .types import (
    TicketComment,
    TicketFieldChange,
    TicketHistoryEntry,
    TicketRecord,
)


MAX_TICKET_COMMENT_LENGTH = 5000
MAX_TICKET_SUMMARY_LENGTH = 500
MAX_TICKET_ASSIGNEE_LENGTH = 200
MAX_TICKET_STATUS_LENGTH = 100
MAX_TICKET_TYPE_LENGTH = 100
MAX_PROJECT_KEY_LENGTH = 100


MOCK_TICKET_NUMBER_PATTERN = re.compile(
    r"^(?P<project>[A-Za-z][A-Za-z0-9_]*)-(?P<number>\d+)$"
)


class TicketMutationResult(
    BaseModel
):
    """
    Provider-neutral ticket mutation result.

    Authorization and approval are intentionally outside this
    service.

    ToolGateway must authorize the exact invocation before any
    method on this service is executed.
    """

    ok: bool

    status: str

    provider: (
        str | None
    ) = None

    ticket_key: (
        str | None
    ) = None

    operation: (
        str | None
    ) = None

    changed: bool = False

    comment_id: (
        str | None
    ) = None

    previous_value: (
        str | None
    ) = None

    new_value: (
        str | None
    ) = None

    message: (
        str | None
    ) = None

    error: (
        str | None
    ) = None


class TicketMutationService(
    ABC
):
    """
    Provider-neutral command boundary for ticket mutations.

    Query operations remain on TicketService.

    This separation makes model-facing state-changing capabilities
    explicit and keeps provider mutation handling isolated from
    ordinary ticket reads.
    """

    @abstractmethod
    def add_comment(
        self,
        ticket_key: str,
        comment: str,
    ) -> TicketMutationResult:
        raise NotImplementedError

    @abstractmethod
    def create_ticket(
        self,
        project_key: str,
        summary: str,
        *,
        ticket_type: str | None = None,
    ) -> TicketMutationResult:
        raise NotImplementedError

    @abstractmethod
    def assign_ticket(
        self,
        ticket_key: str,
        assignee: str,
    ) -> TicketMutationResult:
        raise NotImplementedError

    @abstractmethod
    def transition_ticket(
        self,
        ticket_key: str,
        status: str,
    ) -> TicketMutationResult:
        raise NotImplementedError


def normalize_required_string(
    value: str,
    *,
    field_name: str,
    max_length: int,
) -> tuple[
    str | None,
    str | None,
]:

    if not isinstance(
        value,
        str,
    ):

        return (
            None,
            f"{field_name} must be a string.",
        )

    normalized = (
        value.strip()
    )

    if not normalized:

        return (
            None,
            f"{field_name} must not be empty.",
        )

    if (
        len(
            normalized
        )
        > max_length
    ):

        return (
            None,
            (
                f"{field_name} exceeds the maximum "
                f"length of {max_length} characters."
            ),
        )

    return (
        normalized,
        None,
    )


def normalize_optional_string(
    value: str | None,
    *,
    field_name: str,
    max_length: int,
) -> tuple[
    str | None,
    str | None,
]:

    if value is None:

        return (
            None,
            None,
        )

    return (
        normalize_required_string(
            value,
            field_name=(
                field_name
            ),
            max_length=(
                max_length
            ),
        )
    )


def utc_timestamp(
) -> str:

    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        )
    )


def unknown_mutation_result(
    *,
    provider: str,
    operation: str,
    ticket_key: str | None = None,
    error: str,
) -> TicketMutationResult:
    """
    Represent an uncertain remote mutation.

    After a state-changing HTTP request has been sent, transport or
    server failure does not prove that the mutation did not happen.

    Returning "unknown" prevents callers from interpreting failure
    as safely retryable.
    """

    return (
        TicketMutationResult(
            ok=False,
            status="unknown",
            provider=(
                provider
            ),
            ticket_key=(
                ticket_key
            ),
            operation=(
                operation
            ),
            changed=False,
            error=(
                error
            ),
        )
    )


class MockTicketMutationService(
    TicketMutationService
):
    """
    Command adapter sharing the same in-memory state as the mock
    query service.

    Mutations are therefore immediately observable through the
    existing read capabilities.
    """

    def __init__(
        self,
        ticket_service: MockTicketService,
    ) -> None:

        if not isinstance(
            ticket_service,
            MockTicketService,
        ):

            raise TypeError(
                "MockTicketMutationService requires "
                "MockTicketService."
            )

        self.ticket_service = (
            ticket_service
        )

    def _history_id(
        self,
        ticket_key: str,
    ) -> str:

        history = (
            self.ticket_service
            .history
            .get(
                ticket_key,
                [],
            )
        )

        return (
            f"mock-history-"
            f"{len(history) + 1}"
        )

    def _append_history(
        self,
        *,
        ticket_key: str,
        field: str,
        previous_value: str | None,
        new_value: str | None,
    ) -> None:

        history = (
            self.ticket_service
            .history
            .setdefault(
                ticket_key,
                [],
            )
        )

        history.append(
            TicketHistoryEntry(
                id=(
                    self._history_id(
                        ticket_key
                    )
                ),

                author=(
                    "Agentic ITSM"
                ),

                created_at=(
                    utc_timestamp()
                ),

                changes=[
                    TicketFieldChange(
                        field=(
                            field
                        ),

                        from_value=(
                            previous_value
                        ),

                        to_value=(
                            new_value
                        ),
                    )
                ],
            )
        )

    def add_comment(
        self,
        ticket_key: str,
        comment: str,
    ) -> TicketMutationResult:

        (
            normalized_comment,
            comment_error,
        ) = (
            normalize_required_string(
                comment,
                field_name=(
                    "comment"
                ),
                max_length=(
                    MAX_TICKET_COMMENT_LENGTH
                ),
            )
        )

        if normalized_comment is None:

            return (
                TicketMutationResult(
                    ok=False,
                    status="denied",
                    provider="mock",
                    operation=(
                        "add_comment"
                    ),
                    error=(
                        comment_error
                    ),
                )
            )

        lookup = (
            self.ticket_service
            .get_ticket(
                ticket_key
            )
        )

        if (
            not lookup.ok
            or lookup.ticket
            is None
        ):

            return (
                TicketMutationResult(
                    ok=False,
                    status=(
                        lookup.status
                    ),
                    provider="mock",
                    operation=(
                        "add_comment"
                    ),
                    error=(
                        lookup.error
                    ),
                )
            )

        normalized_key = (
            lookup.ticket.key
        )

        comments = (
            self.ticket_service
            .comments
            .setdefault(
                normalized_key,
                [],
            )
        )

        comment_id = (
            "mock-comment-"
            f"{len(comments) + 1}"
        )

        timestamp = (
            utc_timestamp()
        )

        comments.append(
            TicketComment(
                id=(
                    comment_id
                ),

                author=(
                    "Agentic ITSM"
                ),

                body=(
                    normalized_comment
                ),

                created_at=(
                    timestamp
                ),

                updated_at=(
                    timestamp
                ),
            )
        )

        lookup.ticket.updated_at = (
            timestamp
        )

        return (
            TicketMutationResult(
                ok=True,
                status="success",
                provider="mock",
                ticket_key=(
                    normalized_key
                ),
                operation=(
                    "add_comment"
                ),
                changed=True,
                comment_id=(
                    comment_id
                ),
                message=(
                    "Ticket comment added."
                ),
            )
        )

    def create_ticket(
        self,
        project_key: str,
        summary: str,
        *,
        ticket_type: str | None = None,
    ) -> TicketMutationResult:

        (
            normalized_project,
            project_error,
        ) = (
            normalize_required_string(
                project_key,
                field_name=(
                    "project_key"
                ),
                max_length=(
                    MAX_PROJECT_KEY_LENGTH
                ),
            )
        )

        if normalized_project is None:

            return (
                TicketMutationResult(
                    ok=False,
                    status="denied",
                    provider="mock",
                    operation=(
                        "create_ticket"
                    ),
                    error=(
                        project_error
                    ),
                )
            )

        normalized_project = (
            normalized_project.upper()
        )

        (
            normalized_summary,
            summary_error,
        ) = (
            normalize_required_string(
                summary,
                field_name=(
                    "summary"
                ),
                max_length=(
                    MAX_TICKET_SUMMARY_LENGTH
                ),
            )
        )

        if normalized_summary is None:

            return (
                TicketMutationResult(
                    ok=False,
                    status="denied",
                    provider="mock",
                    operation=(
                        "create_ticket"
                    ),
                    error=(
                        summary_error
                    ),
                )
            )

        (
            normalized_type,
            type_error,
        ) = (
            normalize_optional_string(
                ticket_type,
                field_name=(
                    "ticket_type"
                ),
                max_length=(
                    MAX_TICKET_TYPE_LENGTH
                ),
            )
        )

        if type_error is not None:

            return (
                TicketMutationResult(
                    ok=False,
                    status="denied",
                    provider="mock",
                    operation=(
                        "create_ticket"
                    ),
                    error=(
                        type_error
                    ),
                )
            )

        next_number = 1

        for existing_key in (
            self.ticket_service
            .tickets
        ):

            match = (
                MOCK_TICKET_NUMBER_PATTERN
                .fullmatch(
                    existing_key
                )
            )

            if match is None:

                continue

            if (
                match.group(
                    "project"
                ).upper()
                != normalized_project
            ):

                continue

            number = int(
                match.group(
                    "number"
                )
            )

            next_number = max(
                next_number,
                number + 1,
            )

        ticket_key = (
            f"{normalized_project}-"
            f"{next_number}"
        )

        timestamp = (
            utc_timestamp()
        )

        ticket = (
            TicketRecord(
                provider="mock",
                key=(
                    ticket_key
                ),
                summary=(
                    normalized_summary
                ),
                status="To Do",
                ticket_type=(
                    normalized_type
                    or "Task"
                ),
                priority=None,
                assignee=None,
                reporter=(
                    "Agentic ITSM"
                ),
                project_key=(
                    normalized_project
                ),
                project_name=None,
                created_at=(
                    timestamp
                ),
                updated_at=(
                    timestamp
                ),
            )
        )

        self.ticket_service.tickets[
            ticket_key
        ] = ticket

        self.ticket_service.history[
            ticket_key
        ] = []

        self.ticket_service.comments[
            ticket_key
        ] = []

        return (
            TicketMutationResult(
                ok=True,
                status="success",
                provider="mock",
                ticket_key=(
                    ticket_key
                ),
                operation=(
                    "create_ticket"
                ),
                changed=True,
                new_value=(
                    normalized_summary
                ),
                message=(
                    "Ticket created."
                ),
            )
        )

    def assign_ticket(
        self,
        ticket_key: str,
        assignee: str,
    ) -> TicketMutationResult:

        (
            normalized_assignee,
            assignee_error,
        ) = (
            normalize_required_string(
                assignee,
                field_name=(
                    "assignee"
                ),
                max_length=(
                    MAX_TICKET_ASSIGNEE_LENGTH
                ),
            )
        )

        if normalized_assignee is None:

            return (
                TicketMutationResult(
                    ok=False,
                    status="denied",
                    provider="mock",
                    operation=(
                        "assign_ticket"
                    ),
                    error=(
                        assignee_error
                    ),
                )
            )

        lookup = (
            self.ticket_service
            .get_ticket(
                ticket_key
            )
        )

        if (
            not lookup.ok
            or lookup.ticket
            is None
        ):

            return (
                TicketMutationResult(
                    ok=False,
                    status=(
                        lookup.status
                    ),
                    provider="mock",
                    operation=(
                        "assign_ticket"
                    ),
                    error=(
                        lookup.error
                    ),
                )
            )

        ticket = (
            lookup.ticket
        )

        previous = (
            ticket.assignee
        )

        if (
            previous
            == normalized_assignee
        ):

            return (
                TicketMutationResult(
                    ok=True,
                    status="success",
                    provider="mock",
                    ticket_key=(
                        ticket.key
                    ),
                    operation=(
                        "assign_ticket"
                    ),
                    changed=False,
                    previous_value=(
                        previous
                    ),
                    new_value=(
                        normalized_assignee
                    ),
                    message=(
                        "Ticket already has the "
                        "requested assignee."
                    ),
                )
            )

        ticket.assignee = (
            normalized_assignee
        )

        ticket.updated_at = (
            utc_timestamp()
        )

        self._append_history(
            ticket_key=(
                ticket.key
            ),
            field="assignee",
            previous_value=(
                previous
            ),
            new_value=(
                normalized_assignee
            ),
        )

        return (
            TicketMutationResult(
                ok=True,
                status="success",
                provider="mock",
                ticket_key=(
                    ticket.key
                ),
                operation=(
                    "assign_ticket"
                ),
                changed=True,
                previous_value=(
                    previous
                ),
                new_value=(
                    normalized_assignee
                ),
                message=(
                    "Ticket assignee updated."
                ),
            )
        )

    def transition_ticket(
        self,
        ticket_key: str,
        status: str,
    ) -> TicketMutationResult:

        (
            normalized_status,
            status_error,
        ) = (
            normalize_required_string(
                status,
                field_name=(
                    "status"
                ),
                max_length=(
                    MAX_TICKET_STATUS_LENGTH
                ),
            )
        )

        if normalized_status is None:

            return (
                TicketMutationResult(
                    ok=False,
                    status="denied",
                    provider="mock",
                    operation=(
                        "transition_ticket"
                    ),
                    error=(
                        status_error
                    ),
                )
            )

        lookup = (
            self.ticket_service
            .get_ticket(
                ticket_key
            )
        )

        if (
            not lookup.ok
            or lookup.ticket
            is None
        ):

            return (
                TicketMutationResult(
                    ok=False,
                    status=(
                        lookup.status
                    ),
                    provider="mock",
                    operation=(
                        "transition_ticket"
                    ),
                    error=(
                        lookup.error
                    ),
                )
            )

        ticket = (
            lookup.ticket
        )

        previous = (
            ticket.status
        )

        if (
            previous.casefold()
            == normalized_status.casefold()
        ):

            return (
                TicketMutationResult(
                    ok=True,
                    status="success",
                    provider="mock",
                    ticket_key=(
                        ticket.key
                    ),
                    operation=(
                        "transition_ticket"
                    ),
                    changed=False,
                    previous_value=(
                        previous
                    ),
                    new_value=(
                        previous
                    ),
                    message=(
                        "Ticket is already in the "
                        "requested status."
                    ),
                )
            )

        ticket.status = (
            normalized_status
        )

        ticket.updated_at = (
            utc_timestamp()
        )

        self._append_history(
            ticket_key=(
                ticket.key
            ),
            field="status",
            previous_value=(
                previous
            ),
            new_value=(
                normalized_status
            ),
        )

        return (
            TicketMutationResult(
                ok=True,
                status="success",
                provider="mock",
                ticket_key=(
                    ticket.key
                ),
                operation=(
                    "transition_ticket"
                ),
                changed=True,
                previous_value=(
                    previous
                ),
                new_value=(
                    normalized_status
                ),
                message=(
                    "Ticket status updated."
                ),
            )
        )


class JiraTicketMutationService(
    TicketMutationService
):
    """
    Jira command adapter.

    Reuses the authenticated JiraTicketService HTTP client so
    provider credentials, base URL, timeout, and lifecycle stay
    single-source-of-truth.
    """

    def __init__(
        self,
        ticket_service: JiraTicketService,
    ) -> None:

        if not isinstance(
            ticket_service,
            JiraTicketService,
        ):

            raise TypeError(
                "JiraTicketMutationService requires "
                "JiraTicketService."
            )

        self.ticket_service = (
            ticket_service
        )

    @staticmethod
    def _comment_adf(
        comment: str,
    ) -> dict:

        paragraphs = []

        for line in (
            comment.splitlines()
            or [
                comment,
            ]
        ):

            content = []

            if line:

                content.append(
                    {
                        "type":
                            "text",

                        "text":
                            line,
                    }
                )

            paragraphs.append(
                {
                    "type":
                        "paragraph",

                    "content":
                        content,
                }
            )

        return {
            "type":
                "doc",

            "version":
                1,

            "content":
                paragraphs,
        }

    def _normalized_ticket_key(
        self,
        ticket_key: str,
    ) -> tuple[
        str | None,
        str | None,
    ]:

        return (
            self.ticket_service
            ._normalize_ticket_key(
                ticket_key
            )
        )

    def add_comment(
        self,
        ticket_key: str,
        comment: str,
    ) -> TicketMutationResult:

        (
            normalized_key,
            key_error,
        ) = (
            self._normalized_ticket_key(
                ticket_key
            )
        )

        if normalized_key is None:

            return (
                TicketMutationResult(
                    ok=False,
                    status="denied",
                    provider="jira",
                    operation=(
                        "add_comment"
                    ),
                    error=(
                        key_error
                    ),
                )
            )

        (
            normalized_comment,
            comment_error,
        ) = (
            normalize_required_string(
                comment,
                field_name=(
                    "comment"
                ),
                max_length=(
                    MAX_TICKET_COMMENT_LENGTH
                ),
            )
        )

        if normalized_comment is None:

            return (
                TicketMutationResult(
                    ok=False,
                    status="denied",
                    provider="jira",
                    ticket_key=(
                        normalized_key
                    ),
                    operation=(
                        "add_comment"
                    ),
                    error=(
                        comment_error
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
                self.ticket_service
                .client
                .post(
                    (
                        "/rest/api/3/"
                        f"issue/{encoded_key}/comment"
                    ),

                    json={
                        "body":
                            self._comment_adf(
                                normalized_comment
                            ),
                    },
                )
            )

        except httpx.HTTPError:

            return (
                unknown_mutation_result(
                    provider="jira",
                    operation=(
                        "add_comment"
                    ),
                    ticket_key=(
                        normalized_key
                    ),
                    error=(
                        "The Jira comment request failed and "
                        "the remote mutation state is unknown."
                    ),
                )
            )

        if (
            response.status_code
            == 404
        ):

            return (
                TicketMutationResult(
                    ok=False,
                    status="not_found",
                    provider="jira",
                    ticket_key=(
                        normalized_key
                    ),
                    operation=(
                        "add_comment"
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
                TicketMutationResult(
                    ok=False,
                    status="error",
                    provider="jira",
                    ticket_key=(
                        normalized_key
                    ),
                    operation=(
                        "add_comment"
                    ),
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
                TicketMutationResult(
                    ok=False,
                    status="denied",
                    provider="jira",
                    ticket_key=(
                        normalized_key
                    ),
                    operation=(
                        "add_comment"
                    ),
                    error=(
                        "Jira rejected the supplied "
                        "ticket comment."
                    ),
                )
            )

        if not (
            200
            <= response.status_code
            < 300
        ):

            if (
                response.status_code
                >= 500
            ):

                return (
                    unknown_mutation_result(
                        provider="jira",
                        operation=(
                            "add_comment"
                        ),
                        ticket_key=(
                            normalized_key
                        ),
                        error=(
                            "Jira returned HTTP "
                            f"{response.status_code} "
                            "while adding the comment; "
                            "remote mutation state is unknown."
                        ),
                    )
                )

            return (
                TicketMutationResult(
                    ok=False,
                    status="error",
                    provider="jira",
                    ticket_key=(
                        normalized_key
                    ),
                    operation=(
                        "add_comment"
                    ),
                    error=(
                        "Jira returned HTTP "
                        f"{response.status_code} "
                        "while adding the comment."
                    ),
                )
            )

        comment_id = None

        try:

            payload = (
                response.json()
            )

        except ValueError:

            payload = None

        if isinstance(
            payload,
            dict,
        ):

            comment_id = (
                self.ticket_service
                ._string_field(
                    payload.get(
                        "id"
                    )
                )
            )

        return (
            TicketMutationResult(
                ok=True,
                status="success",
                provider="jira",
                ticket_key=(
                    normalized_key
                ),
                operation=(
                    "add_comment"
                ),
                changed=True,
                comment_id=(
                    comment_id
                ),
                message=(
                    "Ticket comment added."
                ),
            )
        )

    def create_ticket(
        self,
        project_key: str,
        summary: str,
        *,
        ticket_type: str | None = None,
    ) -> TicketMutationResult:

        (
            normalized_project,
            project_error,
        ) = (
            self.ticket_service
            ._normalize_project_key(
                project_key
            )
        )

        if normalized_project is None:

            return (
                TicketMutationResult(
                    ok=False,
                    status="denied",
                    provider="jira",
                    operation=(
                        "create_ticket"
                    ),
                    error=(
                        project_error
                    ),
                )
            )

        (
            normalized_summary,
            summary_error,
        ) = (
            normalize_required_string(
                summary,
                field_name=(
                    "summary"
                ),
                max_length=(
                    MAX_TICKET_SUMMARY_LENGTH
                ),
            )
        )

        if normalized_summary is None:

            return (
                TicketMutationResult(
                    ok=False,
                    status="denied",
                    provider="jira",
                    operation=(
                        "create_ticket"
                    ),
                    error=(
                        summary_error
                    ),
                )
            )

        (
            normalized_type,
            type_error,
        ) = (
            normalize_optional_string(
                ticket_type,
                field_name=(
                    "ticket_type"
                ),
                max_length=(
                    MAX_TICKET_TYPE_LENGTH
                ),
            )
        )

        if type_error is not None:

            return (
                TicketMutationResult(
                    ok=False,
                    status="denied",
                    provider="jira",
                    operation=(
                        "create_ticket"
                    ),
                    error=(
                        type_error
                    ),
                )
            )

        request_body = {
            "fields": {
                "project": {
                    "key":
                        normalized_project,
                },

                "summary":
                    normalized_summary,

                "issuetype": {
                    "name":
                        (
                            normalized_type
                            or "Task"
                        ),
                },
            }
        }

        try:

            response = (
                self.ticket_service
                .client
                .post(
                    "/rest/api/3/issue",
                    json=(
                        request_body
                    ),
                )
            )

        except httpx.HTTPError:

            return (
                unknown_mutation_result(
                    provider="jira",
                    operation=(
                        "create_ticket"
                    ),
                    error=(
                        "The Jira create request failed and "
                        "the remote mutation state is unknown."
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
                TicketMutationResult(
                    ok=False,
                    status="error",
                    provider="jira",
                    operation=(
                        "create_ticket"
                    ),
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
                TicketMutationResult(
                    ok=False,
                    status="denied",
                    provider="jira",
                    operation=(
                        "create_ticket"
                    ),
                    error=(
                        "Jira rejected the supplied "
                        "ticket creation fields."
                    ),
                )
            )

        if not (
            200
            <= response.status_code
            < 300
        ):

            if (
                response.status_code
                >= 500
            ):

                return (
                    unknown_mutation_result(
                        provider="jira",
                        operation=(
                            "create_ticket"
                        ),
                        error=(
                            "Jira returned HTTP "
                            f"{response.status_code} "
                            "while creating the ticket; "
                            "remote mutation state is unknown."
                        ),
                    )
                )

            return (
                TicketMutationResult(
                    ok=False,
                    status="error",
                    provider="jira",
                    operation=(
                        "create_ticket"
                    ),
                    error=(
                        "Jira returned HTTP "
                        f"{response.status_code} "
                        "while creating the ticket."
                    ),
                )
            )

        ticket_key = None

        try:

            payload = (
                response.json()
            )

        except ValueError:

            payload = None

        if isinstance(
            payload,
            dict,
        ):

            ticket_key = (
                self.ticket_service
                ._string_field(
                    payload.get(
                        "key"
                    )
                )
            )

        return (
            TicketMutationResult(
                ok=True,
                status="success",
                provider="jira",
                ticket_key=(
                    ticket_key
                ),
                operation=(
                    "create_ticket"
                ),
                changed=True,
                new_value=(
                    normalized_summary
                ),
                message=(
                    "Ticket created."
                ),
            )
        )

    def assign_ticket(
        self,
        ticket_key: str,
        assignee: str,
    ) -> TicketMutationResult:

        (
            normalized_key,
            key_error,
        ) = (
            self._normalized_ticket_key(
                ticket_key
            )
        )

        if normalized_key is None:

            return (
                TicketMutationResult(
                    ok=False,
                    status="denied",
                    provider="jira",
                    operation=(
                        "assign_ticket"
                    ),
                    error=(
                        key_error
                    ),
                )
            )

        (
            normalized_assignee,
            assignee_error,
        ) = (
            normalize_required_string(
                assignee,
                field_name=(
                    "assignee"
                ),
                max_length=(
                    MAX_TICKET_ASSIGNEE_LENGTH
                ),
            )
        )

        if normalized_assignee is None:

            return (
                TicketMutationResult(
                    ok=False,
                    status="denied",
                    provider="jira",
                    ticket_key=(
                        normalized_key
                    ),
                    operation=(
                        "assign_ticket"
                    ),
                    error=(
                        assignee_error
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
                self.ticket_service
                .client
                .put(
                    (
                        "/rest/api/3/"
                        f"issue/{encoded_key}/assignee"
                    ),

                    json={
                        "accountId":
                            normalized_assignee,
                    },
                )
            )

        except httpx.HTTPError:

            return (
                unknown_mutation_result(
                    provider="jira",
                    operation=(
                        "assign_ticket"
                    ),
                    ticket_key=(
                        normalized_key
                    ),
                    error=(
                        "The Jira assignment request failed and "
                        "the remote mutation state is unknown."
                    ),
                )
            )

        if (
            response.status_code
            == 404
        ):

            return (
                TicketMutationResult(
                    ok=False,
                    status="not_found",
                    provider="jira",
                    ticket_key=(
                        normalized_key
                    ),
                    operation=(
                        "assign_ticket"
                    ),
                    error=(
                        f"Ticket '{normalized_key}' "
                        "or assignee was not found."
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
                TicketMutationResult(
                    ok=False,
                    status="error",
                    provider="jira",
                    ticket_key=(
                        normalized_key
                    ),
                    operation=(
                        "assign_ticket"
                    ),
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
                TicketMutationResult(
                    ok=False,
                    status="denied",
                    provider="jira",
                    ticket_key=(
                        normalized_key
                    ),
                    operation=(
                        "assign_ticket"
                    ),
                    error=(
                        "Jira rejected the requested "
                        "ticket assignee."
                    ),
                )
            )

        if not (
            200
            <= response.status_code
            < 300
        ):

            if (
                response.status_code
                >= 500
            ):

                return (
                    unknown_mutation_result(
                        provider="jira",
                        operation=(
                            "assign_ticket"
                        ),
                        ticket_key=(
                            normalized_key
                        ),
                        error=(
                            "Jira returned HTTP "
                            f"{response.status_code} "
                            "while assigning the ticket; "
                            "remote mutation state is unknown."
                        ),
                    )
                )

            return (
                TicketMutationResult(
                    ok=False,
                    status="error",
                    provider="jira",
                    ticket_key=(
                        normalized_key
                    ),
                    operation=(
                        "assign_ticket"
                    ),
                    error=(
                        "Jira returned HTTP "
                        f"{response.status_code} "
                        "while assigning the ticket."
                    ),
                )
            )

        return (
            TicketMutationResult(
                ok=True,
                status="success",
                provider="jira",
                ticket_key=(
                    normalized_key
                ),
                operation=(
                    "assign_ticket"
                ),
                changed=True,
                new_value=(
                    normalized_assignee
                ),
                message=(
                    "Ticket assignee updated."
                ),
            )
        )

    def transition_ticket(
        self,
        ticket_key: str,
        status: str,
    ) -> TicketMutationResult:

        (
            normalized_key,
            key_error,
        ) = (
            self._normalized_ticket_key(
                ticket_key
            )
        )

        if normalized_key is None:

            return (
                TicketMutationResult(
                    ok=False,
                    status="denied",
                    provider="jira",
                    operation=(
                        "transition_ticket"
                    ),
                    error=(
                        key_error
                    ),
                )
            )

        (
            normalized_status,
            status_error,
        ) = (
            normalize_required_string(
                status,
                field_name=(
                    "status"
                ),
                max_length=(
                    MAX_TICKET_STATUS_LENGTH
                ),
            )
        )

        if normalized_status is None:

            return (
                TicketMutationResult(
                    ok=False,
                    status="denied",
                    provider="jira",
                    ticket_key=(
                        normalized_key
                    ),
                    operation=(
                        "transition_ticket"
                    ),
                    error=(
                        status_error
                    ),
                )
            )

        encoded_key = (
            quote(
                normalized_key,
                safe="",
            )
        )

        transitions_path = (
            "/rest/api/3/"
            f"issue/{encoded_key}/transitions"
        )

        # --------------------------------------------------------
        # READ AVAILABLE TRANSITIONS
        #
        # This request does not mutate state.
        # --------------------------------------------------------

        try:

            response = (
                self.ticket_service
                .client
                .get(
                    transitions_path
                )
            )

        except httpx.HTTPError:

            return (
                TicketMutationResult(
                    ok=False,
                    status="error",
                    provider="jira",
                    ticket_key=(
                        normalized_key
                    ),
                    operation=(
                        "transition_ticket"
                    ),
                    error=(
                        "Jira transition discovery failed."
                    ),
                )
            )

        if (
            response.status_code
            == 404
        ):

            return (
                TicketMutationResult(
                    ok=False,
                    status="not_found",
                    provider="jira",
                    ticket_key=(
                        normalized_key
                    ),
                    operation=(
                        "transition_ticket"
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
                TicketMutationResult(
                    ok=False,
                    status="error",
                    provider="jira",
                    ticket_key=(
                        normalized_key
                    ),
                    operation=(
                        "transition_ticket"
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
                TicketMutationResult(
                    ok=False,
                    status="error",
                    provider="jira",
                    ticket_key=(
                        normalized_key
                    ),
                    operation=(
                        "transition_ticket"
                    ),
                    error=(
                        "Jira returned HTTP "
                        f"{response.status_code} "
                        "while discovering transitions."
                    ),
                )
            )

        try:

            payload = (
                response.json()
            )

        except ValueError:

            payload = None

        if not isinstance(
            payload,
            dict,
        ):

            return (
                TicketMutationResult(
                    ok=False,
                    status="error",
                    provider="jira",
                    ticket_key=(
                        normalized_key
                    ),
                    operation=(
                        "transition_ticket"
                    ),
                    error=(
                        "Jira returned an invalid "
                        "transition response."
                    ),
                )
            )

        raw_transitions = (
            payload.get(
                "transitions"
            )
        )

        if not isinstance(
            raw_transitions,
            list,
        ):

            return (
                TicketMutationResult(
                    ok=False,
                    status="error",
                    provider="jira",
                    ticket_key=(
                        normalized_key
                    ),
                    operation=(
                        "transition_ticket"
                    ),
                    error=(
                        "Jira returned an invalid "
                        "transition list."
                    ),
                )
            )

        transition_id = None
        resolved_status = None

        requested = (
            normalized_status.casefold()
        )

        for transition in (
            raw_transitions
        ):

            if not isinstance(
                transition,
                dict,
            ):

                continue

            candidate_id = (
                self.ticket_service
                ._string_field(
                    transition.get(
                        "id"
                    )
                )
            )

            transition_name = (
                self.ticket_service
                ._string_field(
                    transition.get(
                        "name"
                    )
                )
            )

            target = (
                transition.get(
                    "to"
                )
            )

            target_name = None

            if isinstance(
                target,
                dict,
            ):

                target_name = (
                    self.ticket_service
                    ._string_field(
                        target.get(
                            "name"
                        )
                    )
                )

            names = {
                value.casefold()

                for value
                in (
                    transition_name,
                    target_name,
                )

                if value
                is not None
            }

            if (
                requested
                not in names
            ):

                continue

            if candidate_id is None:

                continue

            transition_id = (
                candidate_id
            )

            resolved_status = (
                target_name
                or transition_name
                or normalized_status
            )

            break

        if transition_id is None:

            return (
                TicketMutationResult(
                    ok=False,
                    status="denied",
                    provider="jira",
                    ticket_key=(
                        normalized_key
                    ),
                    operation=(
                        "transition_ticket"
                    ),
                    error=(
                        "The requested Jira status is not "
                        "available as a transition for this "
                        "ticket."
                    ),
                )
            )

        # --------------------------------------------------------
        # EXECUTE TRANSITION
        # --------------------------------------------------------

        try:

            response = (
                self.ticket_service
                .client
                .post(
                    transitions_path,

                    json={
                        "transition": {
                            "id":
                                transition_id,
                        }
                    },
                )
            )

        except httpx.HTTPError:

            return (
                unknown_mutation_result(
                    provider="jira",
                    operation=(
                        "transition_ticket"
                    ),
                    ticket_key=(
                        normalized_key
                    ),
                    error=(
                        "The Jira transition request failed and "
                        "the remote mutation state is unknown."
                    ),
                )
            )

        if (
            response.status_code
            in {
                400,
                404,
            }
        ):

            return (
                TicketMutationResult(
                    ok=False,
                    status="denied",
                    provider="jira",
                    ticket_key=(
                        normalized_key
                    ),
                    operation=(
                        "transition_ticket"
                    ),
                    error=(
                        "Jira rejected the requested "
                        "ticket transition."
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
                TicketMutationResult(
                    ok=False,
                    status="error",
                    provider="jira",
                    ticket_key=(
                        normalized_key
                    ),
                    operation=(
                        "transition_ticket"
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

            if (
                response.status_code
                >= 500
            ):

                return (
                    unknown_mutation_result(
                        provider="jira",
                        operation=(
                            "transition_ticket"
                        ),
                        ticket_key=(
                            normalized_key
                        ),
                        error=(
                            "Jira returned HTTP "
                            f"{response.status_code} "
                            "while transitioning the ticket; "
                            "remote mutation state is unknown."
                        ),
                    )
                )

            return (
                TicketMutationResult(
                    ok=False,
                    status="error",
                    provider="jira",
                    ticket_key=(
                        normalized_key
                    ),
                    operation=(
                        "transition_ticket"
                    ),
                    error=(
                        "Jira returned HTTP "
                        f"{response.status_code} "
                        "while transitioning the ticket."
                    ),
                )
            )

        return (
            TicketMutationResult(
                ok=True,
                status="success",
                provider="jira",
                ticket_key=(
                    normalized_key
                ),
                operation=(
                    "transition_ticket"
                ),
                changed=True,
                new_value=(
                    resolved_status
                ),
                message=(
                    "Ticket status updated."
                ),
            )
        )


def build_ticket_mutation_service(
    ticket_service: TicketService,
) -> TicketMutationService:
    """
    Build command handling from the already-selected query provider.

    Provider selection remains single-source-of-truth.
    """

    if isinstance(
        ticket_service,
        MockTicketService,
    ):

        return (
            MockTicketMutationService(
                ticket_service
            )
        )

    if isinstance(
        ticket_service,
        JiraTicketService,
    ):

        return (
            JiraTicketMutationService(
                ticket_service
            )
        )

    raise RuntimeError(
        "Unsupported TicketService implementation for "
        "ticket mutations: "
        f"{type(ticket_service).__name__}"
    )
