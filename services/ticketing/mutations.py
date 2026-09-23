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

from .jira_create_policy import (
    DEFAULT_TICKET_TYPE,
    revalidate_jira_ticket_create_snapshot,
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

    mutation_performed: (
        bool
        | None
    ) = None

    verification_ok: bool = False

    reconciled: bool = False

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
        expected_project_id: str | None = None,
        expected_project_key: str | None = None,
        expected_project_name: str | None = None,
        expected_ticket_type_id: str | None = None,
        expected_ticket_type_name: str | None = None,
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
        expected_project_id: str | None = None,
        expected_project_key: str | None = None,
        expected_project_name: str | None = None,
        expected_ticket_type_id: str | None = None,
        expected_ticket_type_name: str | None = None,
    ) -> TicketMutationResult:

        trusted_snapshot_values = (
            expected_project_id,
            expected_project_key,
            expected_project_name,
            expected_ticket_type_id,
            expected_ticket_type_name,
        )

        if any(
            value is not None
            for value
            in trusted_snapshot_values
        ):
            return (
                TicketMutationResult(
                    ok=False,
                    status="denied",
                    provider="mock",
                    operation=(
                        "create_ticket"
                    ),
                    changed=False,
                    mutation_performed=False,
                    verification_ok=False,
                    reconciled=False,
                    error=(
                        "Jira provider snapshot arguments "
                        "cannot be used with the mock "
                        "ticket provider."
                    ),
                )
            )

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

    @staticmethod
    def _exact_create_string(
        value: object,
        *,
        field_name: str,
        max_length: int,
    ) -> str:

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

    @staticmethod
    def _provider_identifier(
        value: object,
        *,
        field_name: str,
    ) -> str:

        if (
            isinstance(
                value,
                int,
            )
            and not isinstance(
                value,
                bool,
            )
        ):
            value = str(
                value
            )

        if not isinstance(
            value,
            str,
        ):
            raise RuntimeError(
                f"Jira returned invalid {field_name}."
            )

        if (
            not value
            or value
            != value.strip()
        ):
            raise RuntimeError(
                f"Jira returned invalid {field_name}."
            )

        return value

    def _read_created_ticket_snapshot(
        self,
        ticket_key: str,
    ) -> dict[
        str,
        str,
    ]:
        """
        Read the exact provider state created by ticket_create.

        This is verification only. It never performs a mutation.
        """

        encoded_key = (
            quote(
                ticket_key,
                safe="",
            )
        )

        try:
            response = (
                self.ticket_service
                .client
                .get(
                    (
                        "/rest/api/3/issue/"
                        f"{encoded_key}"
                    ),
                    params={
                        "fields":
                            (
                                "summary,"
                                "project,"
                                "issuetype"
                            ),
                    },
                )
            )

        except (
            httpx.TimeoutException,
            httpx.TransportError,
        ) as exc:
            raise RuntimeError(
                "Jira ticket read-back failed."
            ) from exc

        if (
            response.status_code
            != 200
        ):
            raise RuntimeError(
                "Jira ticket read-back returned HTTP "
                f"{response.status_code}."
            )

        try:
            payload = (
                response.json()
            )

        except ValueError as exc:
            raise RuntimeError(
                "Jira ticket read-back returned invalid JSON."
            ) from exc

        if not isinstance(
            payload,
            dict,
        ):
            raise RuntimeError(
                "Jira ticket read-back returned invalid data."
            )

        returned_key = (
            self._provider_identifier(
                payload.get(
                    "key"
                ),
                field_name=(
                    "ticket key"
                ),
            )
        )

        fields = (
            payload.get(
                "fields"
            )
        )

        if not isinstance(
            fields,
            dict,
        ):
            raise RuntimeError(
                "Jira ticket read-back did not contain fields."
            )

        summary = (
            self._provider_identifier(
                fields.get(
                    "summary"
                ),
                field_name=(
                    "ticket summary"
                ),
            )
        )

        project = (
            fields.get(
                "project"
            )
        )

        if not isinstance(
            project,
            dict,
        ):
            raise RuntimeError(
                "Jira ticket read-back did not contain "
                "valid project metadata."
            )

        issue_type = (
            fields.get(
                "issuetype"
            )
        )

        if not isinstance(
            issue_type,
            dict,
        ):
            raise RuntimeError(
                "Jira ticket read-back did not contain "
                "valid issue-type metadata."
            )

        return {
            "ticket_key":
                returned_key,

            "summary":
                summary,

            "project_id":
                self._provider_identifier(
                    project.get(
                        "id"
                    ),
                    field_name=(
                        "project ID"
                    ),
                ),

            "project_key":
                self._provider_identifier(
                    project.get(
                        "key"
                    ),
                    field_name=(
                        "project key"
                    ),
                ),

            "project_name":
                self._provider_identifier(
                    project.get(
                        "name"
                    ),
                    field_name=(
                        "project name"
                    ),
                ),

            "ticket_type_id":
                self._provider_identifier(
                    issue_type.get(
                        "id"
                    ),
                    field_name=(
                        "issue-type ID"
                    ),
                ),

            "ticket_type_name":
                self._provider_identifier(
                    issue_type.get(
                        "name"
                    ),
                    field_name=(
                        "issue-type name"
                    ),
                ),
        }

    @staticmethod
    def _create_outcome_unknown(
        *,
        summary: str,
        ticket_key: str | None = None,
        mutation_performed: bool | None = None,
        error: str,
    ) -> TicketMutationResult:

        return (
            TicketMutationResult(
                ok=False,
                status=(
                    "outcome_unknown"
                ),
                provider="jira",
                ticket_key=(
                    ticket_key
                ),
                operation=(
                    "create_ticket"
                ),
                changed=False,
                mutation_performed=(
                    mutation_performed
                ),
                verification_ok=False,
                reconciled=False,
                new_value=(
                    summary
                ),
                error=(
                    error
                ),
            )
        )

    def create_ticket(
        self,
        project_key: str,
        summary: str,
        *,
        ticket_type: str | None = None,
        expected_project_id: str | None = None,
        expected_project_key: str | None = None,
        expected_project_name: str | None = None,
        expected_ticket_type_id: str | None = None,
        expected_ticket_type_name: str | None = None,
    ) -> TicketMutationResult:
        """
        Execute one exact approved Jira issue creation.

        Jira creation without a trusted pre-approval provider
        snapshot fails closed.

        The POST is never blindly retried.
        """

        try:
            resolved_project_key = (
                self._exact_create_string(
                    project_key,
                    field_name=(
                        "project_key"
                    ),
                    max_length=(
                        MAX_PROJECT_KEY_LENGTH
                    ),
                )
            )

            resolved_summary = (
                self._exact_create_string(
                    summary,
                    field_name=(
                        "summary"
                    ),
                    max_length=(
                        MAX_TICKET_SUMMARY_LENGTH
                    ),
                )
            )

            if ticket_type is None:
                resolved_ticket_type = (
                    DEFAULT_TICKET_TYPE
                )

            else:
                resolved_ticket_type = (
                    self._exact_create_string(
                        ticket_type,
                        field_name=(
                            "ticket_type"
                        ),
                        max_length=(
                            MAX_TICKET_TYPE_LENGTH
                        ),
                    )
                )

            trusted_values = {
                "expected_project_id":
                    expected_project_id,

                "expected_project_key":
                    expected_project_key,

                "expected_project_name":
                    expected_project_name,

                "expected_ticket_type_id":
                    expected_ticket_type_id,

                "expected_ticket_type_name":
                    expected_ticket_type_name,
            }

            if any(
                value is None
                for value
                in trusted_values.values()
            ):
                raise ValueError(
                    "Approved Jira ticket creation requires "
                    "a complete trusted provider snapshot."
                )

            resolved_expected_project_id = (
                self._exact_create_string(
                    expected_project_id,
                    field_name=(
                        "expected_project_id"
                    ),
                    max_length=100,
                )
            )

            if not (
                resolved_expected_project_id
                .isdigit()
            ):
                raise ValueError(
                    "Trusted Jira project ID must be numeric."
                )

            resolved_expected_project_key = (
                self._exact_create_string(
                    expected_project_key,
                    field_name=(
                        "expected_project_key"
                    ),
                    max_length=(
                        MAX_PROJECT_KEY_LENGTH
                    ),
                )
            )

            resolved_expected_project_name = (
                self._exact_create_string(
                    expected_project_name,
                    field_name=(
                        "expected_project_name"
                    ),
                    max_length=255,
                )
            )

            resolved_expected_type_id = (
                self._exact_create_string(
                    expected_ticket_type_id,
                    field_name=(
                        "expected_ticket_type_id"
                    ),
                    max_length=100,
                )
            )

            resolved_expected_type_name = (
                self._exact_create_string(
                    expected_ticket_type_name,
                    field_name=(
                        "expected_ticket_type_name"
                    ),
                    max_length=(
                        MAX_TICKET_TYPE_LENGTH
                    ),
                )
            )

            # The provider snapshot may add trusted identity,
            # but it may not alter the persisted semantic request.
            if (
                resolved_expected_project_key
                != resolved_project_key
            ):
                raise ValueError(
                    "Approved Jira project snapshot does not "
                    "match the persisted project key."
                )

            if (
                resolved_expected_type_name
                != resolved_ticket_type
            ):
                raise ValueError(
                    "Approved Jira issue-type snapshot does not "
                    "match the persisted ticket type."
                )

            # =================================================
            # POST-APPROVAL PROVIDER REVALIDATION
            # =================================================

            revalidate_jira_ticket_create_snapshot(
                self.ticket_service,
                expected_project_id=(
                    resolved_expected_project_id
                ),
                expected_project_key=(
                    resolved_expected_project_key
                ),
                expected_project_name=(
                    resolved_expected_project_name
                ),
                expected_ticket_type_id=(
                    resolved_expected_type_id
                ),
                expected_ticket_type_name=(
                    resolved_expected_type_name
                ),
            )

        except (
            PermissionError,
            LookupError,
            ValueError,
        ) as exc:

            return (
                TicketMutationResult(
                    ok=False,
                    status="denied",
                    provider="jira",
                    operation=(
                        "create_ticket"
                    ),
                    changed=False,
                    mutation_performed=False,
                    verification_ok=False,
                    reconciled=False,
                    error=str(
                        exc
                    ),
                )
            )

        except (
            RuntimeError,
            httpx.HTTPError,
        ) as exc:

            return (
                TicketMutationResult(
                    ok=False,
                    status="error",
                    provider="jira",
                    operation=(
                        "create_ticket"
                    ),
                    changed=False,
                    mutation_performed=False,
                    verification_ok=False,
                    reconciled=False,
                    error=str(
                        exc
                    ),
                )
            )

        request_body = {
            "fields": {
                "project": {
                    "id":
                        resolved_expected_project_id,
                },

                "summary":
                    resolved_summary,

                "issuetype": {
                    "id":
                        resolved_expected_type_id,
                },
            }
        }

        # =====================================================
        # EXACTLY ONE MUTATING REQUEST
        # =====================================================

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

        except (
            httpx.TimeoutException,
            httpx.TransportError,
        ):

            return (
                self._create_outcome_unknown(
                    summary=(
                        resolved_summary
                    ),
                    mutation_performed=None,
                    error=(
                        "The Jira create request encountered "
                        "an ambiguous transport failure. "
                        "Automatic retry is disabled."
                    ),
                )
            )

        # Explicit provider rejection means the requested create
        # was not accepted as successful.
        if (
            response.status_code
            in {
                400,
                401,
                403,
                404,
                422,
            }
        ):

            return (
                TicketMutationResult(
                    ok=False,
                    status="denied",
                    provider="jira",
                    operation=(
                        "create_ticket"
                    ),
                    changed=False,
                    mutation_performed=False,
                    verification_ok=False,
                    reconciled=False,
                    new_value=(
                        resolved_summary
                    ),
                    error=(
                        "Jira rejected the approved "
                        "ticket creation request."
                    ),
                )
            )

        if (
            response.status_code
            == 429
        ):

            return (
                TicketMutationResult(
                    ok=False,
                    status=(
                        "rate_limited"
                    ),
                    provider="jira",
                    operation=(
                        "create_ticket"
                    ),
                    changed=False,
                    mutation_performed=False,
                    verification_ok=False,
                    reconciled=False,
                    new_value=(
                        resolved_summary
                    ),
                    error=(
                        "Jira rate limited the approved "
                        "ticket creation request."
                    ),
                )
            )

        if (
            response.status_code
            >= 500
        ):

            return (
                self._create_outcome_unknown(
                    summary=(
                        resolved_summary
                    ),
                    mutation_performed=None,
                    error=(
                        "Jira returned HTTP "
                        f"{response.status_code} after the "
                        "create request was sent. "
                        "Automatic retry is disabled."
                    ),
                )
            )

        # Jira Cloud documents 201 for successful issue creation.
        if (
            response.status_code
            != 201
        ):

            return (
                self._create_outcome_unknown(
                    summary=(
                        resolved_summary
                    ),
                    mutation_performed=None,
                    error=(
                        "Jira returned unexpected HTTP "
                        f"{response.status_code} for issue "
                        "creation. The mutation outcome is "
                        "not trusted."
                    ),
                )
            )

        try:
            payload = (
                response.json()
            )

        except ValueError:

            return (
                self._create_outcome_unknown(
                    summary=(
                        resolved_summary
                    ),
                    mutation_performed=True,
                    error=(
                        "Jira reported successful creation, "
                        "but the response could not identify "
                        "the created ticket for verification."
                    ),
                )
            )

        if not isinstance(
            payload,
            dict,
        ):

            return (
                self._create_outcome_unknown(
                    summary=(
                        resolved_summary
                    ),
                    mutation_performed=True,
                    error=(
                        "Jira reported successful creation, "
                        "but returned invalid create metadata."
                    ),
                )
            )

        raw_ticket_key = (
            payload.get(
                "key"
            )
        )

        if not isinstance(
            raw_ticket_key,
            str,
        ):

            return (
                self._create_outcome_unknown(
                    summary=(
                        resolved_summary
                    ),
                    mutation_performed=True,
                    error=(
                        "Jira reported successful creation, "
                        "but did not return a valid ticket key."
                    ),
                )
            )

        (
            created_ticket_key,
            key_error,
        ) = (
            self.ticket_service
            ._normalize_ticket_key(
                raw_ticket_key
            )
        )

        if (
            created_ticket_key
            is None
        ):

            return (
                self._create_outcome_unknown(
                    summary=(
                        resolved_summary
                    ),
                    mutation_performed=True,
                    error=(
                        key_error
                        or (
                            "Jira returned an invalid "
                            "created ticket key."
                        )
                    ),
                )
            )

        # =====================================================
        # TRUSTED READ-BACK VERIFICATION
        # =====================================================

        try:
            current = (
                self._read_created_ticket_snapshot(
                    created_ticket_key
                )
            )

        except Exception:

            return (
                self._create_outcome_unknown(
                    summary=(
                        resolved_summary
                    ),
                    ticket_key=(
                        created_ticket_key
                    ),
                    mutation_performed=True,
                    error=(
                        "Jira created the ticket, but trusted "
                        "read-back verification could not "
                        "confirm the resulting state."
                    ),
                )
            )

        expected = {
            "ticket_key":
                created_ticket_key,

            "summary":
                resolved_summary,

            "project_id":
                resolved_expected_project_id,

            "project_key":
                resolved_expected_project_key,

            "project_name":
                resolved_expected_project_name,

            "ticket_type_id":
                resolved_expected_type_id,

            "ticket_type_name":
                resolved_expected_type_name,
        }

        if (
            current
            != expected
        ):

            return (
                self._create_outcome_unknown(
                    summary=(
                        resolved_summary
                    ),
                    ticket_key=(
                        created_ticket_key
                    ),
                    mutation_performed=True,
                    error=(
                        "Jira created a ticket, but its "
                        "read-back state does not exactly "
                        "match the approved mutation."
                    ),
                )
            )

        return (
            TicketMutationResult(
                ok=True,
                status="success",
                provider="jira",
                ticket_key=(
                    created_ticket_key
                ),
                operation=(
                    "create_ticket"
                ),
                changed=True,
                mutation_performed=True,
                verification_ok=True,
                reconciled=False,
                new_value=(
                    resolved_summary
                ),
                message=(
                    "Ticket created and verified."
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
