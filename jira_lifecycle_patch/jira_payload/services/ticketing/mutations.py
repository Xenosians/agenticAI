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

from .jira_comment_policy import (
    revalidate_jira_ticket_comment_snapshot,
)

from .jira_assignment_policy import (
    revalidate_jira_ticket_assignment_snapshot,
)

from .jira_create_policy import (
    DEFAULT_TICKET_TYPE,
    revalidate_jira_ticket_create_snapshot,
)

from .jira_issue_policy import (
    read_jira_issue_snapshot,
)

from .jira_transition_policy import (
    revalidate_jira_ticket_transition_snapshot,
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
        *,
        expected_issue_id: str | None = None,
        expected_issue_key: str | None = None,
        expected_issue_summary: str | None = None,
        expected_project_id: str | None = None,
        expected_project_key: str | None = None,
        expected_project_name: str | None = None,
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
        *,
        expected_issue_id: str | None = None,
        expected_issue_key: str | None = None,
        expected_issue_summary: str | None = None,
        expected_project_id: str | None = None,
        expected_project_key: str | None = None,
        expected_project_name: str | None = None,
        expected_previous_assignee_present: bool | None = None,
        expected_previous_assignee_account_id: str | None = None,
        expected_previous_assignee_display_name: str | None = None,
        expected_assignee_account_id: str | None = None,
        expected_assignee_display_name: str | None = None,
    ) -> TicketMutationResult:
        raise NotImplementedError

    @abstractmethod
    def transition_ticket(
        self,
        ticket_key: str,
        status: str,
        *,
        expected_issue_id: str | None = None,
        expected_issue_key: str | None = None,
        expected_issue_summary: str | None = None,
        expected_project_id: str | None = None,
        expected_project_key: str | None = None,
        expected_project_name: str | None = None,
        expected_source_status_id: str | None = None,
        expected_source_status_name: str | None = None,
        expected_transition_noop: bool | None = None,
        expected_transition_id: str | None = None,
        expected_transition_name: str | None = None,
        expected_target_status_id: str | None = None,
        expected_target_status_name: str | None = None,
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
        *,
        expected_issue_id: str | None = None,
        expected_issue_key: str | None = None,
        expected_issue_summary: str | None = None,
        expected_project_id: str | None = None,
        expected_project_key: str | None = None,
        expected_project_name: str | None = None,
    ) -> TicketMutationResult:

        trusted_snapshot_values = (
            expected_issue_id,
            expected_issue_key,
            expected_issue_summary,
            expected_project_id,
            expected_project_key,
            expected_project_name,
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
                        "add_comment"
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
        *,
        expected_issue_id: str | None = None,
        expected_issue_key: str | None = None,
        expected_issue_summary: str | None = None,
        expected_project_id: str | None = None,
        expected_project_key: str | None = None,
        expected_project_name: str | None = None,
        expected_previous_assignee_present: bool | None = None,
        expected_previous_assignee_account_id: str | None = None,
        expected_previous_assignee_display_name: str | None = None,
        expected_assignee_account_id: str | None = None,
        expected_assignee_display_name: str | None = None,
    ) -> TicketMutationResult:

        trusted_snapshot_values = (
            expected_issue_id, expected_issue_key, expected_issue_summary,
            expected_project_id, expected_project_key, expected_project_name,
            expected_previous_assignee_present,
            expected_previous_assignee_account_id,
            expected_previous_assignee_display_name,
            expected_assignee_account_id, expected_assignee_display_name,
        )

        if any(value is not None for value in trusted_snapshot_values):
            return TicketMutationResult(
                ok=False, status="denied", provider="mock",
                operation="assign_ticket", changed=False,
                mutation_performed=False, verification_ok=False,
                reconciled=False,
                error=("Jira provider snapshot arguments cannot be used "
                       "with the mock ticket provider."),
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
        *,
        expected_issue_id: str | None = None,
        expected_issue_key: str | None = None,
        expected_issue_summary: str | None = None,
        expected_project_id: str | None = None,
        expected_project_key: str | None = None,
        expected_project_name: str | None = None,
        expected_source_status_id: str | None = None,
        expected_source_status_name: str | None = None,
        expected_transition_noop: bool | None = None,
        expected_transition_id: str | None = None,
        expected_transition_name: str | None = None,
        expected_target_status_id: str | None = None,
        expected_target_status_name: str | None = None,
    ) -> TicketMutationResult:

        trusted_snapshot_values = (
            expected_issue_id, expected_issue_key, expected_issue_summary,
            expected_project_id, expected_project_key, expected_project_name,
            expected_source_status_id, expected_source_status_name,
            expected_transition_noop, expected_transition_id,
            expected_transition_name, expected_target_status_id,
            expected_target_status_name,
        )

        if any(value is not None for value in trusted_snapshot_values):
            return TicketMutationResult(
                ok=False, status="denied", provider="mock",
                operation="transition_ticket", changed=False,
                mutation_performed=False, verification_ok=False,
                reconciled=False,
                error=("Jira provider snapshot arguments cannot be used "
                       "with the mock ticket provider."),
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
            comment.split("\n")
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

    @staticmethod
    def _exact_comment_string(
        value: object,
        *,
        field_name: str,
        max_length: int,
    ) -> str:
        """Validate without rewriting user-supplied comment text."""

        if not isinstance(
            value,
            str,
        ):
            raise ValueError(
                f"{field_name} must be a string."
            )

        if not value.strip():
            raise ValueError(
                f"{field_name} must not be empty."
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

    @classmethod
    def _comment_adf_exact_text(
        cls,
        value: object,
    ) -> str:
        """Recover exact plain text from the ADF shape used for comments."""

        if isinstance(
            value,
            str,
        ):
            return value

        if isinstance(
            value,
            list,
        ):
            return "".join(
                cls._comment_adf_exact_text(item)
                for item in value
            )

        if not isinstance(
            value,
            dict,
        ):
            return ""

        node_type = value.get("type")

        if node_type == "text":
            text = value.get("text")
            return text if isinstance(text, str) else ""

        if node_type == "hardBreak":
            return "\n"

        content = value.get("content")
        if not isinstance(content, list):
            content = []

        if node_type == "doc":
            return "\n".join(
                cls._comment_adf_exact_text(item)
                for item in content
            )

        return "".join(
            cls._comment_adf_exact_text(item)
            for item in content
        )

    def _read_comment_snapshot(
        self,
        *,
        ticket_key: str,
        comment_id: str,
    ) -> dict[
        str,
        str,
    ]:
        encoded_key = quote(
            ticket_key,
            safe="",
        )

        encoded_comment_id = quote(
            comment_id,
            safe="",
        )

        try:
            response = (
                self.ticket_service
                .client
                .get(
                    (
                        "/rest/api/3/issue/"
                        f"{encoded_key}/comment/"
                        f"{encoded_comment_id}"
                    )
                )
            )

        except (
            httpx.TimeoutException,
            httpx.TransportError,
        ) as exc:
            raise RuntimeError(
                "Jira comment read-back failed."
            ) from exc

        if response.status_code != 200:
            raise RuntimeError(
                "Jira comment read-back returned HTTP "
                f"{response.status_code}."
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                "Jira comment read-back returned invalid JSON."
            ) from exc

        if not isinstance(
            payload,
            dict,
        ):
            raise RuntimeError(
                "Jira comment read-back returned invalid data."
            )

        returned_id = (
            self._provider_identifier(
                payload.get("id"),
                field_name="comment ID",
            )
        )

        body = (
            self._comment_adf_exact_text(
                payload.get("body")
            )
        )

        return {
            "comment_id": returned_id,
            "body": body,
        }

    @staticmethod
    def _comment_outcome_unknown(
        *,
        ticket_key: str,
        comment_id: str | None = None,
        mutation_performed: bool | None = None,
        error: str,
    ) -> TicketMutationResult:
        return (
            TicketMutationResult(
                ok=False,
                status="outcome_unknown",
                provider="jira",
                ticket_key=ticket_key,
                operation="add_comment",
                changed=False,
                mutation_performed=mutation_performed,
                verification_ok=False,
                reconciled=False,
                comment_id=comment_id,
                error=error,
            )
        )

    def add_comment(
        self,
        ticket_key: str,
        comment: str,
        *,
        expected_issue_id: str | None = None,
        expected_issue_key: str | None = None,
        expected_issue_summary: str | None = None,
        expected_project_id: str | None = None,
        expected_project_key: str | None = None,
        expected_project_name: str | None = None,
    ) -> TicketMutationResult:
        """
        Execute one exact approved Jira comment mutation.

        A complete pre-approval provider snapshot is mandatory. The issue is
        revalidated immediately before the POST, the POST is sent exactly once,
        and the returned comment ID is read back for exact body verification.
        """

        try:
            resolved_key = (
                self._exact_create_string(
                    ticket_key,
                    field_name="ticket_key",
                    max_length=100,
                )
            )

            (
                normalized_key,
                key_error,
            ) = self._normalized_ticket_key(
                resolved_key
            )

            if normalized_key is None:
                raise ValueError(
                    key_error
                    or "ticket_key is not a valid Jira issue key."
                )

            if normalized_key != resolved_key:
                raise ValueError(
                    "ticket_key must already use the exact canonical Jira "
                    "issue key; trusted execution will not rewrite it."
                )

            resolved_comment = (
                self._exact_comment_string(
                    comment,
                    field_name="comment",
                    max_length=MAX_TICKET_COMMENT_LENGTH,
                )
            )

            trusted_values = {
                "expected_issue_id": expected_issue_id,
                "expected_issue_key": expected_issue_key,
                "expected_issue_summary": expected_issue_summary,
                "expected_project_id": expected_project_id,
                "expected_project_key": expected_project_key,
                "expected_project_name": expected_project_name,
            }

            if any(
                value is None
                for value in trusted_values.values()
            ):
                raise ValueError(
                    "Approved Jira comment requires a complete trusted "
                    "provider snapshot."
                )

            resolved_expected_issue_id = (
                self._exact_create_string(
                    expected_issue_id,
                    field_name="expected_issue_id",
                    max_length=100,
                )
            )

            resolved_expected_issue_key = (
                self._exact_create_string(
                    expected_issue_key,
                    field_name="expected_issue_key",
                    max_length=100,
                )
            )

            resolved_expected_issue_summary = (
                self._exact_create_string(
                    expected_issue_summary,
                    field_name="expected_issue_summary",
                    max_length=5000,
                )
            )

            resolved_expected_project_id = (
                self._exact_create_string(
                    expected_project_id,
                    field_name="expected_project_id",
                    max_length=100,
                )
            )

            resolved_expected_project_key = (
                self._exact_create_string(
                    expected_project_key,
                    field_name="expected_project_key",
                    max_length=100,
                )
            )

            resolved_expected_project_name = (
                self._exact_create_string(
                    expected_project_name,
                    field_name="expected_project_name",
                    max_length=255,
                )
            )

            if resolved_expected_issue_key != resolved_key:
                raise ValueError(
                    "Approved Jira issue snapshot does not match the "
                    "persisted ticket key."
                )

            revalidate_jira_ticket_comment_snapshot(
                self.ticket_service,
                expected_issue_id=resolved_expected_issue_id,
                expected_issue_key=resolved_expected_issue_key,
                expected_issue_summary=resolved_expected_issue_summary,
                expected_project_id=resolved_expected_project_id,
                expected_project_key=resolved_expected_project_key,
                expected_project_name=resolved_expected_project_name,
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
                    ticket_key=(
                        ticket_key
                        if isinstance(ticket_key, str)
                        else None
                    ),
                    operation="add_comment",
                    changed=False,
                    mutation_performed=False,
                    verification_ok=False,
                    reconciled=False,
                    error=str(exc),
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
                    ticket_key=(
                        ticket_key
                        if isinstance(ticket_key, str)
                        else None
                    ),
                    operation="add_comment",
                    changed=False,
                    mutation_performed=False,
                    verification_ok=False,
                    reconciled=False,
                    error=str(exc),
                )
            )

        encoded_key = quote(
            resolved_key,
            safe="",
        )

        request_body = {
            "body": self._comment_adf(
                resolved_comment
            )
        }

        try:
            response = (
                self.ticket_service
                .client
                .post(
                    (
                        "/rest/api/3/issue/"
                        f"{encoded_key}/comment"
                    ),
                    json=request_body,
                )
            )

        except (
            httpx.TimeoutException,
            httpx.TransportError,
        ):
            return (
                self._comment_outcome_unknown(
                    ticket_key=resolved_key,
                    mutation_performed=None,
                    error=(
                        "The Jira comment request encountered an ambiguous "
                        "transport failure. Automatic retry is disabled."
                    ),
                )
            )

        if response.status_code in {
            400,
            401,
            403,
            404,
            422,
        }:
            return (
                TicketMutationResult(
                    ok=False,
                    status="denied",
                    provider="jira",
                    ticket_key=resolved_key,
                    operation="add_comment",
                    changed=False,
                    mutation_performed=False,
                    verification_ok=False,
                    reconciled=False,
                    error=(
                        "Jira rejected the approved ticket comment request."
                    ),
                )
            )

        if response.status_code == 429:
            return (
                TicketMutationResult(
                    ok=False,
                    status="rate_limited",
                    provider="jira",
                    ticket_key=resolved_key,
                    operation="add_comment",
                    changed=False,
                    mutation_performed=False,
                    verification_ok=False,
                    reconciled=False,
                    error=(
                        "Jira rate limited the approved ticket comment request."
                    ),
                )
            )

        if response.status_code >= 500:
            return (
                self._comment_outcome_unknown(
                    ticket_key=resolved_key,
                    mutation_performed=None,
                    error=(
                        "Jira returned HTTP "
                        f"{response.status_code} after the comment request "
                        "was sent. Automatic retry is disabled."
                    ),
                )
            )

        if response.status_code != 201:
            return (
                self._comment_outcome_unknown(
                    ticket_key=resolved_key,
                    mutation_performed=None,
                    error=(
                        "Jira returned unexpected HTTP "
                        f"{response.status_code} for ticket comment creation. "
                        "The mutation outcome is not trusted."
                    ),
                )
            )

        try:
            payload = response.json()
        except ValueError:
            payload = None

        if not isinstance(
            payload,
            dict,
        ):
            return (
                self._comment_outcome_unknown(
                    ticket_key=resolved_key,
                    mutation_performed=True,
                    error=(
                        "Jira reported successful comment creation, but the "
                        "response could not identify the created comment for "
                        "verification."
                    ),
                )
            )

        try:
            comment_id = (
                self._provider_identifier(
                    payload.get("id"),
                    field_name="comment ID",
                )
            )
        except RuntimeError:
            return (
                self._comment_outcome_unknown(
                    ticket_key=resolved_key,
                    mutation_performed=True,
                    error=(
                        "Jira reported successful comment creation, but did "
                        "not return a valid comment ID."
                    ),
                )
            )

        try:
            current = self._read_comment_snapshot(
                ticket_key=resolved_key,
                comment_id=comment_id,
            )
        except Exception:
            return (
                self._comment_outcome_unknown(
                    ticket_key=resolved_key,
                    comment_id=comment_id,
                    mutation_performed=True,
                    error=(
                        "Jira created the comment, but trusted read-back "
                        "verification could not confirm the resulting state."
                    ),
                )
            )

        expected = {
            "comment_id": comment_id,
            "body": resolved_comment,
        }

        if current != expected:
            return (
                self._comment_outcome_unknown(
                    ticket_key=resolved_key,
                    comment_id=comment_id,
                    mutation_performed=True,
                    error=(
                        "Jira created a comment, but its read-back state does "
                        "not exactly match the approved mutation."
                    ),
                )
            )

        return (
            TicketMutationResult(
                ok=True,
                status="success",
                provider="jira",
                ticket_key=resolved_key,
                operation="add_comment",
                changed=True,
                mutation_performed=True,
                verification_ok=True,
                reconciled=False,
                comment_id=comment_id,
                new_value=resolved_comment,
                message="Ticket comment added and verified.",
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

    @staticmethod
    def _jira_issue_outcome_unknown(
        *,
        operation: str,
        ticket_key: str,
        mutation_performed: bool | None,
        previous_value: str | None = None,
        new_value: str | None = None,
        error: str,
    ) -> TicketMutationResult:
        return TicketMutationResult(
            ok=False,
            status="outcome_unknown",
            provider="jira",
            ticket_key=ticket_key,
            operation=operation,
            changed=False,
            mutation_performed=mutation_performed,
            verification_ok=False,
            reconciled=False,
            previous_value=previous_value,
            new_value=new_value,
            error=error,
        )

    def assign_ticket(
        self,
        ticket_key: str,
        assignee: str,
        *,
        expected_issue_id: str | None = None,
        expected_issue_key: str | None = None,
        expected_issue_summary: str | None = None,
        expected_project_id: str | None = None,
        expected_project_key: str | None = None,
        expected_project_name: str | None = None,
        expected_previous_assignee_present: bool | None = None,
        expected_previous_assignee_account_id: str | None = None,
        expected_previous_assignee_display_name: str | None = None,
        expected_assignee_account_id: str | None = None,
        expected_assignee_display_name: str | None = None,
    ) -> TicketMutationResult:
        """Execute one exact approved Jira assignment and verify read-back."""

        try:
            resolved_key = self._exact_create_string(
                ticket_key,
                field_name="ticket_key",
                max_length=100,
            )
            normalized_key, key_error = self._normalized_ticket_key(resolved_key)
            if normalized_key is None:
                raise ValueError(key_error or "Invalid Jira ticket key.")
            if normalized_key != resolved_key:
                raise ValueError(
                    "ticket_key must already use the exact canonical Jira issue key."
                )

            resolved_assignee = self._exact_create_string(
                assignee,
                field_name="assignee",
                max_length=MAX_TICKET_ASSIGNEE_LENGTH,
            )

            required = (
                expected_issue_id,
                expected_issue_key,
                expected_issue_summary,
                expected_project_id,
                expected_project_key,
                expected_project_name,
                expected_previous_assignee_present,
                expected_assignee_account_id,
            )
            if any(value is None for value in required):
                raise ValueError(
                    "Approved Jira assignment requires a complete trusted provider snapshot."
                )

            if expected_issue_key != resolved_key:
                raise ValueError(
                    "Approved Jira issue snapshot does not match the persisted ticket key."
                )

            validated = revalidate_jira_ticket_assignment_snapshot(
                self.ticket_service,
                expected_issue_id=expected_issue_id,
                expected_issue_key=expected_issue_key,
                expected_issue_summary=expected_issue_summary,
                expected_project_id=expected_project_id,
                expected_project_key=expected_project_key,
                expected_project_name=expected_project_name,
                expected_previous_assignee_present=expected_previous_assignee_present,
                expected_previous_assignee_account_id=(
                    expected_previous_assignee_account_id
                ),
                expected_previous_assignee_display_name=(
                    expected_previous_assignee_display_name
                ),
                expected_assignee_account_id=expected_assignee_account_id,
                expected_assignee_display_name=expected_assignee_display_name,
            )

            previous_account_id = validated["issue"]["assignee_account_id"]
            target_account_id = validated["target"]["account_id"]

        except (PermissionError, LookupError, ValueError) as exc:
            return TicketMutationResult(
                ok=False,
                status="denied",
                provider="jira",
                ticket_key=ticket_key if isinstance(ticket_key, str) else None,
                operation="assign_ticket",
                changed=False,
                mutation_performed=False,
                verification_ok=False,
                reconciled=False,
                error=str(exc),
            )
        except (RuntimeError, httpx.HTTPError) as exc:
            return TicketMutationResult(
                ok=False,
                status="error",
                provider="jira",
                ticket_key=ticket_key if isinstance(ticket_key, str) else None,
                operation="assign_ticket",
                changed=False,
                mutation_performed=False,
                verification_ok=False,
                reconciled=False,
                error=str(exc),
            )

        if previous_account_id == target_account_id:
            return TicketMutationResult(
                ok=True,
                status="success",
                provider="jira",
                ticket_key=resolved_key,
                operation="assign_ticket",
                changed=False,
                mutation_performed=False,
                verification_ok=True,
                reconciled=False,
                previous_value=previous_account_id,
                new_value=target_account_id,
                message="Ticket already has the approved assignee; state verified.",
            )

        encoded_key = quote(resolved_key, safe="")

        try:
            response = self.ticket_service.client.put(
                f"/rest/api/3/issue/{encoded_key}/assignee",
                json={"accountId": target_account_id},
            )
        except (httpx.TimeoutException, httpx.TransportError):
            return self._jira_issue_outcome_unknown(
                operation="assign_ticket",
                ticket_key=resolved_key,
                mutation_performed=None,
                previous_value=previous_account_id,
                new_value=target_account_id,
                error=(
                    "The Jira assignment request encountered an ambiguous transport "
                    "failure. Automatic retry is disabled."
                ),
            )

        if response.status_code in {400, 401, 403, 404, 422}:
            return TicketMutationResult(
                ok=False,
                status="denied",
                provider="jira",
                ticket_key=resolved_key,
                operation="assign_ticket",
                changed=False,
                mutation_performed=False,
                verification_ok=False,
                reconciled=False,
                previous_value=previous_account_id,
                new_value=target_account_id,
                error="Jira rejected the approved assignment request.",
            )

        if response.status_code == 429:
            return TicketMutationResult(
                ok=False,
                status="rate_limited",
                provider="jira",
                ticket_key=resolved_key,
                operation="assign_ticket",
                changed=False,
                mutation_performed=False,
                verification_ok=False,
                reconciled=False,
                previous_value=previous_account_id,
                new_value=target_account_id,
                error="Jira rate limited the approved assignment request.",
            )

        if response.status_code >= 500:
            return self._jira_issue_outcome_unknown(
                operation="assign_ticket",
                ticket_key=resolved_key,
                mutation_performed=None,
                previous_value=previous_account_id,
                new_value=target_account_id,
                error=(
                    f"Jira returned HTTP {response.status_code} after the assignment "
                    "request was sent. Automatic retry is disabled."
                ),
            )

        if response.status_code != 204:
            return self._jira_issue_outcome_unknown(
                operation="assign_ticket",
                ticket_key=resolved_key,
                mutation_performed=None,
                previous_value=previous_account_id,
                new_value=target_account_id,
                error=(
                    f"Jira returned unexpected HTTP {response.status_code} for "
                    "assignment. The mutation outcome is not trusted."
                ),
            )

        try:
            current = read_jira_issue_snapshot(
                self.ticket_service,
                resolved_key,
                include_assignee=True,
                operation="assignment read-back verification",
            )
        except Exception:
            return self._jira_issue_outcome_unknown(
                operation="assign_ticket",
                ticket_key=resolved_key,
                mutation_performed=True,
                previous_value=previous_account_id,
                new_value=target_account_id,
                error=(
                    "Jira accepted the assignment, but trusted read-back verification "
                    "could not confirm the resulting state."
                ),
            )

        expected_readback = {
            "issue_id": expected_issue_id,
            "issue_key": expected_issue_key,
            "issue_summary": expected_issue_summary,
            "project_id": expected_project_id,
            "project_key": expected_project_key,
            "project_name": expected_project_name,
            "assignee_present": True,
            "assignee_account_id": target_account_id,
            "assignee_display_name": expected_assignee_display_name,
        }

        if current != expected_readback:
            return self._jira_issue_outcome_unknown(
                operation="assign_ticket",
                ticket_key=resolved_key,
                mutation_performed=True,
                previous_value=previous_account_id,
                new_value=target_account_id,
                error=(
                    "Jira accepted the assignment, but its read-back state does not "
                    "exactly match the approved mutation."
                ),
            )

        return TicketMutationResult(
            ok=True,
            status="success",
            provider="jira",
            ticket_key=resolved_key,
            operation="assign_ticket",
            changed=True,
            mutation_performed=True,
            verification_ok=True,
            reconciled=False,
            previous_value=previous_account_id,
            new_value=target_account_id,
            message=(
                f"Ticket assigned and verified for {resolved_assignee}."
            ),
        )

    def transition_ticket(
        self,
        ticket_key: str,
        status: str,
        *,
        expected_issue_id: str | None = None,
        expected_issue_key: str | None = None,
        expected_issue_summary: str | None = None,
        expected_project_id: str | None = None,
        expected_project_key: str | None = None,
        expected_project_name: str | None = None,
        expected_source_status_id: str | None = None,
        expected_source_status_name: str | None = None,
        expected_transition_noop: bool | None = None,
        expected_transition_id: str | None = None,
        expected_transition_name: str | None = None,
        expected_target_status_id: str | None = None,
        expected_target_status_name: str | None = None,
    ) -> TicketMutationResult:
        """Execute one exact approved Jira transition and verify read-back."""

        try:
            resolved_key = self._exact_create_string(
                ticket_key,
                field_name="ticket_key",
                max_length=100,
            )
            normalized_key, key_error = self._normalized_ticket_key(resolved_key)
            if normalized_key is None:
                raise ValueError(key_error or "Invalid Jira ticket key.")
            if normalized_key != resolved_key:
                raise ValueError(
                    "ticket_key must already use the exact canonical Jira issue key."
                )

            resolved_status = self._exact_create_string(
                status,
                field_name="status",
                max_length=MAX_TICKET_STATUS_LENGTH,
            )

            required = (
                expected_issue_id,
                expected_issue_key,
                expected_issue_summary,
                expected_project_id,
                expected_project_key,
                expected_project_name,
                expected_source_status_id,
                expected_source_status_name,
                expected_transition_noop,
                expected_target_status_id,
                expected_target_status_name,
            )
            if any(value is None for value in required):
                raise ValueError(
                    "Approved Jira transition requires a complete trusted provider snapshot."
                )

            if expected_issue_key != resolved_key:
                raise ValueError(
                    "Approved Jira issue snapshot does not match the persisted ticket key."
                )

            if expected_target_status_name.casefold() != resolved_status.casefold():
                raise ValueError(
                    "Approved Jira transition target does not match the persisted status request."
                )

            validated = revalidate_jira_ticket_transition_snapshot(
                self.ticket_service,
                expected_issue_id=expected_issue_id,
                expected_issue_key=expected_issue_key,
                expected_issue_summary=expected_issue_summary,
                expected_project_id=expected_project_id,
                expected_project_key=expected_project_key,
                expected_project_name=expected_project_name,
                expected_source_status_id=expected_source_status_id,
                expected_source_status_name=expected_source_status_name,
                expected_transition_noop=expected_transition_noop,
                expected_transition_id=expected_transition_id,
                expected_transition_name=expected_transition_name,
                expected_target_status_id=expected_target_status_id,
                expected_target_status_name=expected_target_status_name,
            )

        except (PermissionError, LookupError, ValueError) as exc:
            return TicketMutationResult(
                ok=False,
                status="denied",
                provider="jira",
                ticket_key=ticket_key if isinstance(ticket_key, str) else None,
                operation="transition_ticket",
                changed=False,
                mutation_performed=False,
                verification_ok=False,
                reconciled=False,
                error=str(exc),
            )
        except (RuntimeError, httpx.HTTPError) as exc:
            return TicketMutationResult(
                ok=False,
                status="error",
                provider="jira",
                ticket_key=ticket_key if isinstance(ticket_key, str) else None,
                operation="transition_ticket",
                changed=False,
                mutation_performed=False,
                verification_ok=False,
                reconciled=False,
                error=str(exc),
            )

        if expected_transition_noop:
            return TicketMutationResult(
                ok=True,
                status="success",
                provider="jira",
                ticket_key=resolved_key,
                operation="transition_ticket",
                changed=False,
                mutation_performed=False,
                verification_ok=True,
                reconciled=False,
                previous_value=expected_source_status_name,
                new_value=expected_target_status_name,
                message="Ticket already has the approved status; state verified.",
            )

        transition = validated["transition"]
        transition_id = transition["transition_id"]
        encoded_key = quote(resolved_key, safe="")

        try:
            response = self.ticket_service.client.post(
                f"/rest/api/3/issue/{encoded_key}/transitions",
                json={"transition": {"id": transition_id}},
            )
        except (httpx.TimeoutException, httpx.TransportError):
            return self._jira_issue_outcome_unknown(
                operation="transition_ticket",
                ticket_key=resolved_key,
                mutation_performed=None,
                previous_value=expected_source_status_name,
                new_value=expected_target_status_name,
                error=(
                    "The Jira transition request encountered an ambiguous transport "
                    "failure. Automatic retry is disabled."
                ),
            )

        if response.status_code in {400, 401, 403, 404, 422}:
            return TicketMutationResult(
                ok=False,
                status="denied",
                provider="jira",
                ticket_key=resolved_key,
                operation="transition_ticket",
                changed=False,
                mutation_performed=False,
                verification_ok=False,
                reconciled=False,
                previous_value=expected_source_status_name,
                new_value=expected_target_status_name,
                error="Jira rejected the approved transition request.",
            )

        if response.status_code == 429:
            return TicketMutationResult(
                ok=False,
                status="rate_limited",
                provider="jira",
                ticket_key=resolved_key,
                operation="transition_ticket",
                changed=False,
                mutation_performed=False,
                verification_ok=False,
                reconciled=False,
                previous_value=expected_source_status_name,
                new_value=expected_target_status_name,
                error="Jira rate limited the approved transition request.",
            )

        if response.status_code >= 500:
            return self._jira_issue_outcome_unknown(
                operation="transition_ticket",
                ticket_key=resolved_key,
                mutation_performed=None,
                previous_value=expected_source_status_name,
                new_value=expected_target_status_name,
                error=(
                    f"Jira returned HTTP {response.status_code} after the transition "
                    "request was sent. Automatic retry is disabled."
                ),
            )

        if response.status_code != 204:
            return self._jira_issue_outcome_unknown(
                operation="transition_ticket",
                ticket_key=resolved_key,
                mutation_performed=None,
                previous_value=expected_source_status_name,
                new_value=expected_target_status_name,
                error=(
                    f"Jira returned unexpected HTTP {response.status_code} for the "
                    "transition. The mutation outcome is not trusted."
                ),
            )

        try:
            current = read_jira_issue_snapshot(
                self.ticket_service,
                resolved_key,
                include_status=True,
                operation="transition read-back verification",
            )
        except Exception:
            return self._jira_issue_outcome_unknown(
                operation="transition_ticket",
                ticket_key=resolved_key,
                mutation_performed=True,
                previous_value=expected_source_status_name,
                new_value=expected_target_status_name,
                error=(
                    "Jira accepted the transition, but trusted read-back verification "
                    "could not confirm the resulting state."
                ),
            )

        expected_readback = {
            "issue_id": expected_issue_id,
            "issue_key": expected_issue_key,
            "issue_summary": expected_issue_summary,
            "project_id": expected_project_id,
            "project_key": expected_project_key,
            "project_name": expected_project_name,
            "status_id": expected_target_status_id,
            "status_name": expected_target_status_name,
        }

        if current != expected_readback:
            return self._jira_issue_outcome_unknown(
                operation="transition_ticket",
                ticket_key=resolved_key,
                mutation_performed=True,
                previous_value=expected_source_status_name,
                new_value=expected_target_status_name,
                error=(
                    "Jira accepted the transition, but its read-back state does not "
                    "exactly match the approved mutation."
                ),
            )

        return TicketMutationResult(
            ok=True,
            status="success",
            provider="jira",
            ticket_key=resolved_key,
            operation="transition_ticket",
            changed=True,
            mutation_performed=True,
            verification_ok=True,
            reconciled=False,
            previous_value=expected_source_status_name,
            new_value=expected_target_status_name,
            message="Ticket status updated and verified.",
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
