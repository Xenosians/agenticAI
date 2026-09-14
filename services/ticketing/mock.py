from .base import (
    TicketService,
)

from .types import (
    TicketComment,
    TicketCommentsResult,
    TicketFieldChange,
    TicketHistoryEntry,
    TicketHistoryResult,
    TicketLookupResult,
    TicketRecord,
    TicketSearchQuery,
    TicketSearchResult,
)


class MockTicketService(
    TicketService
):

    def __init__(
        self,
    ) -> None:

        self.tickets = {
            "ITSM-101": (
                TicketRecord(
                    provider="mock",

                    key="ITSM-101",

                    summary=(
                        "VPN access unavailable "
                        "for new employee"
                    ),

                    status="In Progress",

                    ticket_type="Task",

                    priority="High",

                    assignee="Service Desk",

                    reporter="jdoe",

                    project_key="ITSM",

                    project_name=(
                        "IT Service Management"
                    ),

                    created_at=(
                        "2026-09-01T09:00:00Z"
                    ),

                    updated_at=(
                        "2026-09-14T10:30:00Z"
                    ),
                )
            ),

            "OPS-42": (
                TicketRecord(
                    provider="mock",

                    key="OPS-42",

                    summary=(
                        "Development API service "
                        "is unhealthy"
                    ),

                    status="Open",

                    ticket_type="Incident",

                    priority="Medium",

                    assignee=None,

                    reporter="monitoring",

                    project_key="OPS",

                    project_name="Operations",

                    created_at=(
                        "2026-09-13T18:00:00Z"
                    ),

                    updated_at=(
                        "2026-09-13T18:00:00Z"
                    ),
                )
            ),
        }

        self.history = {
            "ITSM-101": [
                TicketHistoryEntry(
                    id="1",
                    author="Service Desk",
                    created_at=(
                        "2026-09-01T09:10:00Z"
                    ),
                    changes=[
                        TicketFieldChange(
                            field="status",
                            from_value="To Do",
                            to_value="In Progress",
                        ),
                    ],
                ),

                TicketHistoryEntry(
                    id="2",
                    author="Service Desk",
                    created_at=(
                        "2026-09-01T09:15:00Z"
                    ),
                    changes=[
                        TicketFieldChange(
                            field="priority",
                            from_value="Medium",
                            to_value="High",
                        ),
                    ],
                ),
            ],

            "OPS-42": [],
        }

        self.comments = {
            "ITSM-101": [
                TicketComment(
                    id="1001",
                    author="Service Desk",
                    body=(
                        "Investigating VPN access "
                        "for the new employee."
                    ),
                    created_at=(
                        "2026-09-01T09:20:00Z"
                    ),
                    updated_at=(
                        "2026-09-01T09:20:00Z"
                    ),
                )
            ],

            "OPS-42": [],
        }

    @staticmethod
    def _normalize_key(
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
                "ticket_key must be a string.",
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

        return (
            normalized,
            None,
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

    def get_ticket(
        self,
        ticket_key: str,
    ) -> TicketLookupResult:

        (
            normalized_key,
            key_error,
        ) = self._normalize_key(
            ticket_key
        )

        if normalized_key is None:
            return (
                TicketLookupResult(
                    ok=False,
                    status="denied",
                    error=key_error,
                )
            )

        ticket = (
            self.tickets.get(
                normalized_key
            )
        )

        if ticket is None:
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

        tickets = list(
            self.tickets.values()
        )

        if query.text is not None:
            search_text = (
                query.text.casefold()
            )

            tickets = [
                ticket

                for ticket
                in tickets

                if (
                    search_text
                    in ticket.key.casefold()
                    or search_text
                    in ticket.summary.casefold()
                )
            ]

        if query.project_key is not None:
            project_key = (
                query.project_key.casefold()
            )

            tickets = [
                ticket

                for ticket
                in tickets

                if (
                    ticket.project_key
                    is not None
                    and ticket.project_key.casefold()
                    == project_key
                )
            ]

        if query.status is not None:
            status = (
                query.status.casefold()
            )

            tickets = [
                ticket

                for ticket
                in tickets

                if (
                    ticket.status.casefold()
                    == status
                )
            ]

        if query.priority is not None:
            priority = (
                query.priority.casefold()
            )

            tickets = [
                ticket

                for ticket
                in tickets

                if (
                    ticket.priority
                    is not None
                    and ticket.priority.casefold()
                    == priority
                )
            ]

        tickets.sort(
            key=lambda ticket: (
                ticket.updated_at
                or ""
            ),
            reverse=True,
        )

        truncated = (
            len(
                tickets
            )
            > query.limit
        )

        selected = (
            tickets[
                :query.limit
            ]
        )

        return (
            TicketSearchResult(
                ok=True,
                status="success",
                tickets=selected,
                count=len(
                    selected
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
        ) = self._normalize_key(
            ticket_key
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
        ) = self._normalize_limit(
            limit
        )

        if normalized_limit is None:
            return (
                TicketHistoryResult(
                    ok=False,
                    status="denied",
                    error=limit_error,
                )
            )

        if (
            normalized_key
            not in self.tickets
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

        history = (
            self.history.get(
                normalized_key,
                [],
            )
        )

        truncated = (
            len(
                history
            )
            > normalized_limit
        )

        selected = (
            history[
                :normalized_limit
            ]
        )

        return (
            TicketHistoryResult(
                ok=True,
                status="success",
                provider="mock",
                ticket_key=(
                    normalized_key
                ),
                history=selected,
                count=len(
                    selected
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
        ) = self._normalize_key(
            ticket_key
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
        ) = self._normalize_limit(
            limit
        )

        if normalized_limit is None:
            return (
                TicketCommentsResult(
                    ok=False,
                    status="denied",
                    error=limit_error,
                )
            )

        if (
            normalized_key
            not in self.tickets
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

        comments = (
            self.comments.get(
                normalized_key,
                [],
            )
        )

        truncated = (
            len(
                comments
            )
            > normalized_limit
        )

        selected = (
            comments[
                :normalized_limit
            ]
        )

        return (
            TicketCommentsResult(
                ok=True,
                status="success",
                provider="mock",
                ticket_key=(
                    normalized_key
                ),
                comments=selected,
                count=len(
                    selected
                ),
                truncated=truncated,
                error=None,
            )
        )