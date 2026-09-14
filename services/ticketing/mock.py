from .base import (
    TicketService,
)

from .types import (
    TicketLookupResult,
    TicketRecord,
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

    def get_ticket(
        self,
        ticket_key: str,
    ) -> TicketLookupResult:

        if not isinstance(
            ticket_key,
            str,
        ):
            return (
                TicketLookupResult(
                    ok=False,

                    status="denied",

                    error=(
                        "ticket_key must "
                        "be a string."
                    ),
                )
            )

        normalized_key = (
            ticket_key
            .strip()
            .upper()
        )

        if not normalized_key:

            return (
                TicketLookupResult(
                    ok=False,

                    status="denied",

                    error=(
                        "ticket_key must not "
                        "be empty."
                    ),
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
