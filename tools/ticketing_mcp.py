from pydantic import (
    BaseModel,
)

from mcp.server import (
    MCPServer,
)

from services.ticketing import (
    TicketService,
)


class TicketRecordResult(
    BaseModel
):
    provider: str

    key: str
    summary: str
    status: str

    ticket_type: (
        str | None
    ) = None

    priority: (
        str | None
    ) = None

    assignee: (
        str | None
    ) = None

    reporter: (
        str | None
    ) = None

    project_key: (
        str | None
    ) = None

    project_name: (
        str | None
    ) = None

    created_at: (
        str | None
    ) = None

    updated_at: (
        str | None
    ) = None


class TicketLookupMCPResult(
    BaseModel
):
    ok: bool
    status: str

    ticket: (
        TicketRecordResult
        | None
    ) = None

    error: (
        str | None
    ) = None


def register_ticketing_tools(
    server: MCPServer,
    ticket_service: TicketService,
) -> None:
    """
    Register provider-neutral ticketing capabilities.

    Model-facing tools do not expose Jira URLs, credentials,
    HTTP methods, JQL, authentication details, or provider
    internals.
    """

    @server.tool()
    def ticket_get(
        ticket_key: str,
    ) -> TicketLookupMCPResult:
        result = (
            ticket_service
            .get_ticket(
                ticket_key
            )
        )

        return (
            TicketLookupMCPResult(
                **result.model_dump()
            )
        )