from pydantic import (
    BaseModel,
    Field,
    ValidationError,
)

from mcp.server import (
    MCPServer,
)

from services.ticketing import (
    TicketMutationService,
    TicketSearchQuery,
    TicketService,
)


# ============================================================
# TICKET RECORD
# ============================================================


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


# ============================================================
# TICKET SEARCH
# ============================================================


class TicketSearchMCPResult(
    BaseModel
):
    ok: bool
    status: str

    tickets: list[
        TicketRecordResult
    ] = Field(
        default_factory=list
    )

    count: int = 0

    truncated: bool = False

    error: (
        str | None
    ) = None


# ============================================================
# TICKET HISTORY
# ============================================================


class TicketFieldChangeResult(
    BaseModel
):
    field: str

    from_value: (
        str | None
    ) = None

    to_value: (
        str | None
    ) = None


class TicketHistoryEntryResult(
    BaseModel
):
    id: (
        str | None
    ) = None

    author: (
        str | None
    ) = None

    created_at: (
        str | None
    ) = None

    changes: list[
        TicketFieldChangeResult
    ] = Field(
        default_factory=list
    )


class TicketHistoryMCPResult(
    BaseModel
):
    ok: bool
    status: str

    provider: (
        str | None
    ) = None

    ticket_key: (
        str | None
    ) = None

    history: list[
        TicketHistoryEntryResult
    ] = Field(
        default_factory=list
    )

    count: int = 0

    truncated: bool = False

    error: (
        str | None
    ) = None


# ============================================================
# TICKET COMMENTS
# ============================================================


class TicketCommentResult(
    BaseModel
):
    id: str

    author: (
        str | None
    ) = None

    body: str

    created_at: (
        str | None
    ) = None

    updated_at: (
        str | None
    ) = None


class TicketCommentsMCPResult(
    BaseModel
):
    ok: bool
    status: str

    provider: (
        str | None
    ) = None

    ticket_key: (
        str | None
    ) = None

    comments: list[
        TicketCommentResult
    ] = Field(
        default_factory=list
    )

    count: int = 0

    truncated: bool = False

    error: (
        str | None
    ) = None


# ============================================================
# TICKET MUTATION
# ============================================================


class TicketMutationMCPResult(
    BaseModel
):
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

    message: (
        str | None
    ) = None

    error: (
        str | None
    ) = None


# ============================================================
# REGISTRATION
# ============================================================


def register_ticketing_tools(
    server: MCPServer,
    ticket_service: TicketService,
    ticket_mutations: (
        TicketMutationService
        | None
    ) = None,
) -> None:
    """
    Register provider-neutral ticketing capabilities.

    Query capabilities use TicketService.

    Command capabilities use TicketMutationService.

    Jira URLs, credentials, HTTP methods, authentication,
    provider-native query languages, and provider internals remain
    behind the trusted service boundary.

    Mutation authorization is NOT decided here.

    Model-facing execution reaches mutation MCP tools only after
    ToolGateway and approval handling have accepted the exact tool
    invocation.
    """

    # --------------------------------------------------------
    # GET ONE TICKET
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # SEARCH TICKETS
    # --------------------------------------------------------

    @server.tool()
    def ticket_search(
        text: str | None = None,
        project_key: str | None = None,
        status: str | None = None,
        priority: str | None = None,
        limit: int | None = None,
    ) -> TicketSearchMCPResult:

        try:

            query = (
                TicketSearchQuery(
                    text=(
                        text
                    ),

                    project_key=(
                        project_key
                    ),

                    status=(
                        status
                    ),

                    priority=(
                        priority
                    ),

                    limit=(
                        10
                        if limit is None
                        else limit
                    ),
                )
            )

        except ValidationError:

            return (
                TicketSearchMCPResult(
                    ok=False,

                    status="denied",

                    error=(
                        "Invalid ticket "
                        "search filters."
                    ),
                )
            )

        result = (
            ticket_service
            .search_tickets(
                query
            )
        )

        return (
            TicketSearchMCPResult(
                **result.model_dump()
            )
        )

    # --------------------------------------------------------
    # TICKET HISTORY
    # --------------------------------------------------------

    @server.tool()
    def ticket_history(
        ticket_key: str,
        limit: int | None = None,
    ) -> TicketHistoryMCPResult:

        result = (
            ticket_service
            .get_ticket_history(
                ticket_key,

                limit=(
                    20
                    if limit is None
                    else limit
                ),
            )
        )

        return (
            TicketHistoryMCPResult(
                **result.model_dump()
            )
        )

    # --------------------------------------------------------
    # READ TICKET COMMENTS
    # --------------------------------------------------------

    @server.tool()
    def ticket_comments(
        ticket_key: str,
        limit: int | None = None,
    ) -> TicketCommentsMCPResult:

        result = (
            ticket_service
            .get_ticket_comments(
                ticket_key,

                limit=(
                    20
                    if limit is None
                    else limit
                ),
            )
        )

        return (
            TicketCommentsMCPResult(
                **result.model_dump()
            )
        )

    # --------------------------------------------------------
    # MUTATION CAPABILITIES
    #
    # Kept optional for isolated legacy registration tests.
    #
    # Production MCP construction always supplies the mutation
    # service.
    # --------------------------------------------------------

    if (
        ticket_mutations
        is not None
    ):

        @server.tool()
        def ticket_add_comment(
            ticket_key: str,
            comment: str,
        ) -> TicketMutationMCPResult:

            result = (
                ticket_mutations
                .add_comment(
                    ticket_key,
                    comment,
                )
            )

            return (
                TicketMutationMCPResult(
                    **result.model_dump()
                )
            )
