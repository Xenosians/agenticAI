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


class TicketRecordResult(
    BaseModel
):
    provider: str
    key: str
    summary: str
    status: str

    ticket_type: str | None = None
    priority: str | None = None
    assignee: str | None = None
    reporter: str | None = None
    project_key: str | None = None
    project_name: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class TicketLookupMCPResult(
    BaseModel
):
    ok: bool
    status: str
    ticket: TicketRecordResult | None = None
    error: str | None = None


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
    error: str | None = None


class TicketFieldChangeResult(
    BaseModel
):
    field: str
    from_value: str | None = None
    to_value: str | None = None


class TicketHistoryEntryResult(
    BaseModel
):
    id: str | None = None
    author: str | None = None
    created_at: str | None = None

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
    provider: str | None = None
    ticket_key: str | None = None

    history: list[
        TicketHistoryEntryResult
    ] = Field(
        default_factory=list
    )

    count: int = 0
    truncated: bool = False
    error: str | None = None


class TicketCommentResult(
    BaseModel
):
    id: str
    author: str | None = None
    body: str
    created_at: str | None = None
    updated_at: str | None = None


class TicketCommentsMCPResult(
    BaseModel
):
    ok: bool
    status: str
    provider: str | None = None
    ticket_key: str | None = None

    comments: list[
        TicketCommentResult
    ] = Field(
        default_factory=list
    )

    count: int = 0
    truncated: bool = False
    error: str | None = None


class TicketMutationMCPResult(
    BaseModel
):
    ok: bool
    status: str
    provider: str | None = None
    ticket_key: str | None = None
    operation: str | None = None
    changed: bool = False
    mutation_performed: bool | None = None
    verification_ok: bool = False
    reconciled: bool = False
    comment_id: str | None = None
    previous_value: str | None = None
    new_value: str | None = None
    message: str | None = None
    error: str | None = None


def register_ticketing_tools(
    server: MCPServer,
    ticket_service: TicketService,
    ticket_mutations: (
        TicketMutationService
        | None
    ) = None,
) -> None:

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

    if (
        ticket_mutations
        is None
    ):

        return

    @server.tool()
    def ticket_add_comment(
        ticket_key: str,
        comment: str,
        expected_issue_id: str | None = None,
        expected_issue_key: str | None = None,
        expected_issue_summary: str | None = None,
        expected_project_id: str | None = None,
        expected_project_key: str | None = None,
        expected_project_name: str | None = None,
    ) -> TicketMutationMCPResult:

        result = (
            ticket_mutations
            .add_comment(
                ticket_key,
                comment,
                expected_issue_id=(
                    expected_issue_id
                ),
                expected_issue_key=(
                    expected_issue_key
                ),
                expected_issue_summary=(
                    expected_issue_summary
                ),
                expected_project_id=(
                    expected_project_id
                ),
                expected_project_key=(
                    expected_project_key
                ),
                expected_project_name=(
                    expected_project_name
                ),
            )
        )

        return (
            TicketMutationMCPResult(
                **result.model_dump()
            )
        )

    @server.tool()
    def ticket_create(
        project_key: str,
        summary: str,
        ticket_type: str | None = None,
        expected_project_id: str | None = None,
        expected_project_key: str | None = None,
        expected_project_name: str | None = None,
        expected_ticket_type_id: str | None = None,
        expected_ticket_type_name: str | None = None,
    ) -> TicketMutationMCPResult:

        result = (
            ticket_mutations
            .create_ticket(
                project_key,
                summary,
                ticket_type=(
                    ticket_type
                ),
                expected_project_id=(
                    expected_project_id
                ),
                expected_project_key=(
                    expected_project_key
                ),
                expected_project_name=(
                    expected_project_name
                ),
                expected_ticket_type_id=(
                    expected_ticket_type_id
                ),
                expected_ticket_type_name=(
                    expected_ticket_type_name
                ),
            )
        )

        return (
            TicketMutationMCPResult(
                **result.model_dump()
            )
        )

    @server.tool()
    def ticket_assign(
        ticket_key: str,
        assignee: str,
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
    ) -> TicketMutationMCPResult:

        result = ticket_mutations.assign_ticket(
            ticket_key,
            assignee,
            expected_issue_id=expected_issue_id,
            expected_issue_key=expected_issue_key,
            expected_issue_summary=expected_issue_summary,
            expected_project_id=expected_project_id,
            expected_project_key=expected_project_key,
            expected_project_name=expected_project_name,
            expected_previous_assignee_present=expected_previous_assignee_present,
            expected_previous_assignee_account_id=expected_previous_assignee_account_id,
            expected_previous_assignee_display_name=expected_previous_assignee_display_name,
            expected_assignee_account_id=expected_assignee_account_id,
            expected_assignee_display_name=expected_assignee_display_name,
        )

        return TicketMutationMCPResult(**result.model_dump())

    @server.tool()
    def ticket_transition(
        ticket_key: str,
        status: str,
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
    ) -> TicketMutationMCPResult:

        result = ticket_mutations.transition_ticket(
            ticket_key,
            status,
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

        return TicketMutationMCPResult(**result.model_dump())
