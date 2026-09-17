from services.ticketing import (
    MockTicketService,
    build_ticket_mutation_service,
)

from tools.ticketing.mcp import (
    register_ticketing_tools,
)


class FakeMCPServer:

    def __init__(
        self,
    ) -> None:

        self.functions = {}

    def tool(
        self,
    ):

        def decorator(
            function,
        ):

            self.functions[
                function.__name__
            ] = function

            return function

        return decorator


def build_registered_server(
) -> tuple[
    FakeMCPServer,
    MockTicketService,
]:

    server = (
        FakeMCPServer()
    )

    service = (
        MockTicketService()
    )

    mutations = (
        build_ticket_mutation_service(
            service
        )
    )

    register_ticketing_tools(
        server,
        service,
        mutations,
    )

    return (
        server,
        service,
    )


def test_all_ticketing_capabilities_are_registered():

    (
        server,
        _,
    ) = (
        build_registered_server()
    )

    assert set(
        server.functions
    ) == {
        "ticket_get",
        "ticket_search",
        "ticket_history",
        "ticket_comments",
        "ticket_add_comment",
        "ticket_create",
        "ticket_assign",
        "ticket_transition",
    }


def test_ticket_history_mcp_execution():

    (
        server,
        _,
    ) = (
        build_registered_server()
    )

    result = (
        server.functions[
            "ticket_history"
        ](
            ticket_key=(
                "ITSM-101"
            ),
            limit=20,
        )
    )

    assert result.ok is True
    assert result.status == "success"
    assert result.ticket_key == "ITSM-101"
    assert result.count == 2


def test_ticket_comments_mcp_execution():

    (
        server,
        _,
    ) = (
        build_registered_server()
    )

    result = (
        server.functions[
            "ticket_comments"
        ](
            ticket_key=(
                "ITSM-101"
            ),
            limit=20,
        )
    )

    assert result.ok is True
    assert result.status == "success"
    assert result.count == 1


def test_ticket_search_mcp_execution():

    (
        server,
        _,
    ) = (
        build_registered_server()
    )

    result = (
        server.functions[
            "ticket_search"
        ](
            project_key=(
                "ITSM"
            ),
            status=(
                "In Progress"
            ),
        )
    )

    assert result.ok is True
    assert result.status == "success"
    assert result.count == 1


def test_ticket_add_comment_updates_shared_state():

    (
        server,
        service,
    ) = (
        build_registered_server()
    )

    result = (
        server.functions[
            "ticket_add_comment"
        ](
            ticket_key=(
                "ITSM-101"
            ),
            comment=(
                "VPN access was verified."
            ),
        )
    )

    assert result.ok is True
    assert result.changed is True

    comments = (
        service
        .get_ticket_comments(
            "ITSM-101"
        )
    )

    assert comments.count == 2


def test_ticket_create_is_visible_to_reads():

    (
        server,
        service,
    ) = (
        build_registered_server()
    )

    result = (
        server.functions[
            "ticket_create"
        ](
            project_key=(
                "ITSM"
            ),
            summary=(
                "Laptop onboarding failure"
            ),
        )
    )

    assert result.ok is True
    assert result.changed is True
    assert result.ticket_key == "ITSM-102"

    lookup = (
        service.get_ticket(
            "ITSM-102"
        )
    )

    assert lookup.ok is True
    assert lookup.ticket is not None
    assert (
        lookup.ticket.summary
        == "Laptop onboarding failure"
    )


def test_ticket_assign_is_visible_to_reads():

    (
        server,
        service,
    ) = (
        build_registered_server()
    )

    result = (
        server.functions[
            "ticket_assign"
        ](
            ticket_key=(
                "ITSM-101"
            ),
            assignee=(
                "alice"
            ),
        )
    )

    assert result.ok is True
    assert result.changed is True

    lookup = (
        service.get_ticket(
            "ITSM-101"
        )
    )

    assert lookup.ticket is not None
    assert lookup.ticket.assignee == "alice"


def test_ticket_transition_is_visible_to_reads():

    (
        server,
        service,
    ) = (
        build_registered_server()
    )

    result = (
        server.functions[
            "ticket_transition"
        ](
            ticket_key=(
                "ITSM-101"
            ),
            status=(
                "Resolved"
            ),
        )
    )

    assert result.ok is True
    assert result.changed is True

    lookup = (
        service.get_ticket(
            "ITSM-101"
        )
    )

    assert lookup.ticket is not None
    assert lookup.ticket.status == "Resolved"
