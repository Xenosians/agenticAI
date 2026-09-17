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
    assert result.ticket_key == "ITSM-101"
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

    assert (
        result.tickets[
            0
        ].key
        == "ITSM-101"
    )


def test_ticket_add_comment_mcp_execution_updates_shared_state():

    (
        server,
        service,
    ) = (
        build_registered_server()
    )

    mutation = (
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

    assert mutation.ok is True
    assert mutation.status == "success"
    assert mutation.changed is True
    assert mutation.ticket_key == "ITSM-101"

    comments = (
        service
        .get_ticket_comments(
            "ITSM-101",
            limit=20,
        )
    )

    assert comments.ok is True
    assert comments.count == 2

    assert (
        comments.comments[
            -1
        ].body
        == "VPN access was verified."
    )
