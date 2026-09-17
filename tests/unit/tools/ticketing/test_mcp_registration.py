from services.ticketing import (
    MockTicketService,
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
) -> FakeMCPServer:

    server = (
        FakeMCPServer()
    )

    service = (
        MockTicketService()
    )

    register_ticketing_tools(
        server,
        service,
    )

    return server


def test_all_ticketing_capabilities_are_registered():
    server = (
        build_registered_server()
    )

    assert set(
        server.functions
    ) == {
        "ticket_get",
        "ticket_search",
        "ticket_history",
        "ticket_comments",
    }


def test_ticket_history_mcp_execution():
    server = (
        build_registered_server()
    )

    result = (
        server.functions[
            "ticket_history"
        ](
            ticket_key="ITSM-101",
            limit=20,
        )
    )

    assert result.ok is True
    assert result.status == "success"
    assert result.ticket_key == "ITSM-101"
    assert result.count == 2


def test_ticket_comments_mcp_execution():
    server = (
        build_registered_server()
    )

    result = (
        server.functions[
            "ticket_comments"
        ](
            ticket_key="ITSM-101",
            limit=20,
        )
    )

    assert result.ok is True
    assert result.status == "success"
    assert result.ticket_key == "ITSM-101"
    assert result.count == 1


def test_ticket_search_mcp_execution():
    server = (
        build_registered_server()
    )

    result = (
        server.functions[
            "ticket_search"
        ](
            project_key="ITSM",
            status="In Progress",
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