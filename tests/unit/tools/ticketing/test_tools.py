from services.ticketing import (
    MockTicketService,
)

from tools.registry import (
    get_tool,
)

from tools.ticketing.mcp import (
    register_ticketing_tools,
)

from tools.ticketing.presentation import (
    build_ticket_get_card,
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


def test_ticket_get_catalog_is_read_only():
    tool = (
        get_tool(
            "ticket_get"
        )
    )

    assert tool is not None

    assert (
        tool[
            "risk"
        ]
        == "read"
    )

    assert (
        tool[
            "requires_approval"
        ]
        is False
    )

    assert (
        tool[
            "grounded_arguments"
        ]
        == [
            "ticket_key"
        ]
    )


def test_ticket_get_mcp_returns_mock_ticket():
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

    result = (
        server.functions[
            "ticket_get"
        ](
            "ITSM-101"
        )
    )

    assert result.ok is True

    assert (
        result.status
        == "success"
    )

    assert (
        result.ticket
        is not None
    )

    assert (
        result.ticket.key
        == "ITSM-101"
    )

    assert (
        result.ticket.provider
        == "mock"
    )


def test_ticket_get_builds_result_card():
    service = (
        MockTicketService()
    )

    lookup = (
        service.get_ticket(
            "ITSM-101"
        )
    )

    card = (
        build_ticket_get_card(
            lookup.model_dump()
        )
    )

    assert (
        card[
            "schema"
        ]
        == "result-card.v1"
    )

    assert (
        card[
            "kind"
        ]
        == "ticket"
    )

    assert (
        card[
            "title"
        ]
        == "ITSM-101"
    )

    assert (
        card[
            "status"
        ]
        == "success"
    )

    field_values = {
        field[
            "label"
        ]:
            field[
                "value"
            ]

        for field
        in card[
            "fields"
        ]
    }

    assert (
        field_values[
            "Status"
        ]
        == "In Progress"
    )

    assert (
        field_values[
            "Provider"
        ]
        == "mock"
    )


def test_ticket_get_unknown_ticket_is_not_found():
    server = (
        FakeMCPServer()
    )

    register_ticketing_tools(
        server,
        MockTicketService(),
    )

    result = (
        server.functions[
            "ticket_get"
        ](
            "ITSM-999"
        )
    )

    assert result.ok is False

    assert (
        result.status
        == "not_found"
    )

    assert (
        result.ticket
        is None
    )

    assert (
        result.error
        is not None
    )