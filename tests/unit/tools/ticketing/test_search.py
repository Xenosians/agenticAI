import httpx

from services.ticketing import (
    JiraTicketService,
    MockTicketService,
    TicketSearchQuery,
)

from tools.registry import (
    get_tool,
)

from tools.ticketing.mcp import (
    register_ticketing_tools,
)

from tools.ticketing.presentation import (
    build_ticket_search_card,
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


def test_ticket_search_catalog_is_bounded_read_only():
    tool = (
        get_tool(
            "ticket_search"
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
        "jql"
        not in tool[
            "parameters"
        ]
    )

    assert (
        tool[
            "grounded_arguments"
        ]
        == [
            "project_key"
        ]
    )


def test_mock_ticket_search_filters_results():
    service = (
        MockTicketService()
    )

    result = (
        service.search_tickets(
            TicketSearchQuery(
                project_key="ITSM",
                status="In Progress",
                priority="High",
                limit=10,
            )
        )
    )

    assert result.ok is True

    assert (
        result.status
        == "success"
    )

    assert (
        result.count
        == 1
    )

    assert (
        result.truncated
        is False
    )

    assert (
        result.tickets[
            0
        ].key
        == "ITSM-101"
    )


def test_ticket_search_mcp_returns_structured_results():
    server = (
        FakeMCPServer()
    )

    register_ticketing_tools(
        server,
        MockTicketService(),
    )

    result = (
        server.functions[
            "ticket_search"
        ](
            status="Open",
            limit=10,
        )
    )

    assert result.ok is True

    assert (
        result.status
        == "success"
    )

    assert (
        result.count
        == 1
    )

    assert (
        result.tickets[
            0
        ].key
        == "OPS-42"
    )


def test_ticket_search_builds_generic_list_card():
    service = (
        MockTicketService()
    )

    search_result = (
        service.search_tickets(
            TicketSearchQuery(
                limit=10
            )
        )
    )

    card = (
        build_ticket_search_card(
            search_result.model_dump()
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
        == "ticket-list"
    )

    assert (
        card[
            "title"
        ]
        == "Ticket search"
    )

    assert (
        card[
            "status"
        ]
        == "success"
    )

    assert (
        len(
            card[
                "sections"
            ]
        )
        == 1
    )

    assert (
        card[
            "sections"
        ][
            0
        ][
            "kind"
        ]
        == "list"
    )


def test_jira_ticket_search_builds_trusted_jql():
    seen_requests = []

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        seen_requests.append(
            request
        )

        return (
            httpx.Response(
                200,

                json={
                    "issues": [
                        {
                            "key":
                                "KAN-1",

                            "fields": {
                                "summary":
                                    "Task 1",

                                "status": {
                                    "name":
                                        "To Do",
                                },

                                "issuetype": {
                                    "name":
                                        "Task",
                                },

                                "priority":
                                    None,

                                "assignee":
                                    None,

                                "reporter": {
                                    "displayName":
                                        "Test Reporter",
                                },

                                "project": {
                                    "key":
                                        "KAN",

                                    "name":
                                        "My Software Team",
                                },

                                "created": (
                                    "2026-09-14T12:"
                                    "48:55.619+0700"
                                ),

                                "updated": (
                                    "2026-09-14T12:"
                                    "48:56.145+0700"
                                ),
                            },
                        }
                    ],

                    "isLast":
                        True,
                },
            )
        )

    service = (
        JiraTicketService(
            base_url=(
                "https://example.atlassian.net"
            ),

            email=(
                "service@example.com"
            ),

            api_token=(
                "test-token"
            ),

            transport=(
                httpx.MockTransport(
                    handler
                )
            ),
        )
    )

    try:
        result = (
            service.search_tickets(
                TicketSearchQuery(
                    project_key="KAN",
                    status="To Do",
                    limit=10,
                )
            )
        )

    finally:
        service.close()

    assert result.ok is True

    assert (
        result.count
        == 1
    )

    assert (
        result.tickets[
            0
        ].key
        == "KAN-1"
    )

    assert (
        len(
            seen_requests
        )
        == 1
    )

    request = (
        seen_requests[
            0
        ]
    )

    assert (
        request.method
        == "POST"
    )

    assert (
        request.url.path
        == "/rest/api/3/search/jql"
    )

    body = (
        request.content.decode(
            "utf-8"
        )
    )

    assert (
        'project = \\"KAN\\"'
        in body
    )

    assert (
        'status = \\"To Do\\"'
        in body
    )

    assert (
        "ORDER BY updated DESC"
        in body
    )


def test_jira_ticket_search_rejects_invalid_project_without_http():
    called = False

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        nonlocal called

        called = True

        return (
            httpx.Response(
                500
            )
        )

    service = (
        JiraTicketService(
            base_url=(
                "https://example.atlassian.net"
            ),

            email=(
                "service@example.com"
            ),

            api_token=(
                "test-token"
            ),

            transport=(
                httpx.MockTransport(
                    handler
                )
            ),
        )
    )

    try:
        result = (
            service.search_tickets(
                TicketSearchQuery(
                    project_key=(
                        'KAN" OR project = OPS'
                    )
                )
            )
        )

    finally:
        service.close()

    assert result.ok is False

    assert (
        result.status
        == "denied"
    )

    assert called is False