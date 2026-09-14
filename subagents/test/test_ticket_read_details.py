import httpx

from services.ticketing import (
    JiraTicketService,
    MockTicketService,
)

from tools.registry import (
    get_tool,
)

from tools.ticketing_mcp import (
    register_ticketing_tools,
)

from tools.ticketing_presentation import (
    build_ticket_comments_card,
    build_ticket_history_card,
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


def test_ticket_history_catalog_is_read_only():
    tool = (
        get_tool(
            "ticket_history"
        )
    )

    assert tool is not None
    assert tool["risk"] == "read"
    assert tool["requires_approval"] is False

    assert (
        tool["grounded_arguments"]
        == [
            "ticket_key"
        ]
    )


def test_ticket_comments_catalog_is_read_only():
    tool = (
        get_tool(
            "ticket_comments"
        )
    )

    assert tool is not None
    assert tool["risk"] == "read"
    assert tool["requires_approval"] is False

    assert (
        tool["grounded_arguments"]
        == [
            "ticket_key"
        ]
    )


def test_mock_ticket_history():
    service = (
        MockTicketService()
    )

    result = (
        service.get_ticket_history(
            "ITSM-101"
        )
    )

    assert result.ok is True
    assert result.count == 2

    assert (
        result.history[
            0
        ].changes[
            0
        ].field
        == "status"
    )


def test_mock_ticket_comments():
    service = (
        MockTicketService()
    )

    result = (
        service.get_ticket_comments(
            "ITSM-101"
        )
    )

    assert result.ok is True
    assert result.count == 1

    assert (
        "VPN"
        in result.comments[
            0
        ].body
    )


def test_jira_ticket_history_maps_changelog():
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
                    "startAt": 0,
                    "maxResults": 20,
                    "total": 1,
                    "isLast": True,

                    "values": [
                        {
                            "id":
                                "10001",

                            "author": {
                                "displayName":
                                    "Alifio Y.A.S",
                            },

                            "created": (
                                "2026-09-14T13:"
                                "00:00.000+0700"
                            ),

                            "items": [
                                {
                                    "field":
                                        "status",

                                    "fromString":
                                        "To Do",

                                    "toString":
                                        "In Progress",
                                }
                            ],
                        }
                    ],
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
            service.get_ticket_history(
                "KAN-1"
            )
        )

    finally:
        service.close()

    assert result.ok is True
    assert result.provider == "jira"
    assert result.ticket_key == "KAN-1"
    assert result.count == 1

    assert (
        result.history[
            0
        ].author
        == "Alifio Y.A.S"
    )

    assert (
        result.history[
            0
        ].changes[
            0
        ].to_value
        == "In Progress"
    )

    assert (
        seen_requests[
            0
        ].url.path
        == (
            "/rest/api/3/issue/"
            "KAN-1/changelog"
        )
    )


def test_jira_ticket_comments_decodes_adf():
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
                    "startAt": 0,
                    "maxResults": 20,
                    "total": 1,

                    "comments": [
                        {
                            "id":
                                "20001",

                            "author": {
                                "displayName":
                                    "Alifio Y.A.S",
                            },

                            "body": {
                                "type":
                                    "doc",

                                "version":
                                    1,

                                "content": [
                                    {
                                        "type":
                                            "paragraph",

                                        "content": [
                                            {
                                                "type":
                                                    "text",

                                                "text":
                                                    "Investigating "
                                                    "the issue.",
                                            }
                                        ],
                                    }
                                ],
                            },

                            "created": (
                                "2026-09-14T13:"
                                "05:00.000+0700"
                            ),

                            "updated": (
                                "2026-09-14T13:"
                                "05:00.000+0700"
                            ),
                        }
                    ],
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
            service.get_ticket_comments(
                "KAN-1"
            )
        )

    finally:
        service.close()

    assert result.ok is True
    assert result.provider == "jira"
    assert result.count == 1

    assert (
        result.comments[
            0
        ].body
        == "Investigating the issue."
    )

    assert (
        seen_requests[
            0
        ].url.path
        == (
            "/rest/api/3/issue/"
            "KAN-1/comment"
        )
    )


def test_history_card_uses_generic_result_card():
    result = (
        MockTicketService()
        .get_ticket_history(
            "ITSM-101"
        )
    )

    card = (
        build_ticket_history_card(
            result.model_dump()
        )
    )

    assert (
        card["schema"]
        == "result-card.v1"
    )

    assert (
        card["kind"]
        == "ticket-history"
    )

    assert (
        card["title"]
        == "ITSM-101 history"
    )


def test_comments_card_uses_generic_result_card():
    result = (
        MockTicketService()
        .get_ticket_comments(
            "ITSM-101"
        )
    )

    card = (
        build_ticket_comments_card(
            result.model_dump()
        )
    )

    assert (
        card["schema"]
        == "result-card.v1"
    )

    assert (
        card["kind"]
        == "ticket-comments"
    )

    assert (
        card["title"]
        == "ITSM-101 comments"
    )