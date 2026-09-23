import httpx

from services.ticketing import (
    JiraTicketService,
    build_ticket_mutation_service,
)

from tools.ticketing.mcp import (
    register_ticketing_tools,
)


class FakeMCPServer:

    def __init__(
        self,
    ):
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


def test_mcp_ticket_create_accepts_trusted_snapshot_and_returns_verification():

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        if (
            request.url.path
            == "/rest/api/3/project/10001"
        ):
            return (
                httpx.Response(
                    200,
                    json={
                        "id":
                            "10001",

                        "key":
                            "KAN",

                        "name":
                            "Kanban",
                    },
                )
            )

        if (
            request.url.path
            == (
                "/rest/api/3/issue/"
                "createmeta/10001/issuetypes"
            )
        ):
            return (
                httpx.Response(
                    200,
                    json={
                        "issueTypes": [
                            {
                                "id":
                                    "10010",

                                "name":
                                    "Task",

                                "subtask":
                                    False,
                            },
                        ],

                        "total":
                            1,
                    },
                )
            )

        if (
            request.method == "POST"
            and request.url.path
            == "/rest/api/3/issue"
        ):
            return (
                httpx.Response(
                    201,
                    json={
                        "key":
                            "KAN-4",
                    },
                )
            )

        if (
            request.url.path
            == "/rest/api/3/issue/KAN-4"
        ):
            return (
                httpx.Response(
                    200,
                    json={
                        "key":
                            "KAN-4",

                        "fields": {
                            "summary":
                                "VPN login failure",

                            "project": {
                                "id":
                                    "10001",

                                "key":
                                    "KAN",

                                "name":
                                    "Kanban",
                            },

                            "issuetype": {
                                "id":
                                    "10010",

                                "name":
                                    "Task",
                            },
                        },
                    },
                )
            )

        raise AssertionError(
            f"Unexpected request: {request.method} {request.url}"
        )

    ticket_service = (
        JiraTicketService(
            base_url=(
                "https://example.atlassian.net"
            ),

            email=(
                "agent@example.com"
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

    server = (
        FakeMCPServer()
    )

    register_ticketing_tools(
        server,
        ticket_service,
        build_ticket_mutation_service(
            ticket_service
        ),
    )

    result = (
        server.functions[
            "ticket_create"
        ](
            project_key="KAN",
            summary=(
                "VPN login failure"
            ),
            expected_project_id="10001",
            expected_project_key="KAN",
            expected_project_name="Kanban",
            expected_ticket_type_id="10010",
            expected_ticket_type_name="Task",
        )
    )

    assert result.ok is True
    assert result.status == "success"
    assert result.ticket_key == "KAN-4"

    assert (
        result.mutation_performed
        is True
    )

    assert (
        result.verification_ok
        is True
    )

    ticket_service.close()
