import httpx

from services.ticketing import JiraTicketService, build_ticket_mutation_service
from tools.ticketing.mcp import register_ticketing_tools


class FakeMCPServer:
    def __init__(self):
        self.functions = {}

    def tool(self):
        def decorator(function):
            self.functions[function.__name__] = function
            return function
        return decorator


def comment_adf(text):
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": (
                    [{"type": "text", "text": line}]
                    if line
                    else []
                ),
            }
            for line in text.split("\n")
        ],
    }


def test_mcp_ticket_add_comment_accepts_trusted_snapshot_and_returns_verification():
    comment = "Executive onboarding verified"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/rest/api/3/issue/KAN-4":
            return httpx.Response(
                200,
                json={
                    "id": "10004",
                    "key": "KAN-4",
                    "fields": {
                        "summary": "Executive onboarding",
                        "project": {
                            "id": "10001",
                            "key": "KAN",
                            "name": "My Software Team",
                        },
                    },
                },
            )

        if request.method == "POST" and request.url.path == "/rest/api/3/issue/KAN-4/comment":
            return httpx.Response(201, json={"id": "20001"})

        if request.method == "GET" and request.url.path == "/rest/api/3/issue/KAN-4/comment/20001":
            return httpx.Response(
                200,
                json={
                    "id": "20001",
                    "body": comment_adf(comment),
                },
            )

        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    service = JiraTicketService(
        base_url="https://example.atlassian.net",
        email="agent@example.com",
        api_token="test-token",
        transport=httpx.MockTransport(handler),
    )

    server = FakeMCPServer()
    register_ticketing_tools(
        server,
        service,
        build_ticket_mutation_service(service),
    )

    result = server.functions["ticket_add_comment"](
        ticket_key="KAN-4",
        comment=comment,
        expected_issue_id="10004",
        expected_issue_key="KAN-4",
        expected_issue_summary="Executive onboarding",
        expected_project_id="10001",
        expected_project_key="KAN",
        expected_project_name="My Software Team",
    )

    assert result.ok is True
    assert result.status == "success"
    assert result.ticket_key == "KAN-4"
    assert result.comment_id == "20001"
    assert result.mutation_performed is True
    assert result.verification_ok is True
    service.close()
