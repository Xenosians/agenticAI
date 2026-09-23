import httpx

from services.ticketing import JiraTicketService
from services.ticketing.jira_comment_policy import prepare_jira_ticket_comment


def build_service(handler):
    return JiraTicketService(
        base_url="https://example.atlassian.net",
        email="agent@example.com",
        api_token="test-token",
        transport=httpx.MockTransport(handler),
    )


def issue_response(summary="Executive onboarding"):
    return httpx.Response(
        200,
        json={
            "id": "10004",
            "key": "KAN-4",
            "fields": {
                "summary": summary,
                "project": {
                    "id": "10001",
                    "key": "KAN",
                    "name": "My Software Team",
                },
            },
        },
    )


def test_prepare_jira_comment_binds_exact_issue_snapshot_and_comment():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.method == "GET"
        assert request.url.path == "/rest/api/3/issue/KAN-4"
        assert request.url.params["fields"] == "summary,project"
        return issue_response()

    service = build_service(handler)
    comment = "  Keep this exact.\nSecond line  "

    result = prepare_jira_ticket_comment(
        service,
        ticket_key="KAN-4",
        comment=comment,
    )

    assert result["ok"] is True
    assert result["status"] == "ready"
    assert result["risk"] == "medium"
    assert result["requires_approval"] is True
    assert result["execution_arguments"] == {
        "ticket_key": "KAN-4",
        "comment": comment,
        "expected_issue_id": "10004",
        "expected_issue_key": "KAN-4",
        "expected_issue_summary": "Executive onboarding",
        "expected_project_id": "10001",
        "expected_project_key": "KAN",
        "expected_project_name": "My Software Team",
    }
    assert len(calls) == 1
    service.close()


def test_prepare_jira_comment_refuses_provider_key_rewrite():
    def handler(request: httpx.Request) -> httpx.Response:
        return issue_response()

    service = build_service(handler)

    result = prepare_jira_ticket_comment(
        service,
        ticket_key="kan-4",
        comment="Exact comment",
    )

    assert result["ok"] is False
    assert result["status"] == "denied"
    assert "different issue key" in result["error"]
    service.close()
