import httpx

from services.ticketing import JiraTicketService
from services.ticketing.jira_assignment_policy import (
    prepare_jira_ticket_assignment,
)


def build_service(handler):
    return JiraTicketService(
        base_url="https://example.atlassian.net",
        email="agent@example.com",
        api_token="test-token",
        transport=httpx.MockTransport(handler),
    )


def issue_response(assignee=None):
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
                "assignee": assignee,
            },
        },
    )


def test_prepare_assignment_binds_provider_account_id_and_issue_snapshot():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rest/api/3/issue/KAN-4":
            assert request.url.params["fields"] == "summary,project,assignee"
            return issue_response()

        if request.url.path == "/rest/api/3/user/assignable/search":
            assert request.url.params["issueKey"] == "KAN-4"
            assert request.url.params["query"] == "Alice Example"
            return httpx.Response(
                200,
                json=[
                    {
                        "accountId": "acc-alice",
                        "displayName": "Alice Example",
                        "emailAddress": "alice@example.com",
                        "active": True,
                    }
                ],
            )

        raise AssertionError(f"Unexpected request: {request.url}")

    service = build_service(handler)
    result = prepare_jira_ticket_assignment(
        service,
        ticket_key="KAN-4",
        assignee="Alice Example",
    )

    assert result["ok"] is True
    assert result["risk"] == "medium"
    assert result["execution_arguments"] == {
        "ticket_key": "KAN-4",
        "assignee": "Alice Example",
        "expected_issue_id": "10004",
        "expected_issue_key": "KAN-4",
        "expected_issue_summary": "Executive onboarding",
        "expected_project_id": "10001",
        "expected_project_key": "KAN",
        "expected_project_name": "My Software Team",
        "expected_previous_assignee_present": False,
        "expected_previous_assignee_account_id": None,
        "expected_previous_assignee_display_name": None,
        "expected_assignee_account_id": "acc-alice",
        "expected_assignee_display_name": "Alice Example",
    }
    service.close()


def test_prepare_assignment_rejects_ambiguous_display_name():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rest/api/3/issue/KAN-4":
            return issue_response()

        if request.url.path == "/rest/api/3/user/assignable/search":
            return httpx.Response(
                200,
                json=[
                    {"accountId": "a1", "displayName": "Alex", "active": True},
                    {"accountId": "a2", "displayName": "Alex", "active": True},
                ],
            )

        raise AssertionError(f"Unexpected request: {request.url}")

    service = build_service(handler)
    result = prepare_jira_ticket_assignment(
        service,
        ticket_key="KAN-4",
        assignee="Alex",
    )

    assert result["ok"] is False
    assert result["status"] == "denied"
    assert "ambiguous" in result["error"].lower()
    service.close()
