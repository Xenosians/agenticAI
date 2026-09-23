import httpx

from services.ticketing import JiraTicketService
from services.ticketing.jira_transition_policy import (
    prepare_jira_ticket_transition,
)


def build_service(handler):
    return JiraTicketService(
        base_url="https://example.atlassian.net",
        email="agent@example.com",
        api_token="test-token",
        transport=httpx.MockTransport(handler),
    )


def issue_response(status_name="To Do", status_id="1"):
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
                "status": {
                    "id": status_id,
                    "name": status_name,
                },
            },
        },
    )


def test_prepare_transition_binds_exact_transition_id_and_source_state():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rest/api/3/issue/KAN-4":
            return issue_response()

        if request.url.path == "/rest/api/3/issue/KAN-4/transitions":
            return httpx.Response(
                200,
                json={
                    "transitions": [
                        {
                            "id": "31",
                            "name": "Resolve issue",
                            "to": {"id": "5", "name": "Resolved"},
                        }
                    ]
                },
            )

        raise AssertionError(f"Unexpected request: {request.url}")

    service = build_service(handler)
    result = prepare_jira_ticket_transition(
        service,
        ticket_key="KAN-4",
        status="Resolved",
    )

    assert result["ok"] is True
    assert result["risk"] == "medium"
    args = result["execution_arguments"]
    assert args["expected_source_status_name"] == "To Do"
    assert args["expected_transition_noop"] is False
    assert args["expected_transition_id"] == "31"
    assert args["expected_transition_name"] == "Resolve issue"
    assert args["expected_target_status_id"] == "5"
    assert args["expected_target_status_name"] == "Resolved"
    service.close()


def test_prepare_transition_rejects_ambiguous_provider_transition():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rest/api/3/issue/KAN-4":
            return issue_response()

        if request.url.path == "/rest/api/3/issue/KAN-4/transitions":
            return httpx.Response(
                200,
                json={
                    "transitions": [
                        {"id": "31", "name": "Resolve", "to": {"id": "5", "name": "Resolved"}},
                        {"id": "32", "name": "Resolved", "to": {"id": "5", "name": "Resolved"}},
                    ]
                },
            )

        raise AssertionError(f"Unexpected request: {request.url}")

    service = build_service(handler)
    result = prepare_jira_ticket_transition(
        service,
        ticket_key="KAN-4",
        status="Resolved",
    )

    assert result["ok"] is False
    assert result["status"] == "denied"
    assert "ambiguous" in result["error"].lower()
    service.close()
