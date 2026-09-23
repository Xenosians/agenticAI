import json

import httpx

from services.ticketing import JiraTicketService, build_ticket_mutation_service


APPROVED = {
    "ticket_key": "KAN-4",
    "status": "Resolved",
    "expected_issue_id": "10004",
    "expected_issue_key": "KAN-4",
    "expected_issue_summary": "Executive onboarding",
    "expected_project_id": "10001",
    "expected_project_key": "KAN",
    "expected_project_name": "My Software Team",
    "expected_source_status_id": "1",
    "expected_source_status_name": "To Do",
    "expected_transition_noop": False,
    "expected_transition_id": "31",
    "expected_transition_name": "Resolve issue",
    "expected_target_status_id": "5",
    "expected_target_status_name": "Resolved",
}


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
                "status": {"id": status_id, "name": status_name},
            },
        },
    )


def transitions_response():
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


def test_transition_revalidates_posts_exact_id_once_and_verifies_status():
    post_count = 0
    issue_reads = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal post_count, issue_reads

        if request.method == "GET" and request.url.path == "/rest/api/3/issue/KAN-4":
            issue_reads += 1
            if issue_reads == 1:
                return issue_response()
            return issue_response("Resolved", "5")

        if request.method == "GET" and request.url.path == "/rest/api/3/issue/KAN-4/transitions":
            return transitions_response()

        if request.method == "POST" and request.url.path == "/rest/api/3/issue/KAN-4/transitions":
            post_count += 1
            assert json.loads(request.content) == {"transition": {"id": "31"}}
            return httpx.Response(204)

        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    service = build_service(handler)
    result = build_ticket_mutation_service(service).transition_ticket(**APPROVED)

    assert result.ok is True
    assert result.status == "success"
    assert result.changed is True
    assert result.mutation_performed is True
    assert result.verification_ok is True
    assert result.previous_value == "To Do"
    assert result.new_value == "Resolved"
    assert post_count == 1
    service.close()


def test_transition_5xx_is_outcome_unknown_without_retry():
    post_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal post_count

        if request.method == "GET" and request.url.path == "/rest/api/3/issue/KAN-4":
            return issue_response()

        if request.method == "GET" and request.url.path == "/rest/api/3/issue/KAN-4/transitions":
            return transitions_response()

        if request.method == "POST" and request.url.path == "/rest/api/3/issue/KAN-4/transitions":
            post_count += 1
            return httpx.Response(503)

        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    service = build_service(handler)
    result = build_ticket_mutation_service(service).transition_ticket(**APPROVED)

    assert result.ok is False
    assert result.status == "outcome_unknown"
    assert result.mutation_performed is None
    assert result.verification_ok is False
    assert post_count == 1
    service.close()
