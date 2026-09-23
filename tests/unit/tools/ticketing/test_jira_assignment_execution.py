import json

import httpx

from services.ticketing import JiraTicketService, build_ticket_mutation_service


APPROVED = {
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


def assignable_response():
    return httpx.Response(
        200,
        json=[
            {
                "accountId": "acc-alice",
                "displayName": "Alice Example",
                "active": True,
            }
        ],
    )


def test_assignment_revalidates_puts_once_and_verifies_readback():
    put_count = 0
    issue_reads = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal put_count, issue_reads

        if request.method == "GET" and request.url.path == "/rest/api/3/issue/KAN-4":
            issue_reads += 1
            if issue_reads == 1:
                return issue_response()
            return issue_response(
                {
                    "accountId": "acc-alice",
                    "displayName": "Alice Example",
                }
            )

        if request.method == "GET" and request.url.path == "/rest/api/3/user/assignable/search":
            assert request.url.params["accountId"] == "acc-alice"
            return assignable_response()

        if request.method == "PUT" and request.url.path == "/rest/api/3/issue/KAN-4/assignee":
            put_count += 1
            assert json.loads(request.content) == {"accountId": "acc-alice"}
            return httpx.Response(204)

        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    service = build_service(handler)
    result = build_ticket_mutation_service(service).assign_ticket(**APPROVED)

    assert result.ok is True
    assert result.status == "success"
    assert result.changed is True
    assert result.mutation_performed is True
    assert result.verification_ok is True
    assert result.new_value == "acc-alice"
    assert put_count == 1
    service.close()


def test_assignment_ambiguous_transport_failure_is_outcome_unknown_without_retry():
    put_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal put_count

        if request.method == "GET" and request.url.path == "/rest/api/3/issue/KAN-4":
            return issue_response()

        if request.method == "GET" and request.url.path == "/rest/api/3/user/assignable/search":
            return assignable_response()

        if request.method == "PUT" and request.url.path == "/rest/api/3/issue/KAN-4/assignee":
            put_count += 1
            raise httpx.ReadTimeout("ambiguous")

        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    service = build_service(handler)
    result = build_ticket_mutation_service(service).assign_ticket(**APPROVED)

    assert result.ok is False
    assert result.status == "outcome_unknown"
    assert result.mutation_performed is None
    assert result.verification_ok is False
    assert put_count == 1
    service.close()
