import json

import httpx

from services.ticketing import JiraTicketService, build_ticket_mutation_service


COMMENT = "  Exact first line\n\nExact third line  "

APPROVED = {
    "ticket_key": "KAN-4",
    "comment": COMMENT,
    "expected_issue_id": "10004",
    "expected_issue_key": "KAN-4",
    "expected_issue_summary": "Executive onboarding",
    "expected_project_id": "10001",
    "expected_project_key": "KAN",
    "expected_project_name": "My Software Team",
}


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


def test_jira_comment_revalidates_posts_once_and_verifies_exact_body():
    post_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal post_count

        if request.method == "GET" and request.url.path == "/rest/api/3/issue/KAN-4":
            return issue_response()

        if request.method == "POST" and request.url.path == "/rest/api/3/issue/KAN-4/comment":
            post_count += 1
            assert json.loads(request.content) == {"body": comment_adf(COMMENT)}
            return httpx.Response(201, json={"id": "20001"})

        if request.method == "GET" and request.url.path == "/rest/api/3/issue/KAN-4/comment/20001":
            return httpx.Response(
                200,
                json={
                    "id": "20001",
                    "body": comment_adf(COMMENT),
                },
            )

        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    service = build_service(handler)
    result = build_ticket_mutation_service(service).add_comment(**APPROVED)

    assert result.ok is True
    assert result.status == "success"
    assert result.ticket_key == "KAN-4"
    assert result.comment_id == "20001"
    assert result.mutation_performed is True
    assert result.verification_ok is True
    assert result.changed is True
    assert post_count == 1
    service.close()


def test_jira_comment_stale_snapshot_fails_closed_before_post():
    post_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal post_count
        if request.method == "GET" and request.url.path == "/rest/api/3/issue/KAN-4":
            return issue_response(summary="Changed after approval")
        if request.method == "POST":
            post_count += 1
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    service = build_service(handler)
    result = build_ticket_mutation_service(service).add_comment(**APPROVED)

    assert result.ok is False
    assert result.status == "denied"
    assert result.mutation_performed is False
    assert result.verification_ok is False
    assert post_count == 0
    service.close()


def test_jira_comment_transport_failure_is_outcome_unknown_no_retry():
    post_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal post_count
        if request.method == "GET" and request.url.path == "/rest/api/3/issue/KAN-4":
            return issue_response()
        if request.method == "POST" and request.url.path == "/rest/api/3/issue/KAN-4/comment":
            post_count += 1
            raise httpx.ReadTimeout("ambiguous timeout")
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    service = build_service(handler)
    result = build_ticket_mutation_service(service).add_comment(**APPROVED)

    assert result.ok is False
    assert result.status == "outcome_unknown"
    assert result.mutation_performed is None
    assert result.verification_ok is False
    assert post_count == 1
    service.close()


def test_jira_comment_5xx_is_outcome_unknown_no_retry():
    post_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal post_count
        if request.method == "GET" and request.url.path == "/rest/api/3/issue/KAN-4":
            return issue_response()
        if request.method == "POST" and request.url.path == "/rest/api/3/issue/KAN-4/comment":
            post_count += 1
            return httpx.Response(503, json={"error": "unavailable"})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    service = build_service(handler)
    result = build_ticket_mutation_service(service).add_comment(**APPROVED)

    assert result.ok is False
    assert result.status == "outcome_unknown"
    assert result.mutation_performed is None
    assert result.verification_ok is False
    assert post_count == 1
    service.close()


def test_jira_comment_readback_mismatch_is_outcome_unknown():
    post_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal post_count
        if request.method == "GET" and request.url.path == "/rest/api/3/issue/KAN-4":
            return issue_response()
        if request.method == "POST" and request.url.path == "/rest/api/3/issue/KAN-4/comment":
            post_count += 1
            return httpx.Response(201, json={"id": "20001"})
        if request.method == "GET" and request.url.path == "/rest/api/3/issue/KAN-4/comment/20001":
            return httpx.Response(
                200,
                json={
                    "id": "20001",
                    "body": comment_adf("Different body"),
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    service = build_service(handler)
    result = build_ticket_mutation_service(service).add_comment(**APPROVED)

    assert result.ok is False
    assert result.status == "outcome_unknown"
    assert result.mutation_performed is True
    assert result.verification_ok is False
    assert result.comment_id == "20001"
    assert post_count == 1
    service.close()
