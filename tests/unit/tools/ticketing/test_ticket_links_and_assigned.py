import json

import httpx

from services.ticketing import JiraTicketService, MockTicketService
from services.ticketing.types import TicketLink


def build_service(handler):
    return JiraTicketService(
        base_url="https://example.atlassian.net",
        email="agent@example.com",
        api_token="test-token",
        transport=httpx.MockTransport(handler),
    )


BLOCKS = {"id": "10000", "name": "Blocks", "inward": "is blocked by", "outward": "blocks"}
RELATES = {"id": "10003", "name": "Relates", "inward": "relates to", "outward": "relates to"}


def linked_issue(key, summary, status):
    return {"id": "1", "key": key, "fields": {"summary": summary, "status": {"name": status}}}


def test_jira_links_read_relationship_from_requested_issue_perspective():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/rest/api/3/issue/SUP-12"
        assert request.url.params["fields"] == "issuelinks"
        return httpx.Response(
            200,
            json={
                "id": "10012",
                "key": "SUP-12",
                "fields": {
                    "issuelinks": [
                        {"id": "501", "type": BLOCKS, "outwardIssue": linked_issue("SUP-9", "Printer", "Open")},
                        {"id": "502", "type": BLOCKS, "inwardIssue": linked_issue("SUP-3", "Network", "Done")},
                        {"id": "503", "type": RELATES, "outwardIssue": linked_issue("ITSM-4", "VPN", "Pending")},
                        {"id": "504", "type": {"name": "Broken"}},
                    ]
                },
            },
        )

    result = build_service(handler).get_ticket_links("sup-12")

    assert result.ok is True
    assert result.status == "success"
    assert result.provider == "jira"
    assert result.ticket_key == "SUP-12"
    assert result.count == 3
    assert [(link.relationship, link.other_key) for link in result.links] == [
        ("blocks", "SUP-9"),
        ("is blocked by", "SUP-3"),
        ("relates to", "ITSM-4"),
    ]
    assert result.links[0] == TicketLink(
        id="501",
        link_type="Blocks",
        direction="outward",
        relationship="blocks",
        other_key="SUP-9",
        other_summary="Printer",
        other_status="Open",
    )
    assert result.links[1].direction == "inward"


def test_jira_links_map_missing_issue_and_invalid_key():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"errorMessages": ["Issue does not exist"]})

    service = build_service(handler)

    missing = service.get_ticket_links("SUP-999")
    assert missing.ok is False
    assert missing.status == "not_found"

    invalid = service.get_ticket_links("not a key")
    assert invalid.ok is False
    assert invalid.status == "denied"


def search_payload(*keys):
    return {
        "issues": [
            {
                "id": str(index),
                "key": key,
                "fields": {
                    "summary": f"Summary {key}",
                    "status": {"name": "Open"},
                    "assignee": {"accountId": "acc-1", "displayName": "Alex Agent"},
                },
            }
            for index, key in enumerate(keys)
        ],
        "isLast": True,
    }


def test_jira_assigned_resolves_exact_email_and_searches_by_account_id():
    searches = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rest/api/3/user/search":
            assert request.url.params["query"] == "alex@example.test"
            return httpx.Response(
                200,
                json=[
                    {"accountId": "acc-1", "displayName": "Alex Agent", "emailAddress": "alex@example.test"},
                    {"accountId": "acc-2", "displayName": "Alexandra", "emailAddress": "alexandra@example.test"},
                ],
            )
        if request.url.path == "/rest/api/3/search/jql":
            body = json.loads(request.content)
            searches.append(body)
            return httpx.Response(200, json=search_payload("SUP-1", "SUP-2"))
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    result = build_service(handler).get_assigned_tickets(
        "alex@example.test",
        status='In "Review"',
        limit=5,
    )

    assert result.ok is True
    assert result.status == "success"
    assert result.assignee_account_id == "acc-1"
    assert result.assignee_display_name == "Alex Agent"
    assert [ticket.key for ticket in result.tickets] == ["SUP-1", "SUP-2"]
    assert searches == [
        {
            "jql": 'assignee = "acc-1" AND status = "In \\"Review\\"" ORDER BY updated DESC',
            "maxResults": 5,
            "fields": searches[0]["fields"],
        }
    ]


def test_jira_assigned_accepts_exact_account_id():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rest/api/3/user":
            assert request.url.params["accountId"] == "acc-1"
            return httpx.Response(200, json={"accountId": "acc-1", "displayName": "Alex Agent"})
        if request.url.path == "/rest/api/3/search/jql":
            assert json.loads(request.content)["jql"] == 'assignee = "acc-1" ORDER BY updated DESC'
            return httpx.Response(200, json=search_payload("SUP-1"))
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    result = build_service(handler).get_assigned_tickets("acc-1")

    assert result.ok is True
    assert result.count == 1
    assert result.assignee_account_id == "acc-1"


def test_jira_assigned_refuses_self_reference_without_any_request():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no provider request is allowed")

    service = build_service(handler)

    for reference in ("me", "Myself", "  current user "):
        result = service.get_assigned_tickets(reference)
        assert result.ok is False
        assert result.status == "denied"
        assert "service account" in result.error


def test_jira_assigned_denies_ambiguous_and_unknown_users():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rest/api/3/user":
            return httpx.Response(404, json={"errorMessages": ["not found"]})
        if request.url.path == "/rest/api/3/user/search":
            if request.url.params["query"] == "Sam":
                return httpx.Response(
                    200,
                    json=[
                        {"accountId": "a", "displayName": "Sam"},
                        {"accountId": "b", "displayName": "sam"},
                    ],
                )
            return httpx.Response(200, json=[])
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    service = build_service(handler)

    ambiguous = service.get_assigned_tickets("Sam")
    assert ambiguous.ok is False
    assert ambiguous.status == "denied"
    assert "2 Jira accounts" in ambiguous.error

    unknown = service.get_assigned_tickets("Nobody Here")
    assert unknown.ok is False
    assert unknown.status == "not_found"


def test_jira_assigned_rejects_out_of_range_limit_without_request():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no provider request is allowed")

    result = build_service(handler).get_assigned_tickets("acc-1", limit=26)

    assert result.ok is False
    assert result.status == "denied"


def test_mock_links_and_assigned_share_state():
    service = MockTicketService()

    links = service.get_ticket_links("ITSM-101")
    assert links.ok is True
    assert links.count == 0

    assigned = service.get_assigned_tickets("service desk")
    assert assigned.ok is True
    assert [ticket.key for ticket in assigned.tickets] == ["ITSM-101"]

    assert service.get_assigned_tickets("me").status == "denied"
    assert service.get_ticket_links("NOPE-1").status == "not_found"
