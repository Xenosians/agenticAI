import json

import httpx
import pytest

from services.ticketing import JiraTicketService, build_ticket_mutation_service
from services.ticketing.jira_link_policy import prepare_jira_ticket_link
from services.ticketing.jira_worklog_policy import prepare_jira_worklog_add
from services.ticketing.validation import (
    duration_seconds,
    parse_duration,
    resolve_link_relationship,
)


LINK_TYPES = [
    {"id": "10000", "name": "Blocks", "inward": "is blocked by", "outward": "blocks"},
    {"id": "10002", "name": "Duplicate", "inward": "is duplicated by", "outward": "duplicates"},
    {"id": "10003", "name": "Relates", "inward": "relates to", "outward": "relates to"},
]


def build_service(handler):
    return JiraTicketService(
        base_url="https://example.atlassian.net",
        email="agent@example.com",
        api_token="test-token",
        transport=httpx.MockTransport(handler),
    )


def issue_payload(key, issue_id, links=None):
    fields = {
        "summary": f"Summary {key}",
        "project": {"id": "10068", "key": "SUP", "name": "Support"},
    }
    if links is not None:
        fields["issuelinks"] = links
    return {"id": issue_id, "key": key, "fields": fields}


class FakeLinks:
    """Mock Jira that stores links the way Jira reports them per issue."""

    def __init__(self, *, permission=True):
        self.links = []
        self.posts = []
        self.permission = permission

    def _view(self, key):
        view = []
        for link in self.links:
            link_type = next(t for t in LINK_TYPES if t["id"] == link["type"])
            # Jira: "<inwardIssue> <outward phrase> <outwardIssue>". From the
            # inward issue the other side appears as outwardIssue.
            if link["inward"] == key:
                view.append({"id": link["id"], "type": link_type, "outwardIssue": {"key": link["outward"]}})
            if link["outward"] == key:
                view.append({"id": link["id"], "type": link_type, "inwardIssue": {"key": link["inward"]}})
        return view

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/rest/api/3/issueLinkType":
            return httpx.Response(200, json={"issueLinkTypes": LINK_TYPES})
        if path == "/rest/api/3/mypermissions":
            assert request.url.params["permissions"] == "LINK_ISSUES"
            return httpx.Response(200, json={"permissions": {"LINK_ISSUES": {"havePermission": self.permission}}})
        if request.method == "GET" and path in {"/rest/api/3/issue/SUP-12", "/rest/api/3/issue/SUP-9"}:
            key = path.rsplit("/", 1)[1]
            links = self._view(key) if request.url.params.get("fields") == "issuelinks" else None
            return httpx.Response(200, json=issue_payload(key, "1" + key[-1], links))
        if request.method == "POST" and path == "/rest/api/3/issueLink":
            body = json.loads(request.content)
            self.posts.append(body)
            self.links.append(
                {
                    "id": str(20000 + len(self.links)),
                    "type": body["type"]["id"],
                    "inward": body["inwardIssue"]["key"],
                    "outward": body["outwardIssue"]["key"],
                }
            )
            return httpx.Response(201)
        raise AssertionError(f"unexpected request {request.method} {request.url}")


@pytest.mark.parametrize(
    ("words", "direction", "inward", "outward"),
    [
        ("blocks", "outward", "SUP-12", "SUP-9"),
        ("is blocked by", "inward", "SUP-9", "SUP-12"),
    ],
)
def test_link_posts_the_direction_that_reads_back_as_the_user_sentence(words, direction, inward, outward):
    fake = FakeLinks()
    service = build_service(fake.handler)

    prepared = prepare_jira_ticket_link(
        service, ticket_key="SUP-12", other_ticket_key="SUP-9", relationship=words
    )
    assert prepared["ok"] is True
    arguments = prepared["execution_arguments"]
    assert arguments["expected_link_direction"] == direction
    assert arguments["expected_link_exists"] is False

    result = build_ticket_mutation_service(service).link_tickets(**arguments)

    assert result.ok is True
    assert result.changed is True
    assert result.verification_ok is True
    assert result.new_value == f"SUP-12 {words} SUP-9"
    assert fake.posts == [
        {"type": {"id": "10000"}, "inwardIssue": {"key": inward}, "outwardIssue": {"key": outward}}
    ]

    links = service.get_ticket_links("SUP-12")
    assert [(link.relationship, link.other_key) for link in links.links] == [(words, "SUP-9")]


def test_existing_identical_link_is_success_without_duplicate():
    fake = FakeLinks()
    service = build_service(fake.handler)
    mutations = build_ticket_mutation_service(service)

    first = prepare_jira_ticket_link(service, ticket_key="SUP-12", other_ticket_key="SUP-9", relationship="relates to")
    assert mutations.link_tickets(**first["execution_arguments"]).changed is True

    # A symmetric link created from the other side also counts as existing.
    second = prepare_jira_ticket_link(service, ticket_key="SUP-9", other_ticket_key="SUP-12", relationship="relates to")
    assert second["execution_arguments"]["expected_link_exists"] is True

    result = mutations.link_tickets(**second["execution_arguments"])
    assert result.ok is True
    assert result.changed is False
    assert len(fake.posts) == 1


def test_link_approved_as_existing_is_refused_after_the_link_disappears():
    fake = FakeLinks()
    service = build_service(fake.handler)
    mutations = build_ticket_mutation_service(service)

    first = prepare_jira_ticket_link(service, ticket_key="SUP-12", other_ticket_key="SUP-9", relationship="blocks")
    mutations.link_tickets(**first["execution_arguments"])

    approved_noop = prepare_jira_ticket_link(service, ticket_key="SUP-12", other_ticket_key="SUP-9", relationship="blocks")
    assert approved_noop["execution_arguments"]["expected_link_exists"] is True

    fake.links.clear()
    result = mutations.link_tickets(**approved_noop["execution_arguments"])

    assert result.ok is False
    assert result.status == "denied"
    assert len(fake.posts) == 1


def test_link_denies_unknown_words_self_link_and_missing_permission():
    fake = FakeLinks(permission=False)
    service = build_service(fake.handler)

    unknown = prepare_jira_ticket_link(service, ticket_key="SUP-12", other_ticket_key="SUP-9", relationship="depends on")
    assert unknown["status"] == "denied"
    assert "Valid relationships: blocks, is blocked by" in unknown["error"]

    self_link = prepare_jira_ticket_link(service, ticket_key="SUP-12", other_ticket_key="SUP-12", relationship="blocks")
    assert self_link["status"] == "denied"

    no_permission = prepare_jira_ticket_link(service, ticket_key="SUP-12", other_ticket_key="SUP-9", relationship="blocks")
    assert no_permission["status"] == "denied"
    assert "LINK_ISSUES" in no_permission["error"]
    assert fake.posts == []


def test_relationship_resolution_is_exact_and_reports_ambiguity():
    assert resolve_link_relationship("Is Blocked By", LINK_TYPES)["direction"] == "inward"
    assert resolve_link_relationship("relates", LINK_TYPES)["link_type_name"] == "Relates"

    with pytest.raises(ValueError, match="ambiguous"):
        resolve_link_relationship(
            "blocks",
            LINK_TYPES + [{"id": "9", "name": "Gates", "inward": "is gated by", "outward": "blocks"}],
        )


def test_duration_parsing_is_exact():
    assert parse_duration("1h 30m") == {"h": 1, "m": 30}
    assert duration_seconds(parse_duration("1w 1d"), hours_per_day=8, days_per_week=5) == 48 * 3600

    for bad in ("", "90", "1.5h", "30m 1h", "1h  30m", "next friday", "0m"):
        with pytest.raises(ValueError):
            parse_duration(bad)


def worklog_handler(state, *, readback_seconds=None):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/rest/api/3/configuration":
            return httpx.Response(
                200,
                json={
                    "timeTrackingEnabled": True,
                    "timeTrackingConfiguration": {"workingHoursPerDay": 8.0, "workingDaysPerWeek": 5.0},
                },
            )
        if path == "/rest/api/3/mypermissions":
            return httpx.Response(200, json={"permissions": {"WORK_ON_ISSUES": {"havePermission": True}}})
        if request.method == "GET" and path == "/rest/api/3/issue/SUP-9":
            return httpx.Response(200, json=issue_payload("SUP-9", "10077"))
        if request.method == "POST" and path == "/rest/api/3/issue/SUP-9/worklog":
            assert request.url.params["adjustEstimate"] == "leave"
            state["posts"].append(json.loads(request.content))
            return httpx.Response(201, json={"id": "30001"})
        if request.method == "GET" and path == "/rest/api/3/issue/SUP-9/worklog/30001":
            posted = state["posts"][-1]
            return httpx.Response(
                200,
                json={
                    "id": "30001",
                    "issueId": "10077",
                    "timeSpentSeconds": readback_seconds if readback_seconds is not None else 5400,
                    "comment": posted.get("comment"),
                },
            )
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    return handler


def test_worklog_posts_adf_comment_once_and_verifies_read_back():
    state = {"posts": []}
    service = build_service(worklog_handler(state))

    prepared = prepare_jira_worklog_add(
        service, ticket_key="SUP-9", time_spent="1h 30m", comment="Checked logs\nRestarted VPN"
    )
    assert prepared["ok"] is True
    assert prepared["execution_arguments"]["expected_time_spent_seconds"] == 5400

    result = build_ticket_mutation_service(service).add_worklog(**prepared["execution_arguments"])

    assert result.ok is True
    assert result.worklog_id == "30001"
    assert result.verification_ok is True
    assert state["posts"] == [
        {
            "timeSpent": "1h 30m",
            "comment": {
                "type": "doc",
                "version": 1,
                "content": [
                    {"type": "paragraph", "content": [{"type": "text", "text": "Checked logs"}]},
                    {"type": "paragraph", "content": [{"type": "text", "text": "Restarted VPN"}]},
                ],
            },
        }
    ]


def test_worklog_read_back_mismatch_is_outcome_unknown():
    state = {"posts": []}
    service = build_service(worklog_handler(state, readback_seconds=60))
    prepared = prepare_jira_worklog_add(service, ticket_key="SUP-9", time_spent="1h 30m")

    result = build_ticket_mutation_service(service).add_worklog(**prepared["execution_arguments"])

    assert result.ok is False
    assert result.status == "outcome_unknown"
    assert result.worklog_id == "30001"


def test_worklog_denies_relative_duration_without_any_post():
    state = {"posts": []}
    service = build_service(worklog_handler(state))

    prepared = prepare_jira_worklog_add(service, ticket_key="SUP-9", time_spent="all morning")
    assert prepared["status"] == "denied"

    unapproved = build_ticket_mutation_service(service).add_worklog("SUP-9", "15m")
    assert unapproved.status == "denied"
    assert state["posts"] == []
