import json

import httpx
import pytest

from services.servicedesk import ServiceDeskService, build_service_desk_service

DESKS = [
    {"id": "1", "projectId": "10068", "projectKey": "SUP", "projectName": "Support"},
    {"id": "2", "projectId": "10069", "projectKey": "ITSM", "projectName": "ITSM"},
]
TYPES = [
    {"id": "1", "name": "Get IT help", "canCreateRequest": True},
    {"id": "8", "name": "Report a system problem", "canCreateRequest": True},
    {"id": "3", "name": "Request a new account", "canCreateRequest": True},
]
FIELDS = {
    "1": [
        {"fieldId": "summary", "name": "Summary", "required": True},
        {"fieldId": "description", "name": "Description", "required": False},
        {"fieldId": "attachment", "name": "Attachment", "required": False},
    ],
    "8": [
        {"fieldId": "summary", "name": "Summarize the problem", "required": True},
        {"fieldId": "description", "name": "Describe what happened", "required": True},
        {"fieldId": "customfield_10116", "name": "Urgency", "required": False},
    ],
    "3": [
        {"fieldId": "summary", "name": "Summary", "required": True},
        {"fieldId": "components", "name": "Select a system", "required": True},
    ],
}


def page(values, last=True):
    return httpx.Response(200, json={"size": len(values), "start": 0, "limit": 50, "isLastPage": last, "values": values})


class FakeJira:
    """A minimal Jira Service Management site: two desks, three request types."""

    def __init__(self, created=None):
        self.posts = []
        self.created = created
        self.created_key = "SUP-20"

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path == "/rest/servicedeskapi/request":
            body = json.loads(request.content)
            self.posts.append(body)
            self.created = self.created or {
                "issueKey": self.created_key,
                "serviceDeskId": body["serviceDeskId"],
                "requestTypeId": body["requestTypeId"],
                "requestFieldValues": [
                    {"fieldId": name, "value": value} for name, value in body["requestFieldValues"].items()
                ],
                "currentStatus": {"status": "Waiting for support"},
            }
            return httpx.Response(201, json={"issueKey": self.created_key})
        if path == "/rest/servicedeskapi/servicedesk":
            return page(DESKS)
        if path == "/rest/servicedeskapi/servicedesk/1/requesttype":
            return page(TYPES)
        if path.startswith("/rest/servicedeskapi/servicedesk/1/requesttype/") and path.endswith("/field"):
            return httpx.Response(200, json={"requestTypeFields": FIELDS[path.split("/")[-2]]})
        if path == f"/rest/servicedeskapi/request/{self.created_key}" and self.created:
            return httpx.Response(200, json=self.created)
        return httpx.Response(404, json={"errorMessage": "sd.portal.error.issue.not.found"})


def build(handler):
    return ServiceDeskService(
        base_url="https://example.atlassian.net",
        email="agent@example.com",
        api_token="test-token",
        transport=httpx.MockTransport(handler),
    )


ARGS = {"service_desk": "SUP", "request_type": "Get IT help", "summary": "Printer offline"}


def test_desk_resolves_by_key_or_name_and_lists_known_desks_when_missing():
    service = build(FakeJira())
    assert service.resolve_service_desk("sup").id == "1"
    assert service.resolve_service_desk("Support").project_key == "SUP"

    queues = service.queue_list("HR")
    assert queues.ok is False and queues.status == "not_found"
    assert "SUP" in queues.error and "ITSM" in queues.error


def test_ambiguous_desk_names_are_refused():
    twins = [dict(DESKS[0]), {**DESKS[1], "projectName": "Support"}]
    service = build(lambda request: page(twins))
    result = service.queue_list("Support")
    assert result.status == "denied" and result.error_code == "ambiguous"


def test_queue_list_reads_counts_for_one_desk():
    def handler(request):
        if request.url.path == "/rest/servicedeskapi/servicedesk/1/queue":
            assert request.url.params["includeCount"] == "true"
            return page([
                {"id": "8", "name": "All open", "issueCount": 9, "jql": "project = SUP"},
                {"id": "9", "name": "Assigned to me", "issueCount": 4},
                {"id": "10", "name": "Unsized"},
            ])
        return FakeJira()(request)

    result = build(handler).queue_list("SUP")
    assert result.ok is True and result.status == "success"
    assert (result.service_desk, result.service_desk_name, result.count) == ("SUP", "Support", 3)
    assert [(queue.name, queue.issue_count) for queue in result.queues] == [
        ("All open", 9), ("Assigned to me", 4), ("Unsized", None),
    ]


def test_sla_status_maps_every_cycle_state():
    def friendly(text):
        return {"friendly": text}

    metrics = [
        {"name": "Time to first response", "completedCycles": [],
         "ongoingCycle": {"breached": False, "paused": False, "goalDuration": friendly("4h"),
                          "elapsedTime": friendly("1h 5m"), "remainingTime": friendly("2h 55m")}},
        {"name": "Time to resolution", "completedCycles": [],
         "ongoingCycle": {"breached": True, "paused": True, "goalDuration": friendly("8h"),
                          "elapsedTime": friendly("9h"), "remainingTime": friendly("-1h")}},
        {"name": "Time to close", "completedCycles": [
            {"breached": False, "goalDuration": friendly("1d"), "elapsedTime": friendly("2h"), "remainingTime": friendly("22h")}]},
        {"name": "Time to approve", "completedCycles": []},
    ]

    def handler(request):
        assert request.url.path == "/rest/servicedeskapi/request/SUP-12/sla"
        return page(metrics)

    result = build(handler).sla_status("sup-12")
    assert result.ok is True and result.request_key == "SUP-12" and result.count == 4
    states = [(metric.state, metric.breached, metric.remaining) for metric in result.metrics]
    assert states == [("ongoing", False, "2h 55m"), ("paused", True, "-1h"), ("completed", False, "22h"), ("not_started", False, None)]
    assert result.metrics[2].completed_cycles == 1


def test_sla_status_refuses_bad_keys_and_reports_missing_requests():
    service = build(FakeJira())
    bad = service.sla_status("request twelve")
    assert bad.status == "denied" and bad.error_code == "invalid_argument"

    missing = service.sla_status("SUP-404")
    assert missing.status == "not_found"
    assert "not a service desk request" in missing.error
    assert "sd.portal" not in missing.error


def test_prepare_freezes_desk_and_request_type():
    assert build(FakeJira()).prepare_request(**ARGS) == {
        "expected_service_desk_id": "1",
        "expected_request_type_id": "1",
    }


@pytest.mark.parametrize(
    ("arguments", "code", "phrase"),
    [
        ({**ARGS, "request_type": "Request a new account"}, "unsupported_fields", "Select a system"),
        ({**ARGS, "request_type": "Report a system problem"}, "invalid_argument", "requires a description"),
        ({**ARGS, "request_type": "Order pizza"}, "not_found", "Get IT help"),
        ({**ARGS, "summary": " padded"}, "invalid_argument", "whitespace"),
    ],
)
def test_prepare_refuses_requests_it_cannot_fill_exactly(arguments, code, phrase):
    from services.servicedesk import ServiceDeskError

    with pytest.raises(ServiceDeskError) as caught:
        build(FakeJira()).prepare_request(**arguments)
    assert caught.value.code == code
    assert phrase in str(caught.value)


def test_create_raises_once_and_verifies_by_reading_back():
    jira = FakeJira()
    result = build(jira).create_request(
        **ARGS, description="Third floor", expected_service_desk_id="1", expected_request_type_id="1",
    )
    assert result.ok is True and result.status == "success" and result.verification_ok is True
    assert (result.ticket_key, result.request_status, result.service_desk) == ("SUP-20", "Waiting for support", "SUP")
    assert jira.posts == [{
        "serviceDeskId": "1",
        "requestTypeId": "1",
        "requestFieldValues": {"summary": "Printer offline", "description": "Third floor"},
    }]


def test_create_refuses_when_the_target_changed_after_approval():
    jira = FakeJira()
    result = build(jira).create_request(**ARGS, expected_service_desk_id="1", expected_request_type_id="8")
    assert result.status == "denied" and "changed after approval" in result.error
    assert jira.posts == []

    unprepared = build(jira).create_request(**ARGS)
    assert unprepared.status == "denied" and jira.posts == []


def test_create_reports_unknown_outcome_when_read_back_differs():
    jira = FakeJira(created={"issueKey": "SUP-20", "serviceDeskId": "1", "requestTypeId": "1",
                             "requestFieldValues": [{"fieldId": "summary", "value": "Something else"}]})
    result = build(jira).create_request(**ARGS, expected_service_desk_id="1", expected_request_type_id="1")
    assert result.ok is False and result.status == "outcome_unknown"
    assert result.mutation_performed is True and result.ticket_key == "SUP-20"


def test_create_never_retries_an_ambiguous_transport_failure():
    calls = []

    def handler(request):
        if request.method == "POST":
            calls.append(request)
            raise httpx.ReadTimeout("slow", request=request)
        return FakeJira()(request)

    result = build(handler).create_request(**ARGS, expected_service_desk_id="1", expected_request_type_id="1")
    assert result.status == "outcome_unknown" and result.mutation_performed is None
    assert len(calls) == 1


def test_free_plan_refusal_maps_to_plan_required():
    def handler(request):
        if request.method == "POST":
            return httpx.Response(402, json={"errorMessage": "Upgrade required"})
        return FakeJira()(request)

    result = build(handler).create_request(**ARGS, expected_service_desk_id="1", expected_request_type_id="1")
    assert result.status == "denied" and result.error_code == "plan_required"


def test_provider_is_only_built_for_configured_jira():
    class Settings:
        ticketing_backend = "mock"
        jira_base_url = jira_email = jira_api_token = None

    assert build_service_desk_service(Settings()) is None
