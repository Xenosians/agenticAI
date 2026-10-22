import json

import httpx

from services.ticketing import JiraTicketService, build_ticket_mutation_service
from services.ticketing.jira_update_policy import prepare_jira_ticket_update


def build_service(handler):
    return JiraTicketService(
        base_url="https://example.atlassian.net",
        email="agent@example.com",
        api_token="test-token",
        transport=httpx.MockTransport(handler),
    )


class FakeIssue:
    """Mutable issue state behind a mock Jira transport."""

    def __init__(self, *, labels=None, priority=("3", "Medium"), due=None, duedate_editable=True):
        self.labels = list(labels or [])
        self.priority = priority
        self.due = due
        self.duedate_editable = duedate_editable
        self.puts = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "GET" and path == "/rest/api/3/issue/SUP-9":
            return httpx.Response(
                200,
                json={
                    "id": "10077",
                    "key": "SUP-9",
                    "fields": {
                        "summary": "Disposable",
                        "project": {"id": "10068", "key": "SUP", "name": "Support"},
                        "priority": {"id": self.priority[0], "name": self.priority[1]},
                        "labels": list(self.labels),
                        "duedate": self.due,
                    },
                },
            )
        if request.method == "GET" and path == "/rest/api/3/issue/SUP-9/editmeta":
            fields = {
                "priority": {
                    "operations": ["set"],
                    "allowedValues": [{"id": "2", "name": "High"}, {"id": "3", "name": "Medium"}, {"id": "4", "name": "Low"}],
                },
                "labels": {"operations": ["add", "set", "remove"]},
            }
            if self.duedate_editable:
                fields["duedate"] = {"operations": ["set"]}
            return httpx.Response(200, json={"fields": fields})
        if request.method == "GET" and path == "/rest/api/3/priority":
            return httpx.Response(
                200,
                json=[
                    {"id": "1", "name": "Highest"},
                    {"id": "2", "name": "High"},
                    {"id": "3", "name": "Medium"},
                    {"id": "4", "name": "Low"},
                ],
            )
        if request.method == "PUT" and path == "/rest/api/3/issue/SUP-9":
            body = json.loads(request.content)
            self.puts.append(body)
            fields = body.get("fields", {})
            if "priority" in fields:
                names = {"2": "High", "3": "Medium", "4": "Low"}
                self.priority = (fields["priority"]["id"], names[fields["priority"]["id"]])
            if "duedate" in fields:
                self.due = fields["duedate"]
            for operation in body.get("update", {}).get("labels", []):
                if "add" in operation and operation["add"] not in self.labels:
                    self.labels.append(operation["add"])
                if "remove" in operation and operation["remove"] in self.labels:
                    self.labels.remove(operation["remove"])
            return httpx.Response(204)
        raise AssertionError(f"unexpected request {request.method} {request.url}")


def test_prepare_freezes_current_values_and_resolves_priority_case_insensitively():
    issue = FakeIssue(labels=["vpn"])
    result = prepare_jira_ticket_update(
        build_service(issue.handler),
        ticket_key="SUP-9",
        priority="high",
        add_labels=["matrix-check"],
    )

    assert result["ok"] is True
    assert result["status"] == "ready"
    assert result["execution_arguments"] == {
        "ticket_key": "SUP-9",
        "priority": "high",
        "add_labels": ["matrix-check"],
        "expected_issue_id": "10077",
        "expected_issue_key": "SUP-9",
        "expected_issue_summary": "Disposable",
        "expected_project_id": "10068",
        "expected_project_key": "SUP",
        "expected_project_name": "Support",
        "expected_priority_id": "3",
        "expected_priority_name": "Medium",
        "expected_labels": ["vpn"],
        "expected_due_date": None,
        "expected_target_priority_id": "2",
        "expected_target_priority_name": "High",
    }
    assert issue.puts == []


def test_prepare_denies_empty_update_unknown_priority_relative_date_and_hidden_field():
    issue = FakeIssue(duedate_editable=False)
    service = build_service(issue.handler)

    empty = prepare_jira_ticket_update(service, ticket_key="SUP-9")
    assert empty["status"] == "denied"
    assert "at least one change" in empty["error"]

    unknown = prepare_jira_ticket_update(service, ticket_key="SUP-9", priority="Urgent")
    assert unknown["status"] == "denied"
    assert "Valid priorities: High, Medium, Low" in unknown["error"]

    relative = prepare_jira_ticket_update(service, ticket_key="SUP-9", due_date="next Friday")
    assert relative["status"] == "denied"
    assert "YYYY-MM-DD" in relative["error"]

    hidden = prepare_jira_ticket_update(service, ticket_key="SUP-9", due_date="2026-12-01")
    assert hidden["status"] == "denied"
    assert "Due date cannot be changed" in hidden["error"]

    conflicting = prepare_jira_ticket_update(
        service, ticket_key="SUP-9", add_labels=["a"], remove_labels=["a"]
    )
    assert conflicting["status"] == "denied"


def test_execute_uses_label_operations_and_verifies_read_back():
    issue = FakeIssue(labels=["vpn", "old"])
    service = build_service(issue.handler)
    prepared = prepare_jira_ticket_update(
        service,
        ticket_key="SUP-9",
        priority="Low",
        add_labels=["matrix-check"],
        remove_labels=["old"],
        due_date="2026-12-01",
    )

    result = build_ticket_mutation_service(service).update_ticket(**prepared["execution_arguments"])

    assert result.ok is True
    assert result.status == "success"
    assert result.changed is True
    assert result.verification_ok is True
    assert issue.puts == [
        {
            "fields": {"priority": {"id": "4"}, "duedate": "2026-12-01"},
            "update": {"labels": [{"add": "matrix-check"}, {"remove": "old"}]},
        }
    ]
    assert [(c.field, c.from_value, c.to_value) for c in result.changes] == [
        ("priority", "Medium", "Low"),
        ("labels", "old, vpn", "matrix-check, vpn"),
        ("duedate", None, "2026-12-01"),
    ]


def test_execute_refuses_changed_snapshot_without_put():
    issue = FakeIssue()
    service = build_service(issue.handler)
    prepared = prepare_jira_ticket_update(service, ticket_key="SUP-9", add_labels=["x"])

    issue.labels.append("added-elsewhere")
    result = build_ticket_mutation_service(service).update_ticket(**prepared["execution_arguments"])

    assert result.ok is False
    assert result.status == "denied"
    assert "changed after approval" in result.error
    assert issue.puts == []


def test_execute_is_a_verified_no_op_when_values_already_match():
    issue = FakeIssue(labels=["x"], priority=("2", "High"))
    service = build_service(issue.handler)
    prepared = prepare_jira_ticket_update(service, ticket_key="SUP-9", priority="High", add_labels=["x"])

    result = build_ticket_mutation_service(service).update_ticket(**prepared["execution_arguments"])

    assert result.ok is True
    assert result.changed is False
    assert result.mutation_performed is False
    assert issue.puts == []


def test_execute_requires_complete_trusted_snapshot():
    issue = FakeIssue()
    result = build_ticket_mutation_service(build_service(issue.handler)).update_ticket(
        "SUP-9",
        priority="High",
    )

    assert result.ok is False
    assert result.status == "denied"
    assert "complete trusted provider snapshot" in result.error
    assert issue.puts == []


def test_execute_maps_rejection_plan_limit_and_unverified_result():
    for status_code, expected_status, expected_code in (
        (400, "denied", None),
        (402, "denied", "plan_required"),
        (503, "outcome_unknown", None),
    ):
        issue = FakeIssue()
        base = issue.handler

        def handler(request, base=base, status_code=status_code):
            if request.method == "PUT":
                return httpx.Response(status_code, json={"errors": {"priority": "bad"}})
            return base(request)

        service = build_service(handler)
        prepared = prepare_jira_ticket_update(service, ticket_key="SUP-9", priority="High")
        result = build_ticket_mutation_service(service).update_ticket(**prepared["execution_arguments"])

        assert result.ok is False
        assert result.status == expected_status
        assert result.error_code == expected_code

    issue = FakeIssue()
    base = issue.handler

    def silent(request):
        if request.method == "PUT":
            return httpx.Response(204)
        return base(request)

    service = build_service(silent)
    prepared = prepare_jira_ticket_update(service, ticket_key="SUP-9", priority="High")
    result = build_ticket_mutation_service(service).update_ticket(**prepared["execution_arguments"])

    assert result.ok is False
    assert result.status == "outcome_unknown"
    assert result.mutation_performed is True
