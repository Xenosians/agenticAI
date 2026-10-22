import pytest

from services.ticketing import MockTicketService, build_ticket_mutation_service
from tools.registry import format_approval_required, format_tool_result, get_tool
from tools.ticketing import policy
from tools.ticketing.mcp import register_ticketing_tools


class FakeMCPServer:
    def __init__(self):
        self.functions = {}

    def tool(self):
        def decorator(function):
            self.functions[function.__name__] = function
            return function

        return decorator


def registered():
    server = FakeMCPServer()
    service = MockTicketService()
    register_ticketing_tools(server, service, build_ticket_mutation_service(service))
    return server.functions, service


MUTATIONS = ("ticket_update", "ticket_link", "ticket_worklog_add")
READS = ("ticket_links", "ticket_assigned")


def test_catalog_safety_contract_for_new_tools():
    for name in MUTATIONS:
        tool = get_tool(name)
        assert tool["risk"] == "medium"
        assert tool["requires_approval"] is True
        assert callable(tool["approval_formatter"])
        assert callable(tool["policy_resolver"])
        assert set(tool["trusted_policy_arguments"]).isdisjoint(tool["parameters"])
        assert set(tool["required_arguments"]) <= set(tool["grounded_arguments"])

    for name in READS:
        tool = get_tool(name)
        assert tool["risk"] == "read"
        assert tool["requires_approval"] is False
        assert "policy_resolver" not in tool


def test_mcp_round_trip_on_mock_provider():
    functions, service = registered()

    update = functions["ticket_update"](ticket_key="ITSM-101", priority="Low", add_labels=["matrix-check"])
    assert update.ok is True
    assert [change.field for change in update.changes] == ["priority", "labels"]

    link = functions["ticket_link"](ticket_key="ITSM-101", other_ticket_key="OPS-42", relationship="blocks")
    assert link.ok is True and link.changed is True

    links = functions["ticket_links"](ticket_key="OPS-42")
    assert links.count == 1
    assert links.links[0].relationship == "is blocked by"

    assigned = functions["ticket_assigned"](assignee="Service Desk", status="In Progress")
    assert [ticket.key for ticket in assigned.tickets] == ["ITSM-101"]

    worklog = functions["ticket_worklog_add"](ticket_key="OPS-42", time_spent="15m", comment="matrix worklog")
    assert worklog.ok is True
    assert worklog.worklog_id == "mock-worklog-1"
    assert service.worklogs["OPS-42"][0]["comment"] == "matrix worklog"


def test_mock_refuses_jira_snapshot_arguments():
    functions, _ = registered()

    result = functions["ticket_update"](ticket_key="ITSM-101", priority="Low", expected_issue_id="1")
    assert result.ok is False
    assert result.status == "denied"


def test_provider_neutral_policies_keep_exact_arguments(monkeypatch):
    class MockSettings:
        ticketing_backend = "mock"

    monkeypatch.setattr(policy, "Settings", MockSettings)

    update = policy.evaluate_ticket_update_policy(ticket_key="OPS-42", add_labels=["x"])
    assert update["execution_arguments"] == {"ticket_key": "OPS-42", "add_labels": ["x"]}
    assert update["requires_approval"] is True

    empty = policy.evaluate_ticket_update_policy(ticket_key="OPS-42")
    assert empty["ok"] is False and empty["status"] == "denied"

    link = policy.evaluate_ticket_link_policy("OPS-42", "ITSM-101", "relates to")
    assert link["execution_arguments"] == {
        "ticket_key": "OPS-42",
        "other_ticket_key": "ITSM-101",
        "relationship": "relates to",
    }

    worklog = policy.evaluate_ticket_worklog_add_policy("OPS-42", "15m")
    assert worklog["execution_arguments"] == {"ticket_key": "OPS-42", "time_spent": "15m"}


def test_approval_text_lists_old_to_new_and_plain_link_sentence():
    update = format_approval_required(
        "ticket_update",
        {
            "ticket_key": "SUP-9",
            "priority": "high",
            "add_labels": ["matrix-check"],
            "due_date": "2026-12-01",
            "expected_priority_name": "Medium",
            "expected_target_priority_name": "High",
            "expected_labels": ["vpn"],
            "expected_due_date": None,
        },
        "abc123",
    )
    assert update == (
        "Updating SUP-9 requires approval: priority: Medium → High; "
        "labels: vpn → matrix-check, vpn; due date: (none) → 2026-12-01. "
        "Approval ID: abc123."
    )

    link = format_approval_required(
        "ticket_link",
        {
            "ticket_key": "SUP-12",
            "other_ticket_key": "SUP-9",
            "relationship": "blocks",
            "expected_link_phrase": "blocks",
            "expected_link_exists": False,
        },
        None,
    )
    assert link == "Linking requires approval: SUP-12 blocks SUP-9."

    neutral = format_approval_required(
        "ticket_update",
        {"ticket_key": "OPS-42", "priority": "Low", "add_labels": ["x"]},
        None,
    )
    assert neutral == "Updating OPS-42 requires approval: priority → Low; labels: add x."

    worklog = format_approval_required(
        "ticket_worklog_add",
        {"ticket_key": "SUP-9", "time_spent": "15m", "comment": "matrix worklog"},
        None,
    )
    assert worklog == "Logging 15m of work on SUP-9 requires approval, with the comment: matrix worklog."


@pytest.mark.parametrize(
    ("tool", "result", "expected"),
    [
        (
            "ticket_links",
            {
                "status": "success",
                "ticket_key": "SUP-12",
                "links": [{"relationship": "blocks", "other_key": "SUP-9", "other_summary": "Printer", "other_status": "Open"}],
            },
            "Linked issues:\nSUP-12 blocks SUP-9 — Printer · Open",
        ),
        ("ticket_links", {"status": "success", "ticket_key": "SUP-12", "links": []}, "No issues are linked to SUP-12."),
        (
            "ticket_assigned",
            {"status": "success", "assignee_display_name": "Alex", "tickets": []},
            "No tickets are assigned to Alex.",
        ),
        (
            "ticket_update",
            {
                "message": "Ticket updated and verified.",
                "changes": [{"field": "priority", "from_value": "Medium", "to_value": "Low"}],
            },
            "Ticket updated and verified.\npriority: Medium → Low",
        ),
    ],
)
def test_result_formatting(tool, result, expected):
    assert format_tool_result(tool, result) == expected


def test_result_cards_validate():
    links_card = get_tool("ticket_links")["presentation_builder"](
        {"status": "success", "ticket_key": "SUP-12", "count": 1, "provider": "jira",
         "links": [{"relationship": "blocks", "other_key": "SUP-9"}]}
    )
    assert links_card["kind"] == "ticket-links"
    assert links_card["sections"][0]["content"] == ["SUP-12 blocks SUP-9"]

    assigned_card = get_tool("ticket_assigned")["presentation_builder"](
        {"status": "success", "assignee_display_name": "Alex", "assignee_account_id": "acc-1", "count": 1,
         "tickets": [{"key": "SUP-1", "summary": "VPN", "status": "Open", "provider": "jira"}]}
    )
    assert assigned_card["title"] == "Tickets assigned to Alex"
    assert assigned_card["sections"][0]["content"] == ["SUP-1 — VPN · Open"]
