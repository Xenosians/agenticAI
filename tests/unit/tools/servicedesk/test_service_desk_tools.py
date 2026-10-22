import pytest

from services.servicedesk import (
    NOT_CONFIGURED,
    QueueListResult,
    ServiceDeskError,
    ServiceDeskQueue,
    ServiceRequestResult,
    SlaMetric,
    SlaStatusResult,
)
from tools.registry import format_approval_required, format_tool_result, get_tool
from tools.servicedesk import policy
from tools.servicedesk.mcp import register_service_desk_tools

TOOLS = ("service_request_create", "sla_status", "queue_list")


class FakeMCPServer:
    def __init__(self):
        self.functions = {}

    def tool(self):
        def decorator(function):
            self.functions[function.__name__] = function
            return function

        return decorator


class StubProvider:
    """Canned provider results; HTTP behaviour is covered by the service tests."""

    def __init__(self, refusal=None):
        self.calls = []
        self.closed = False
        self.refusal = refusal

    def prepare_request(self, **arguments):
        self.calls.append(("prepare", arguments))
        if self.refusal:
            raise self.refusal
        return {"expected_service_desk_id": "1", "expected_request_type_id": "7"}

    def create_request(self, **arguments):
        self.calls.append(("create", arguments))
        return ServiceRequestResult(ok=True, status="success", ticket_key="SUP-20", message="Raised SUP-20.")

    def sla_status(self, request_key):
        return SlaStatusResult(
            ok=True, status="success", request_key=request_key, count=1,
            metrics=[SlaMetric(name="Time to resolution", state="ongoing", goal="8h", elapsed="5h", remaining="3h")],
        )

    def queue_list(self, service_desk):
        return QueueListResult(
            ok=True, status="success", service_desk="SUP", service_desk_name="Support", count=2,
            queues=[ServiceDeskQueue(id="8", name="All open", issue_count=9), ServiceDeskQueue(id="9", name="Mine", issue_count=1)],
        )

    def close(self):
        self.closed = True


def registered(provider=None, settings=None):
    server = FakeMCPServer()
    register_service_desk_tools(server, settings, service=provider)
    return server.functions


def test_catalog_safety_contract():
    create = get_tool("service_request_create")
    assert create["risk"] == "medium" and create["requires_approval"] is True
    assert callable(create["policy_resolver"]) and callable(create["approval_formatter"])
    assert set(create["trusted_policy_arguments"]).isdisjoint(create["parameters"])
    assert set(create["required_arguments"]) <= set(create["grounded_arguments"]) | set(create["derived_arguments"])

    for name in ("sla_status", "queue_list"):
        tool = get_tool(name)
        assert tool["risk"] == "read" and tool["requires_approval"] is False
        assert "policy_resolver" not in tool
        assert tool["description"].startswith("Read only")


def test_tools_pass_exact_arguments_to_the_provider():
    provider = StubProvider()
    tools = registered(provider)
    assert set(tools) == set(TOOLS)

    created = tools["service_request_create"](
        "SUP", "Get IT help", "Printer offline", expected_service_desk_id="1", expected_request_type_id="7",
    )
    assert created.ticket_key == "SUP-20"
    assert provider.calls == [("create", {
        "service_desk": "SUP",
        "request_type": "Get IT help",
        "summary": "Printer offline",
        "description": None,
        "expected_service_desk_id": "1",
        "expected_request_type_id": "7",
    })]
    assert tools["sla_status"]("SUP-20").request_key == "SUP-20"
    assert tools["queue_list"]("SUP").count == 2


def test_tools_report_not_configured_without_jira():
    class MockSettings:
        ticketing_backend = "mock"

    tools = registered(settings=MockSettings())
    calls = {"service_request_create": ("SUP", "Get IT help", "Printer offline"), "sla_status": ("SUP-1",), "queue_list": ("SUP",)}
    for name, arguments in calls.items():
        result = tools[name](*arguments)
        assert result.ok is False and result.error_code == "not_configured"
        assert result.error == NOT_CONFIGURED


@pytest.fixture
def provider(monkeypatch):
    stub = StubProvider()
    monkeypatch.setattr(policy, "Settings", lambda: None)
    monkeypatch.setattr(policy, "build_service_desk_service", lambda settings: stub)
    return stub


def test_policy_freezes_provider_ids_and_keeps_exact_arguments(provider):
    decision = policy.evaluate_service_request_create_policy("SUP", "Get IT help", "Printer offline")

    assert decision["ok"] is True and decision["status"] == "ready" and decision["requires_approval"] is True
    assert decision["execution_arguments"] == {
        "service_desk": "SUP",
        "request_type": "Get IT help",
        "summary": "Printer offline",
        "expected_service_desk_id": "1",
        "expected_request_type_id": "7",
    }
    assert provider.closed is True


def test_policy_refuses_before_approval(provider):
    provider.refusal = ServiceDeskError("unsupported_fields", "The request type also requires Select a system.")
    decision = policy.evaluate_service_request_create_policy("SUP", "Request a new account", "New account")
    assert decision["ok"] is False and decision["status"] == "denied"
    assert "Select a system" in decision["error"]


def test_policy_without_jira_is_not_configured(monkeypatch):
    monkeypatch.setattr(policy, "Settings", lambda: None)
    monkeypatch.setattr(policy, "build_service_desk_service", lambda settings: None)
    decision = policy.evaluate_service_request_create_policy("SUP", "Get IT help", "Printer offline")
    assert decision["status"] == "error" and decision["error"] == NOT_CONFIGURED


def test_approval_text_names_desk_type_and_exact_text():
    text = format_approval_required(
        "service_request_create",
        {"service_desk": "SUP", "request_type": "Get IT help", "summary": "Printer offline", "description": "Third floor."},
        "approval-1",
    )
    for part in ("Get IT help", "SUP service desk", "Printer offline", "Third floor."):
        assert part in text
    assert ".." not in text


def test_read_results_format_as_text_and_cards():
    stub = StubProvider()
    sla = stub.sla_status("SUP-20").model_dump()
    queues = stub.queue_list("SUP").model_dump()

    assert format_tool_result("sla_status", sla) == (
        "SLA status of SUP-20: Time to resolution: ongoing, 3h remaining, 5h elapsed, goal 8h"
    )
    assert format_tool_result("queue_list", queues) == (
        "Queues of the Support (SUP) service desk: All open (9 issues); Mine (1 issue)"
    )
    assert format_tool_result("sla_status", {"ok": False, "error": "SUP-404 was not found."}) == (
        "The service desk SLA lookup did not succeed: SUP-404 was not found."
    )

    card = get_tool("queue_list")["presentation_builder"](queues)
    assert card["kind"] == "service-desk-queues" and card["title"] == "Support queues"
    assert card["sections"][0]["content"] == ["All open (9 issues)", "Mine (1 issue)"]
    assert get_tool("sla_status")["presentation_builder"](sla)["title"] == "SLA status of SUP-20"
