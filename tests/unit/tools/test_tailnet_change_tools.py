"""Tailnet device, DNS and key changes as approval-gated tools (owner decision 20 Nov 2026, item 1).

The tools follow the grant tools' contract: the trusted policy reads the target and freezes it; the
gateway binding accepts exactly the declared trusted arguments; the approval names the target and the
value it replaces; the executable re-reads, sends once and reads back; a removal or revocation is
confirmed by typing the device name or key id.
"""
import pytest

from subagents.core.tooling.gateway import resolve_policy_execution_arguments
from tests.unit.services.test_tailnet_changes import HUMMINGBIRD, LAB, Tailnet
from tools.networking import catalog as networking_catalog
from tools.networking.mcp import register_networking_tools
from tools.registry import TOOLS, format_approval_required, format_tool_result

NEW_TOOLS = ("network_tailnet_device_set", "network_tailnet_device_remove", "network_tailnet_dns_set",
             "network_tailnet_key_revoke")


class Server:
    def __init__(self):
        self.tools = {}

    def tool(self):
        return lambda function: self.tools.setdefault(function.__name__, function)


@pytest.fixture
def tailnet(monkeypatch):
    fake = Tailnet()
    monkeypatch.setattr(networking_catalog, "_policy_tailnet", lambda: fake.service())
    return fake


def _policy(name: str, **asked) -> dict:
    return TOOLS[name]["policy_resolver"](**asked)


def _propose(name: str, **asked) -> tuple[dict, dict]:
    """A proposal the trusted policy accepts, bound exactly as the gateway binds it."""
    tool = TOOLS[name]
    result = _policy(name, **asked)
    assert result["ok"], result
    resolved, error = resolve_policy_execution_arguments(tool_name=name, tool=tool, original_arguments=dict(asked),
                                                         policy_result=result)
    assert error is None and resolved is not None, error
    return result, resolved


def test_every_new_tool_is_an_approval_gated_mutation_and_removals_need_the_typed_target():
    for name in NEW_TOOLS:
        tool = TOOLS[name]
        assert tool["effect"] == "mutation" and tool["requires_approval"] is True, name
        assert tool["policy_owns_preconditions"] is True and callable(tool["policy_resolver"]), name
    assert TOOLS["network_tailnet_device_remove"]["confirmation_argument"] == "device_name"
    assert TOOLS["network_tailnet_key_revoke"]["confirmation_argument"] == "key_id"
    assert "confirmation_argument" not in TOOLS["network_tailnet_device_set"]
    assert TOOLS["network_tailnet_device_remove"]["risk"] == TOOLS["network_tailnet_dns_set"]["risk"] == "high"


def test_route_approval_is_proposed_frozen_executed_and_read_back(tailnet):
    # The live bridge (owner decision item 2): hummingbird's advertised guest subnet is approved.
    result, frozen = _propose("network_tailnet_device_set", device="hummingbird", routes=["10.77.0.0/24"])
    assert result["risk"] == "medium" and tailnet.writes == []
    assert frozen == {"device": "hummingbird", "routes": ["10.77.0.0/24"], "setting": "routes",
                      "device_id": HUMMINGBIRD["nodeId"], "device_name": "hummingbird", "device_tags": [],
                      "previous": {"enabled_routes": []}}
    text = format_approval_required("network_tailnet_device_set", frozen, "a1b2c3d4")
    assert "routes of tailnet device hummingbird to 10.77.0.0/24 (now enabled routes none)" in text
    assert "reach those subnets" in text and "a1b2c3d4" in text
    server = Server()
    register_networking_tools(server, None, tailnet.service())
    receipt = server.tools["network_tailnet_device_set"](**frozen)
    assert (receipt.ok, receipt.status, receipt.verification_ok, receipt.retry_safe) == (True, "executed", True, False)
    assert len(tailnet.writes) == 1
    assert "Janus read it back: enabled routes 10.77.0.0/24." in format_tool_result(
        "network_tailnet_device_set", receipt.model_dump())
    # The same approval run again finds the device moved and sends nothing.
    again = server.tools["network_tailnet_device_set"](**frozen)
    assert (again.ok, again.status) == (False, "conflict") and len(tailnet.writes) == 1


def test_two_settings_in_one_request_are_refused_before_approval(tailnet):
    refused = _policy("network_tailnet_device_set", device="janus-router-lab", authorized=False,
                      new_name="lab-two")
    assert (refused["ok"], refused["status"]) == (False, "denied") and "one setting" in refused["error"]


def test_deauthorizing_is_high_risk_and_says_the_device_loses_access(tailnet):
    result, frozen = _propose("network_tailnet_device_set", device="janus-router-lab", authorized=False)
    assert result["risk"] == "high"
    assert "loses tailnet access at once" in format_approval_required("network_tailnet_device_set", frozen, "x")


def test_device_removal_freezes_the_device_and_is_confirmed_by_its_name(tailnet):
    _, frozen = _propose("network_tailnet_device_remove", device="janus-router-lab")
    assert "remove" not in frozen and frozen["device_name"] == "janus-router-lab"
    assert frozen["device_tags"] == ["tag:janus-lab"]
    from agent.approvals import confirmation_text
    assert confirmation_text("network_tailnet_device_remove", frozen) == "janus-router-lab"
    server = Server()
    register_networking_tools(server, None, tailnet.service())
    receipt = server.tools["network_tailnet_device_remove"](**frozen)
    assert (receipt.ok, receipt.verification_ok) == (True, True) and LAB["nodeId"] not in tailnet.devices


def test_dns_change_freezes_the_previous_value_and_split_dns_sets_one_domain(tailnet):
    _, frozen = _propose("network_tailnet_dns_set", split_dns_domain="lab.internal",
                         split_dns_nameservers=["10.77.0.1"])
    assert frozen["setting"] == "split_dns" and frozen["previous"] == {"domain": "lab.internal", "nameservers": None}
    text = format_approval_required("network_tailnet_dns_set", frozen, "x")
    assert "lab.internal -> 10.77.0.1" in text
    server = Server()
    register_networking_tools(server, None, tailnet.service())
    receipt = server.tools["network_tailnet_dns_set"](**frozen)
    assert receipt.ok and tailnet.split == {"lab.internal": ["10.77.0.1"]}


def test_key_revocation_is_confirmed_by_the_key_id_and_never_janus_own(tailnet):
    _, frozen = _propose("network_tailnet_key_revoke", key_id="kAAAA1111CNTRL")
    assert frozen["key_tags"] == ["tag:janus-lab"]
    server = Server()
    register_networking_tools(server, None, tailnet.service())
    receipt = server.tools["network_tailnet_key_revoke"](**frozen)
    # Live 20 Nov 2026: the revoked key still reads back, marked revoked and invalid.
    assert (receipt.ok, receipt.verification_ok) == (True, True)
    assert tailnet.keys["kAAAA1111CNTRL"]["invalid"] is True


def test_without_a_tailnet_link_every_new_tool_names_the_fix():
    server = Server()
    register_networking_tools(server, None, None)
    assert server.tools["network_tailnet_device_set"](
        device="x", routes=["10.0.0.0/24"], setting="routes", device_id="n1", device_name="x", device_tags=[],
        previous={"enabled_routes": []}).error_code == "not_configured"
    assert server.tools["network_tailnet_dns_set"](magic_dns=True, setting="magic_dns",
                                                   previous={"magic_dns": False}).error_code == "not_configured"
