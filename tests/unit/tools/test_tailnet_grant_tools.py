"""D2: governed tailnet grant changes as approval-gated tools (direction note D2, owner decision 5).

Two typed tools add or remove one grant; nothing else in the policy can be written. The trusted
policy validates and freezes the change before approval; a removal is destructive and is
confirmed by typing the grant as the approval shows it. The tools resolve the proposing account's
own Tailscale link (per account, not per machine).
"""
import httpx
import pytest

from services.tailnet import TailnetConfig, TailnetService
from subagents.core.tooling.gateway import resolve_policy_execution_arguments
from tools.networking import catalog as networking_catalog
from tools.networking.mcp import register_networking_tools
from tools.registry import TOOLS, format_approval_required, format_tool_result

TEXT = '// policy\n{\n\t"grants": [\n\t\t{"src": ["autogroup:member"], "dst": ["*"], "ip": ["*"]},\n\t],\n}\n'
GRANT_TOOLS = ("network_tailnet_grant_add", "network_tailnet_grant_remove")
ASKED = {"sources": ["tag:janus-lab"], "destinations": ["tag:janus-lab"], "ports": ["22"]}


class Policy:
    def __init__(self, text=TEXT):
        self.text, self.version, self.writes = text, 1, 0

    def handler(self, request):
        path = request.url.path
        if path == "/api/v2/oauth/token":
            return httpx.Response(200, json={"access_token": "token", "expires_in": 3600})
        if path == "/api/v2/tailnet/-/settings":
            return httpx.Response(200, json={"aclsExternallyManagedOn": False})
        if path == "/api/v2/tailnet/-/acl/validate":
            return httpx.Response(200, json={})
        if path == "/api/v2/tailnet/-/acl" and request.method == "GET":
            return httpx.Response(200, text=self.text, headers={"ETag": f'"v{self.version}"'})
        if path == "/api/v2/tailnet/-/acl" and request.method == "POST":
            if request.headers.get("if-match") != f'"v{self.version}"':
                return httpx.Response(412)
            self.text, self.version, self.writes = request.content.decode(), self.version + 1, self.writes + 1
            return httpx.Response(200, text=self.text, headers={"ETag": f'"v{self.version}"'})
        raise AssertionError(path)

    def service(self):
        return TailnetService(TailnetConfig("id", "secret"), httpx.Client(transport=httpx.MockTransport(self.handler)))


@pytest.fixture
def policy(monkeypatch):
    stored = Policy()
    monkeypatch.setattr(networking_catalog, "_policy_tailnet", lambda: stored.service())
    return stored


def test_both_are_approval_gated_mutations_and_only_the_removal_needs_the_typed_grant():
    for name in GRANT_TOOLS:
        tool = TOOLS[name]
        assert tool["effect"] == "mutation" and tool["requires_approval"] is True
        assert tool["policy_owns_preconditions"] is True and callable(tool["policy_resolver"])
        assert set(tool["trusted_policy_arguments"]) == {"expected_etag", "policy_sha256", "new_policy_sha256",
                                                          "grant"}
    assert TOOLS["network_tailnet_grant_add"].get("confirmation_argument") is None
    assert TOOLS["network_tailnet_grant_remove"]["confirmation_argument"] == "grant"
    assert TOOLS["network_tailnet_grant_remove"]["risk"] == "high"


def test_the_policy_freezes_the_change_and_the_gateway_binding_accepts_it(policy):
    tool = TOOLS["network_tailnet_grant_add"]
    result = tool["policy_resolver"](**ASKED)
    assert result["ok"] is True and result["requires_approval"] is True and policy.writes == 0
    resolved, error = resolve_policy_execution_arguments(
        tool_name="network_tailnet_grant_add", tool=tool, original_arguments=dict(ASKED), policy_result=result)
    assert error is None
    assert resolved["grant"] == "tag:janus-lab -> tag:janus-lab : 22"
    assert set(resolved) == set(ASKED) | {"expected_etag", "policy_sha256", "new_policy_sha256", "grant"}


def test_a_refused_change_needs_no_approval(policy):
    refused = TOOLS["network_tailnet_grant_remove"]["policy_resolver"](**ASKED)
    assert (refused["ok"], refused["status"]) == (False, "denied") and "no grant" in refused["error"]
    bad = TOOLS["network_tailnet_grant_add"]["policy_resolver"](
        sources=["alice@example.com"], destinations=["*"], ports=["*"])
    assert (bad["ok"], bad["status"]) == (False, "denied") and "user" in bad["error"]


def test_the_approval_names_the_exact_grant_and_what_janus_checks():
    arguments = {**ASKED, "grant": "tag:janus-lab -> tag:janus-lab : 22", "expected_etag": '"v1"',
                 "policy_sha256": "a" * 64, "new_policy_sha256": "b" * 64}
    add = format_approval_required("network_tailnet_grant_add", arguments, "a1b2c3d4")
    assert "Add the tailnet grant tag:janus-lab -> tag:janus-lab : 22" in add and "a1b2c3d4" in add
    assert "validated" in add and "concurrent" in add
    remove = format_approval_required("network_tailnet_grant_remove", arguments, "a1b2c3d4")
    assert "Remove the tailnet grant tag:janus-lab -> tag:janus-lab : 22" in remove
    assert "lose" in remove


def test_the_mcp_tool_writes_the_approved_change_and_reports_the_receipt(policy):
    class Server:
        tools = {}

        def tool(self):
            return lambda function: self.tools.setdefault(function.__name__, function)

    server = Server()
    register_networking_tools(server, None, policy.service())
    frozen = TOOLS["network_tailnet_grant_add"]["policy_resolver"](**ASKED)["execution_arguments"]
    result = server.tools["network_tailnet_grant_add"](**frozen)
    assert (result.ok, result.status, result.verification_ok, result.retry_safe) == (True, "executed", True, False)
    assert policy.writes == 1
    text = format_tool_result("network_tailnet_grant_add", result.model_dump())
    assert "tag:janus-lab -> tag:janus-lab : 22" in text and "read the policy back" in text
    # A replay of the same approval finds the policy moved and writes nothing.
    again = server.tools["network_tailnet_grant_add"](**frozen)
    assert (again.ok, again.status) == (False, "conflict") and policy.writes == 1


def test_without_a_tailnet_link_the_tool_names_the_fix():
    class Server:
        tools = {}

        def tool(self):
            return lambda function: self.tools.setdefault(function.__name__, function)

    server = Server()
    register_networking_tools(server, None, None)
    result = server.tools["network_tailnet_grant_add"](
        **ASKED, grant="x", expected_etag='"v1"', policy_sha256="a" * 64, new_policy_sha256="b" * 64)
    assert (result.ok, result.status, result.error_code) == (False, "error", "not_configured")
