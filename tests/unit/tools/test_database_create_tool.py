"""proxmox_database_create as an approval-gated tool (owner decision 20 Nov 2026, item 5).

The trusted policy asks Proxmox for the next free VMID and the addresses guests already use on the bridged
subnet, then freezes the whole plan; the executable refuses a plan that does not match the request and
creates exactly what was approved.
"""
import httpx
import pytest

from subagents.core.tooling.gateway import resolve_policy_execution_arguments
from tests.unit.services.test_database_guests import NODE, Cluster
from tools.proxmox import catalog as proxmox_catalog
from tools.proxmox.mcp import register_proxmox_tools
from tools.registry import TOOLS, format_approval_required

ASKED = {"node": NODE, "engine": "postgresql", "version": "17-alpine", "hostname": "db-orders"}


class BridgedCluster(Cluster):
    """The trainer already holds 10.77.0.10 and one database guest holds .20."""

    def __init__(self):
        super().__init__()
        self.existing = {991: "name=eth0,bridge=janusnat,ip=10.77.0.10/24,gw=10.77.0.1",
                         992: "name=eth0,bridge=janusnat,ip=10.77.0.20/24,gw=10.77.0.1"}

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/api2/json")
        if path == "/cluster/resources":
            # Existing guests plus anything created during the test; a new guest is listed without a name.
            rows = [{"type": "lxc", "vmid": vmid, "node": NODE, "status": "running", "name": f"g{vmid}"}
                    for vmid in self.existing]
            rows += [{"type": "lxc", "vmid": vmid, "node": NODE, "status": "stopped", "name": None}
                     for vmid in self.guests]
            return httpx.Response(200, json={"data": rows})
        for vmid, net0 in self.existing.items():
            if path == f"/nodes/{NODE}/lxc/{vmid}/config":
                return httpx.Response(200, json={"data": {"net0": net0, "hostname": f"g{vmid}"}})
        return super().handler(request)


class Server:
    def __init__(self):
        self.tools = {}

    def tool(self):
        return lambda function: self.tools.setdefault(function.__name__, function)


@pytest.fixture
def cluster(monkeypatch):
    fake = BridgedCluster()
    monkeypatch.setattr(proxmox_catalog, "_policy_service", lambda: fake.service())
    return fake


def test_the_policy_freezes_a_plan_with_the_next_vmid_and_the_first_free_bridge_address(cluster):
    tool = TOOLS["proxmox_database_create"]
    assert tool["effect"] == "mutation" and tool["requires_approval"] is True and tool["risk"] == "medium"
    result = tool["policy_resolver"](**ASKED)
    assert result["ok"] is True, result
    resolved, error = resolve_policy_execution_arguments(tool_name="proxmox_database_create", tool=tool,
                                                         original_arguments=dict(ASKED), policy_result=result)
    assert error is None and resolved is not None
    plan = resolved["plan"]
    assert resolved["allocated_vmid"] == 120 == plan["vmid"]
    assert plan["address"] == "10.77.0.21" and plan["create"]["pool"] == "janus"
    assert plan["create"]["tags"] == "janus-db,db-postgresql"
    text = format_approval_required("proxmox_database_create", resolved, "a1b2c3d4")
    assert "db-orders (LXC 120)" in text and "docker.io/library/postgres:17-alpine" in text
    assert "10.77.0.21:5432" in text and "No password is sent" in text
    assert cluster.posts == []


def test_a_floating_tag_or_unknown_engine_is_refused_before_approval(cluster):
    tool = TOOLS["proxmox_database_create"]
    assert tool["policy_resolver"](**{**ASKED, "version": "latest"})["status"] == "denied"
    assert tool["policy_resolver"](**{**ASKED, "engine": "mongodb"})["status"] == "denied"


def test_the_approved_plan_is_created_and_a_mismatched_plan_is_refused(cluster):
    tool = TOOLS["proxmox_database_create"]
    resolved, _ = resolve_policy_execution_arguments(
        tool_name="proxmox_database_create", tool=tool, original_arguments=dict(ASKED),
        policy_result=tool["policy_resolver"](**ASKED))
    assert resolved is not None
    server = Server()
    register_proxmox_tools(server, cluster.service())
    tampered = server.tools["proxmox_database_create"](**{**resolved, "hostname": "db-other"})
    assert (tampered.ok, tampered.status) == (False, "denied") and cluster.posts == []
    receipt = server.tools["proxmox_database_create"](**resolved)
    assert (receipt.ok, receipt.status, receipt.verification_ok) == (True, "executed", True)
    assert [path for path, _ in cluster.posts] == [f"/nodes/{NODE}/storage/local/oci-registry-pull",
                                                   f"/nodes/{NODE}/lxc"]
