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
            rows += [{"type": "lxc", "vmid": vmid, "node": NODE, "status": guest["status"], "name": None,
                      "pool": guest.get("pool")} for vmid, guest in self.guests.items()]
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
    # Tags follow the create (live 20 Nov: a pool token may not tag a guest that does not exist yet).
    assert "tags" not in plan["create"] and plan["tags"] == "janus-db,db-postgresql"
    text = format_approval_required("proxmox_database_create", resolved, "a1b2c3d4")
    assert "db-orders (LXC 120)" in text and "docker.io/library/postgres:17-alpine" in text
    assert "10.77.0.21:5432" in text and "No password is sent" in text
    # 23 Nov: one approval covers create, start and the health read; the approval says so.
    assert "starts it" in text
    assert cluster.posts == []


def test_a_floating_tag_or_unknown_engine_is_refused_before_approval(cluster):
    tool = TOOLS["proxmox_database_create"]
    assert tool["policy_resolver"](**{**ASKED, "version": "latest"})["status"] == "denied"
    assert tool["policy_resolver"](**{**ASKED, "engine": "mongodb"})["status"] == "denied"


@pytest.fixture
def owner_store(monkeypatch, tmp_path):
    from services.extensions.context import request_owner
    from services.extensions.database_credentials import DatabaseCredentials
    from services.extensions.store import ExtensionStore

    store = DatabaseCredentials(ExtensionStore(tmp_path))
    monkeypatch.setattr("services.extensions.database_credentials.database_credentials", lambda: store)
    monkeypatch.setattr("tools.proxmox.mcp.reachable", lambda address, port, timeout=3.0: True)
    token = request_owner.set("owner-a")
    yield store
    request_owner.reset(token)


def test_the_approved_plan_is_created_started_and_its_login_kept_for_the_account(cluster, owner_store):
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
    assert (receipt.ok, receipt.status, receipt.verification_ok) == (True, "executed", True), receipt
    assert [path for path, _ in cluster.posts] == [f"/nodes/{NODE}/storage/local/oci-registry-pull",
                                                   f"/nodes/{NODE}/lxc", f"/nodes/{NODE}/lxc/120/status/start"]
    login = owner_store.get("owner-a", NODE, 120)
    assert login is not None and login["username"] == "postgres" and login["address"] == "10.77.0.21"
    # The password is in the account's store only: never in the receipt or a request.
    assert login["password"] not in repr(receipt.model_dump()) and login["password"] not in repr(cluster.puts)
    assert owner_store.list("owner-b") == []


def test_without_a_signed_in_account_nothing_is_sent(cluster):
    tool = TOOLS["proxmox_database_create"]
    resolved, _ = resolve_policy_execution_arguments(
        tool_name="proxmox_database_create", tool=tool, original_arguments=dict(ASKED),
        policy_result=tool["policy_resolver"](**ASKED))
    server = Server()
    register_proxmox_tools(server, cluster.service())
    receipt = server.tools["proxmox_database_create"](**resolved)
    assert (receipt.ok, receipt.status) == (False, "denied") and cluster.posts == []


def test_a_create_that_changed_nothing_leaves_no_login_behind(cluster, owner_store):
    cluster.pull_status = "command 'skopeo copy' failed: exit code 1"
    tool = TOOLS["proxmox_database_create"]
    resolved, _ = resolve_policy_execution_arguments(
        tool_name="proxmox_database_create", tool=tool, original_arguments=dict(ASKED),
        policy_result=tool["policy_resolver"](**ASKED))
    server = Server()
    register_proxmox_tools(server, cluster.service())
    receipt = server.tools["proxmox_database_create"](**resolved)
    assert (receipt.ok, receipt.status) == (False, "failed") and owner_store.get("owner-a", NODE, 120) is None


class DeletingService:
    """Only guest_delete, answering as the real one does (deleted=True only once the guest is gone)."""

    def __init__(self, outcome):
        self.outcome = outcome

    def guest_delete(self, node, vmid, *, expected_type, expected_name):
        return {"ok": self.outcome == "executed", "status": self.outcome, "node": node, "vmid": vmid,
                "deleted": self.outcome == "executed", "changed": self.outcome == "executed"}


@pytest.mark.parametrize("outcome, kept", [("executed", False), ("outcome_unknown", True), ("conflict", True)])
def test_a_deleted_database_guest_takes_its_stored_login_with_it(owner_store, outcome, kept):
    # 23 Nov 2026: a governed delete left the deleted guest's login in the account's store.
    owner_store.save("owner-a", node=NODE, vmid=120, name="db-orders", engine="postgresql", address="10.77.0.21",
                     port=5432, username="postgres", password="p" * 32)
    owner_store.save("owner-a", node=NODE, vmid=121, name="db-wiki", engine="mariadb", address="10.77.0.22",
                     port=3306, username="root", password="q" * 32)
    server = Server()
    register_proxmox_tools(server, DeletingService(outcome))  # type: ignore[arg-type]
    receipt = server.tools["proxmox_guest_delete"](node=NODE, vmid=120, expected_type="lxc",
                                                    expected_name="db-orders")
    assert receipt.status == outcome
    # Only a confirmed delete forgets the login, and only that guest's.
    assert (owner_store.get("owner-a", NODE, 120) is not None) is kept
    assert owner_store.get("owner-a", NODE, 121) is not None
