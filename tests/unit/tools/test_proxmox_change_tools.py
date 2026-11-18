"""Proxmox guest changes: approval-gated, exact-target policy, auto VMID, honest receipts."""
import pytest

from services.proxmox import ProxmoxConflict
from tools.proxmox import catalog as proxmox_catalog
from tools.proxmox.catalog import PROXMOX_CHANGE_TOOLS, PROXMOX_TOOLS
from tools.proxmox.mcp import register_proxmox_tools
from tools.registry import TOOLS, format_approval_required, format_tool_result

CHANGE_TOOLS = set(PROXMOX_CHANGE_TOOLS)
DESTRUCTIVE = {"proxmox_snapshot_delete", "proxmox_snapshot_rollback", "proxmox_guest_delete",
               "proxmox_backup_restore"}


class Changes:
    """Answers the read-only change policy like ProxmoxService for a stopped LXC guest."""

    def __init__(self, status="stopped", template=False, lock=None, used=()):
        self.status = status
        self.template = template
        self.lock = lock
        self.used = set(used)
        self.calls = []

    def guest_change_target(self, node, vmid):
        self.calls.append(("target", node, vmid))
        if (node, vmid) != ("hummingbird", 990):
            from services.provider_errors import ProviderResourceNotFound
            raise ProviderResourceNotFound(f"Proxmox VMID {vmid} was not found on node {node}.")
        return {"ok": True, "status": "success", "data": {
            "node": node, "vmid": vmid, "type": "lxc", "name": "janus-lab", "status": self.status,
            "lock": self.lock, "template": self.template, "digest": "d990"}}

    def next_vmid(self):
        self.calls.append(("nextid",))
        return 992

    def new_guest_target(self, node, newid):
        self.calls.append(("new", node, newid))
        if newid in self.used:
            raise ProxmoxConflict(f"VMID {newid} is already used on node {node}; choose a free VMID.")
        return {"ok": True, "status": "success", "data": {"node": node, "vmid": newid, "free": True}}

    def close(self):
        pass


@pytest.fixture
def changes(monkeypatch):
    provider = Changes(used={101})
    monkeypatch.setattr(proxmox_catalog, "_policy_service", lambda: provider)
    return provider


def test_every_change_is_an_approval_gated_mutation_with_the_owning_account_policy():
    for name in CHANGE_TOOLS:
        tool = TOOLS[name]
        assert tool["effect"] == "mutation" and tool["requires_approval"] is True
        assert tool["policy_owns_preconditions"] is True
        assert tool["risk"] in {"low", "medium", "high"}
        expected = "name" if name == "proxmox_backup_restore" else (
            "expected_name" if name in DESTRUCTIVE else None)
        assert tool.get("confirmation_argument") == expected


def test_configuration_changes_freeze_the_digest_and_the_guest_identity(changes):
    policy = TOOLS["proxmox_guest_set_resources"]["policy_resolver"]
    result = policy(node="hummingbird", vmid=990, cores=2, memory_mib=256)
    assert result["ok"] is True and result["requires_approval"] is True
    assert result["execution_arguments"] == {
        "node": "hummingbird", "vmid": 990, "cores": 2, "memory_mib": 256,
        "expected_type": "lxc", "expected_name": "janus-lab", "expected_digest": "d990"}


def test_a_template_locked_or_running_guest_is_refused_without_an_approval(monkeypatch):
    for provider, tool_name, words in [
        (Changes(template=True), "proxmox_guest_set_resources", "template"),
        (Changes(lock="backup"), "proxmox_snapshot_create", "locked"),
        (Changes(status="running"), "proxmox_guest_delete", "running"),
    ]:
        monkeypatch.setattr(proxmox_catalog, "_policy_service", lambda provider=provider: provider)
        arguments = {"node": "hummingbird", "vmid": 990}
        if tool_name == "proxmox_guest_set_resources":
            arguments["cores"] = 2
        if tool_name == "proxmox_snapshot_create":
            arguments["snapshot"] = "before"
        refused = TOOLS[tool_name]["policy_resolver"](**arguments)
        assert (refused["ok"], refused["status"]) == (False, "denied")
        assert words in refused["error"]


def test_new_guests_take_proxmox_own_next_free_vmid_when_none_is_named(changes):
    result = TOOLS["proxmox_guest_clone"]["policy_resolver"](
        node="hummingbird", vmid=990, name="lab-clone", full=True)
    assert result["ok"] is True
    assert result["execution_arguments"]["allocated_vmid"] == 992
    assert ("nextid",) in changes.calls

    restore = TOOLS["proxmox_backup_restore"]["policy_resolver"](
        node="hummingbird", volume="local:backup/vzdump-lxc-990-x.tar.zst", name="restored")
    assert restore["execution_arguments"]["allocated_vmid"] == 992


def test_an_explicit_used_vmid_is_refused_and_an_explicit_free_one_is_kept(changes):
    refused = TOOLS["proxmox_guest_clone"]["policy_resolver"](
        node="hummingbird", vmid=990, newid=101, name="lab-clone", full=True)
    assert (refused["ok"], refused["status"]) == (False, "denied")
    assert "101" in refused["error"]

    kept = TOOLS["proxmox_guest_clone"]["policy_resolver"](
        node="hummingbird", vmid=990, newid=993, name="lab-clone", full=True)
    assert "allocated_vmid" not in kept["execution_arguments"]
    assert kept["execution_arguments"]["newid"] == 993


def test_approval_text_names_the_exact_change_and_the_free_vmid(changes):
    arguments = {"node": "hummingbird", "vmid": 990, "cores": 2, "memory_mib": 256,
                 "expected_type": "lxc", "expected_name": "janus-lab", "expected_digest": "d990"}
    text = format_approval_required("proxmox_guest_set_resources", arguments, "a1b2c3d4")
    assert "LXC 990 (janus-lab)" in text and "2 CPU core(s)" in text and "256 MiB" in text
    assert "a1b2c3d4" in text

    clone = {"node": "hummingbird", "vmid": 990, "name": "lab-clone", "full": True,
             "expected_type": "lxc", "expected_name": "janus-lab", "allocated_vmid": 992}
    clone_text = format_approval_required("proxmox_guest_clone", clone, "a1b2c3d4")
    assert "992 (lab-clone)" in clone_text and "next free VMID" in clone_text


def test_an_unknown_outcome_is_never_presented_as_success():
    unknown = {"ok": False, "status": "outcome_unknown", "error": "Proxmox did not answer.",
               "node": "hummingbird", "vmid": 990, "type": "lxc", "expected_name": "janus-lab"}
    text = format_tool_result("proxmox_guest_delete", unknown)
    assert "unknown outcome" in text and "Do not repeat" in text


def test_mcp_uses_the_approved_free_vmid_and_reports_the_receipt():
    class Server:
        tools = {}

        def tool(self):
            return lambda function: self.tools.setdefault(function.__name__, function)

    class Provider:
        def __init__(self):
            self.calls = []

        def guest_clone(self, node, vmid, *, newid, name, full, pool, expected_type, expected_name):
            self.calls.append((node, vmid, newid, name, full, expected_type, expected_name))
            return {"ok": True, "status": "executed", "provider": "proxmox", "operation": "guest_clone",
                    "node": node, "vmid": newid, "type": expected_type, "name": name, "upid": "UPID:x",
                    "changed": True, "verification_ok": True, "retry_safe": False, "error": None}

    server = Server()
    provider = Provider()
    register_proxmox_tools(server, provider)
    assert CHANGE_TOOLS <= set(server.tools)
    result = server.tools["proxmox_guest_clone"](
        "hummingbird", 990, "lab-clone", True, expected_type="lxc", expected_name="janus-lab",
        allocated_vmid=992)
    assert (result.ok, result.verification_ok, result.vmid) == (True, True, 992)
    assert provider.calls == [("hummingbird", 990, 992, "lab-clone", True, "lxc", "janus-lab")]


def test_mcp_never_acts_without_an_approved_vmid_for_a_new_guest():
    class Server:
        tools = {}

        def tool(self):
            return lambda function: self.tools.setdefault(function.__name__, function)

    server = Server()
    register_proxmox_tools(server, None, missing_settings=("PROXMOX_BASE_URL",))
    result = server.tools["proxmox_guest_create"](
        "hummingbird", "lab-2", "local:vztmpl/alpine.tar.xz", 1, 128, 2, "local-lvm")
    assert (result.ok, result.status) == (False, "error")
    assert "No VMID" in result.error
