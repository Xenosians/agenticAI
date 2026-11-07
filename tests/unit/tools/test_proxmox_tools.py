"""Catalog and setup-state coverage for the Proxmox read-only tool family."""
from config.settings import Settings
from services.integration_readiness import integration_readiness, provider_for_capability
from tools.proxmox.catalog import PROXMOX_TOOLS
from tools.proxmox.mcp import register_proxmox_tools
from tools.registry import TOOLS


def test_each_proxmox_read_is_registered_with_read_only_semantics_and_one_provider():
    assert set(PROXMOX_TOOLS) == {
        "proxmox_cluster_status", "proxmox_nodes", "proxmox_guests",
        "proxmox_node_storage", "proxmox_recent_tasks",
    }
    assert set(PROXMOX_TOOLS) <= set(TOOLS)
    for name in PROXMOX_TOOLS:
        assert provider_for_capability(name) == "proxmox"
        assert TOOLS[name]["effect"] == "read"
        assert TOOLS[name]["requires_approval"] is False


def test_proxmox_setup_reports_only_missing_setting_names():
    row = next(item for item in integration_readiness(Settings(_env_file=None)) if item["id"] == "proxmox")
    assert row["state"] == "not_configured"
    assert set(row["missing"]) == {"PROXMOX_BASE_URL", "PROXMOX_API_TOKEN_ID", "PROXMOX_API_TOKEN_SECRET"}
    assert row["configured"] is False


def test_mcp_registers_only_the_fixed_reads_and_unconfigured_calls_fail_honestly():
    class Server:
        tools = {}

        def tool(self):
            return lambda function: self.tools.setdefault(function.__name__, function)

    server = Server()
    register_proxmox_tools(server, None)
    assert set(server.tools) == set(PROXMOX_TOOLS)
    try:
        server.tools["proxmox_nodes"]()
    except RuntimeError as exc:
        assert "not configured" in str(exc)
    else:
        raise AssertionError("An unconfigured Proxmox provider was reported as successful.")
