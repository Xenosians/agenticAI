from pathlib import Path
from types import SimpleNamespace

from services.extensions.service import workflow_capabilities


def test_catalog_discloses_registered_owners_and_only_grounded_field_descriptors():
    settings = SimpleNamespace(agents_dir=Path(__file__).resolve().parents[3] / "subagents/agents")
    rows = {row["id"]: row for row in workflow_capabilities(settings)}
    account = rows["account_create"]
    ticket = rows["ticket_create"]
    assert account["agent_names"] == ["account-specialist"]
    assert "ticket_create" in account["success_continuations"]
    assert {"given_name", "family_name"} <= set(account["grounded_arguments"])
    # A capability shared by a provider-specific and a provider-neutral
    # specialist stays selectable; every registered owner is disclosed.
    assert "jira-specialist" in ticket["agent_names"]
    assert ticket["agent_names"] == sorted(ticket["agent_names"])
    assert "project_key" in ticket["grounded_arguments"]
    assert "summary" not in ticket["parameters"]
    for row in rows.values():
        assert set(row["parameters"]) <= set(row["grounded_arguments"])
        for descriptor in row["parameters"].values():
            assert set(descriptor) <= {"type", "description", "enum"}
        assert not {"api_key", "password", "policy_resolver", "function", "credentials"} & set(row)


def test_catalog_discloses_each_capability_effect_risk_and_approval_from_its_contract():
    # The workspace's capability page groups and labels operations from these
    # fields, so they must come from the trusted contract, never a UI guess.
    from tools.capability_contract import build_capability_contract
    from tools.registry import TOOLS

    settings = SimpleNamespace(agents_dir=Path(__file__).resolve().parents[3] / "subagents/agents")
    rows = {row["id"]: row for row in workflow_capabilities(settings)}
    assert rows.keys() == TOOLS.keys()
    for name, row in rows.items():
        contract = build_capability_contract(name, TOOLS[name])
        assert (row["effect"], row["risk"], row["requires_approval"], row["resource_type"]) == (
            contract.effect, contract.risk, contract.requires_approval, contract.resource_type), name
    assert rows["account_create"]["requires_approval"] is True
    assert rows["ticket_get"]["requires_approval"] is False


def test_capabilities_without_loaded_owners_are_not_workflow_choices():
    rows = workflow_capabilities(SimpleNamespace())
    assert rows
    assert all(row["agent_names"] == [] for row in rows)


def test_each_capability_names_the_provider_whose_setup_it_needs():
    # The workspace joins each operation to its provider's setup state (FR-081).
    from services.integration_readiness import provider_for_capability

    settings = SimpleNamespace(agents_dir=Path(__file__).resolve().parents[3] / "subagents/agents")
    rows = {row["id"]: row for row in workflow_capabilities(settings)}
    assert rows["ticket_create"]["provider"] == "ticketing"
    assert rows["account_create"]["provider"] == "directory"
    assert rows["network_dhcp_leases"]["provider"] == "openwrt"
    assert rows["workspace_git_status"]["provider"] == "workspace"
    for name, row in rows.items():
        assert row["provider"] == provider_for_capability(name), name
