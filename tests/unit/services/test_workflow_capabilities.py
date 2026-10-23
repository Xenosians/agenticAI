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


def test_capabilities_without_loaded_owners_are_not_workflow_choices():
    rows = workflow_capabilities(SimpleNamespace())
    assert rows
    assert all(row["agent_names"] == [] for row in rows)
