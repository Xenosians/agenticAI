import pytest

from tools.capability_contract import (
    build_capability_contract,
    validate_capability_registry,
)
from tools.registry import TOOLS
from tools.semantic_catalog import (
    CAPABILITY_SEMANTICS,
    SEMANTIC_FIELDS,
)


def test_semantic_catalog_has_exact_registry_coverage():
    assert set(CAPABILITY_SEMANTICS) == set(TOOLS)


def test_every_capability_has_full_explicit_semantics():
    for tool_name, metadata in TOOLS.items():
        expected = CAPABILITY_SEMANTICS[tool_name]

        for field in SEMANTIC_FIELDS:
            assert metadata[field] == expected[field]

    validate_capability_registry(
        TOOLS,
        require_runtime_schema=False,
    )


@pytest.mark.parametrize(
    "field",
    [
        "resource_type",
        "operation_kind",
        "effect",
        "permission",
    ],
)
def test_missing_semantic_field_is_never_inferred(field: str):
    metadata = {
        "resource_type": "ticket",
        "operation_kind": "read",
        "effect": "read",
        "permission": "ticket.read",
        "risk": "read",
        "requires_approval": False,
        "required_arguments": ["ticket_key"],
        "grounded_arguments": ["ticket_key"],
        "parameters": {
            "ticket_key": {
                "type": "str",
            },
        },
    }

    metadata.pop(field)

    with pytest.raises(
        ValueError,
        match="requires explicit " + field + " metadata",
    ):
        build_capability_contract(
            "ticket_get",
            metadata,
        )


def test_action_permissions_are_not_collapsed_to_generic_action():
    expected = {
        "unlock_user": "account.unlock",
        "reset_password": "account.reset_password",
        "enable_user": "account.enable",
        "disable_user": "account.disable",
        "grant_access": "access.grant",
        "revoke_access": "access.revoke",
        "asset_assign": "asset.assign",
        "asset_unassign": "asset.unassign",
        "ticket_assign": "ticket.assign",
        "ticket_transition": "ticket.transition",
        "workspace_git_commit": "workspace.git.commit",
        "workspace_git_push": "workspace.git.push",
        "workspace_git_switch_branch": "workspace.git.branch.switch",
        "process_exec": "workspace.process.execute",
        "workspace_run_build": "workspace.build.run",
        "workspace_run_tests": "workspace.test.run",
    }

    for tool_name, permission in expected.items():
        assert TOOLS[tool_name]["permission"] == permission
