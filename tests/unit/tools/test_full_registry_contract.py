from mcp_server import register_capability_signatures
from tools.capability_contract import validate_capability_registry
from tools.registry import TOOLS
from tools.registry import merge_capability_catalog
import pytest
from tools import runtime_schema
from tools.runtime_schema import (
    runtime_registered_tools,
)


def test_duplicate_registration_cannot_silently_replace_an_executor_contract():
    original = {"read": {"effect": "read"}}
    with pytest.raises(ValueError, match="Duplicate capability"):
        merge_capability_catalog(original, {"read": {"effect": "mutation"}})
    assert original["read"]["effect"] == "read"


def test_all_registered_capabilities_have_standard_contract(monkeypatch):
    # Other contract unit tests intentionally register synthetic capabilities.
    # Rebuild this audit from a clean runtime-schema namespace so this test
    # represents only production MCP capabilities.
    monkeypatch.setattr(runtime_schema, "_SCHEMAS", {})

    # Registration captures executable Python signatures without starting
    # the MCP transport. Requiredness therefore comes from real callables.
    register_capability_signatures()

    contracts = validate_capability_registry(
        TOOLS,
        require_runtime_schema=True,
    )

    assert len(contracts) == len(TOOLS)
    assert set(runtime_registered_tools()) == set(TOOLS)

    for contract in contracts:
        assert contract.resource_type
        assert contract.operation_kind in {
            "create",
            "read",
            "update",
            "delete",
            "search",
            "action",
        }
        assert contract.effect in {"read", "mutation"}
        assert contract.permission
        assert "." in contract.permission
        assert set(contract.required_arguments).isdisjoint(
            contract.optional_arguments
        )
        assert set(contract.required_arguments) | set(
            contract.optional_arguments
        ) == set(contract.parameters)

        # Runtime signatures may additionally contain deterministic
        # policy-injected values. They must never leak into the model-facing
        # parameter surface.
        assert set(contract.parameters).isdisjoint(
            contract.trusted_policy_arguments
        )
        assert set(contract.execution_parameters) == (
            set(contract.parameters)
            | set(contract.trusted_policy_arguments)
        )

        assert set(contract.grounded_arguments).isdisjoint(
            contract.derived_arguments
        )
        assert set(contract.quoted_arguments).isdisjoint(
            set(contract.grounded_arguments) | set(contract.derived_arguments)
        )
        assert set(contract.required_arguments) <= (
            set(contract.grounded_arguments) | set(contract.derived_arguments)
        ), f"{contract.name} has required arguments without a trusted semantic role"
