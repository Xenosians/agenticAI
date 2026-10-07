from mcp_server import create_mcp_server
from tools.capability_contract import validate_capability_registry
from tools.registry import TOOLS
from tools.runtime_schema import (
    clear_runtime_schemas,
    runtime_registered_tools,
)


def test_all_registered_capabilities_have_standard_contract():
    # Other contract unit tests intentionally register synthetic capabilities.
    # Rebuild this audit from a clean runtime-schema namespace so this test
    # represents only production MCP capabilities.
    clear_runtime_schemas()

    # Registration captures executable Python signatures without starting
    # the MCP transport. Requiredness therefore comes from real callables.
    create_mcp_server()

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
