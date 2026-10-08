from __future__ import annotations

from mcp_server import create_mcp_server
from tools.capability_contract import validate_capability_registry
from tools.registry import TOOLS
from tools.runtime_schema import runtime_registered_tools


def main() -> None:
    # Registration captures executable signatures. The returned server does
    # not need to be started for this static contract audit.
    create_mcp_server()

    contracts = validate_capability_registry(
        TOOLS,
        require_runtime_schema=True,
    )

    print("Capability Contract v1")
    print("======================")
    print("registered_mcp_tools:", len(runtime_registered_tools()))
    print("validated_catalog_tools:", len(contracts))

    for contract in contracts:
        print(
            f"{contract.name}: resource={contract.resource_type} "
            f"operation={contract.operation_kind} effect={contract.effect} "
            f"required={list(contract.required_arguments)} "
            f"optional={list(contract.optional_arguments)} "
            f"grounded={list(contract.grounded_arguments)} "
            f"derived={list(contract.derived_arguments)} "
            f"permission={contract.permission}"
        )


if __name__ == "__main__":
    main()
