from __future__ import annotations

import sys

from pathlib import Path


REPOSITORY_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPOSITORY_ROOT),
    )


from mcp_server import create_mcp_server
from tools.capability_contract import (
    normalize_capability_catalog,
    validate_capability_registry,
)
from tools.registry import TOOLS
from tools.runtime_schema import (
    clear_runtime_schemas,
    runtime_registered_tools,
)


def main() -> None:
    # Make this audit deterministic even when imported/executed from a
    # process that previously registered synthetic test capabilities.
    clear_runtime_schemas()

    # Registration captures executable signatures. The returned server
    # does not need to start its transport.
    create_mcp_server()

    contracts = validate_capability_registry(
        TOOLS,
        require_runtime_schema=True,
    )

    normalized = normalize_capability_catalog(
        TOOLS
    )

    print("Capability Contract v1")
    print("======================")
    print(
        "registered_mcp_tools:",
        len(runtime_registered_tools()),
    )
    print(
        "validated_catalog_tools:",
        len(contracts),
    )
    print()

    for contract in contracts:
        item = normalized[contract.name]

        print(
            f"{item['name']}: "
            f"resource={item['resource_type']} "
            f"operation={item['operation_kind']} "
            f"effect={item['effect']} "
            f"permission={item['permission']} "
            f"risk={item['risk']} "
            f"approval={item['requires_approval']} "
            f"required={item['required_arguments']} "
            f"optional={item['optional_arguments']} "
            f"grounded={item['grounded_arguments']} "
            f"derived={item['derived_arguments']} "
            f"trusted_policy="
            f"{item['trusted_policy_arguments']}"
        )


if __name__ == "__main__":
    main()
