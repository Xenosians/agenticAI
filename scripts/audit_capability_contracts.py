from __future__ import annotations


import sys

from pathlib import Path


REPOSITORY_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if (
    str(REPOSITORY_ROOT)
    not in sys.path
):
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
from tools.runtime_schema import runtime_registered_tools


def main() -> None:
    # Registration captures executable signatures. The returned server does
    # not need to be started for this static contract audit.
    create_mcp_server()

    contracts = validate_capability_registry(
        TOOLS,
        require_runtime_schema=True,
    )

    normalized = normalize_capability_catalog(TOOLS)

    print("Capability Contract v1")
    print("======================")
    print("registered_mcp_tools:", len(runtime_registered_tools()))
    print("validated_catalog_tools:", len(contracts))
    print()

    for contract in contracts:
        item = normalized[contract.name]
        print(
            f"{item['name']}: resource={item['resource_type']} "
            f"operation={item['operation_kind']} effect={item['effect']} "
            f"permission={item['permission']} risk={item['risk']} "
            f"approval={item['requires_approval']} "
            f"required={item['required_arguments']} optional={item['optional_arguments']} "
            f"grounded={item['grounded_arguments']} derived={item['derived_arguments']}"
        )


if __name__ == "__main__":
    main()
