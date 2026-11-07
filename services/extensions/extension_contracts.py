"""Load and validate the bundled, versioned extension wire contracts."""
from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

CONTRACTS_ROOT = Path(__file__).with_name("contracts")
MANIFEST_NAME = "lua-row-transform.manifest.json"
MANIFEST_ID = "lua-row-transform"
MANIFEST_VERSION = "1.0.0"
CONTRACT_VERSION = 1


class ExtensionContractError(ValueError):
    """A content-free refusal for malformed or incompatible extension data."""


def load_extension_contract(root: Path | None = None) -> tuple[dict, dict, dict]:
    """Load the declared Lua transform and its schemas before any worker starts."""
    contract_root = root or CONTRACTS_ROOT
    try:
        manifest_schema = _read(contract_root / "manifest.schema.json")
        invocation_schema = _read(contract_root / "invocation.schema.json")
        result_schema = _read(contract_root / "result.schema.json")
        manifest = _read(contract_root / MANIFEST_NAME)

        for schema in (manifest_schema, invocation_schema, result_schema):
            Draft202012Validator.check_schema(schema)
        Draft202012Validator(manifest_schema).validate(manifest)

        if (manifest["id"], manifest["version"], manifest["contract_version"]) != (
                MANIFEST_ID, MANIFEST_VERSION, CONTRACT_VERSION):
            raise ExtensionContractError
        if (manifest["input_schema"] != invocation_schema.get("$id") or
                manifest["output_schema"] != result_schema.get("$id")):
            raise ExtensionContractError
    except ExtensionContractError:
        raise
    except (OSError, ValueError, TypeError, KeyError, SchemaError, ValidationError):
        raise ExtensionContractError from None
    return manifest, invocation_schema, result_schema


def validate_extension_document(schema: dict, document: dict) -> None:
    """Validate one invocation or result without exposing payloads in errors."""
    try:
        Draft202012Validator(schema).validate(document)
    except (SchemaError, ValidationError, TypeError):
        raise ExtensionContractError from None


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExtensionContractError
    return value
