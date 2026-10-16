#!/usr/bin/env python3
"""Import an operator-supplied JSON inventory into the configured SQLite provider."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Settings
from services.assets import SQLiteAssetService, InventoryImportError, validate_inventory_records


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InventoryImportError("JSON objects cannot contain duplicate field names.")
        result[key] = value
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="JSON array of complete inventory records.")
    parser.add_argument("--replace", action="store_true", help="Explicitly replace existing IDs; other records are retained.")
    parser.add_argument("--validate-only", action="store_true", help="Validate input/configuration without creating a database or importing records.")
    args = parser.parse_args(argv)
    try:
        with args.source.open("rb") as source:
            content = source.read(8 * 1024 * 1024 + 1)
        if len(content) > 8 * 1024 * 1024:
            raise InventoryImportError("Inventory input exceeds 8 MiB.")
        records = json.loads(content.decode("utf-8"), object_pairs_hook=_unique_fields)
        validated = validate_inventory_records(records)
        settings = Settings()
        if settings.asset_backend != "sqlite" or settings.asset_sqlite_path is None:
            raise InventoryImportError("Configure ASSET_BACKEND=sqlite and ASSET_SQLITE_PATH before importing.")
        if args.validate_only:
            print(json.dumps({"outcome": "validated", "record_count": len(validated), "database_modified": False}))
        else:
            count = SQLiteAssetService(settings.asset_sqlite_path).import_records(records, replace=args.replace)
            print(json.dumps({"outcome": "imported", "record_count": count, "replace_existing": args.replace}))
        return 0
    except (ValueError, OSError, sqlite3.Error) as exc:
        detail = str(exc) if isinstance(exc, InventoryImportError) else "Inventory input, configuration, or storage is invalid."
        print(json.dumps({"outcome": "failed", "error_type": type(exc).__name__, "detail": detail}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
