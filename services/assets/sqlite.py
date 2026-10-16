"""Persistent local inventory; importing records is an operator action, not a tool."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .base import AssetService
from .types import (
    AssetLookupResult, AssetMutationResult, AssetRecord, AssetSearchQuery, AssetSearchResult,
)


class InventoryImportError(ValueError):
    """A complete import was rejected without modifying inventory records."""


class InventoryRecord(BaseModel):
    """Operator input; provider identity is supplied by the implementation."""

    model_config = ConfigDict(extra="forbid", strict=True)

    asset_id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=200)
    asset_type: str = Field(min_length=1, max_length=100)
    status: Literal["available", "in_use", "repair", "retired"]
    owner_id: str | None = Field(default=None, min_length=1, max_length=100)
    serial_number: str | None = Field(default=None, min_length=1, max_length=200)
    platform: str | None = Field(default=None, min_length=1, max_length=200)
    location: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("asset_id", "name", "asset_type", "owner_id", "serial_number", "platform", "location")
    @classmethod
    def visible_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value or any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError("Inventory fields must contain visible text.")
        return value

    @field_validator("asset_id")
    @classmethod
    def normalize_id(cls, value: str) -> str:
        return value.upper()

    @field_validator("owner_id")
    @classmethod
    def normalize_owner(cls, value: str | None) -> str | None:
        return value.lower() if value is not None else None

    @model_validator(mode="after")
    def ownership_matches_status(self):
        if self.status == "available" and self.owner_id is not None:
            raise ValueError("Available assets cannot have an owner.")
        if self.status == "in_use" and self.owner_id is None:
            raise ValueError("In-use assets require an owner.")
        return self


def validate_inventory_records(records: object) -> list[InventoryRecord]:
    if not isinstance(records, list) or not 1 <= len(records) <= 10_000:
        raise InventoryImportError("Import must contain between 1 and 10000 inventory records.")
    validated = []
    seen = set()
    for index, value in enumerate(records):
        if not isinstance(value, dict):
            raise InventoryImportError(f"Inventory record {index + 1} must be an object.")
        try:
            record = InventoryRecord.model_validate(value)
        except ValueError as exc:
            # ValidationError text can contain private field values.
            raise InventoryImportError(f"Inventory record {index + 1} is invalid.") from exc
        if record.asset_id in seen:
            raise InventoryImportError("Import contains duplicate asset IDs.")
        seen.add(record.asset_id)
        validated.append(record)
    return validated


_COLUMNS = "asset_id, name, asset_type, status, owner_id, serial_number, platform, location"
_FIELDS = _COLUMNS.split(", ")


class SQLiteAssetService(AssetService):
    """Dedicated SQLite inventory, shared by all four existing asset capabilities."""

    def __init__(self, database_path: Path | str):
        if str(database_path).strip() in {"", ":memory:"}:
            raise ValueError("A persistent asset database path is required.")
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path.touch(mode=0o600, exist_ok=True)
        with self._connection() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS asset_inventory (
                    asset_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    asset_type TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('available','in_use','repair','retired')),
                    owner_id TEXT,
                    serial_number TEXT,
                    platform TEXT,
                    location TEXT,
                    search_text TEXT NOT NULL,
                    asset_type_key TEXT NOT NULL,
                    CHECK (status != 'available' OR owner_id IS NULL),
                    CHECK (status != 'in_use' OR owner_id IS NOT NULL)
                )
            """)
            connection.execute("CREATE INDEX IF NOT EXISTS asset_inventory_owner ON asset_inventory(owner_id)")
            connection.execute("CREATE INDEX IF NOT EXISTS asset_inventory_status_type ON asset_inventory(status, asset_type_key)")

    @contextmanager
    def _connection(self, *, write: bool = False):
        # Rollback journaling also works for a WSL process on a Windows mount.
        connection = sqlite3.connect(self.database_path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA synchronous=FULL")
            if write:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            if write:
                connection.commit()
        except BaseException:
            if write:
                connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _asset(row: sqlite3.Row) -> AssetRecord:
        return AssetRecord(provider="sqlite", **dict(row))

    @staticmethod
    def _identifier(value: object, *, owner: bool = False) -> str:
        if not isinstance(value, str) or not 1 <= len(value.strip()) <= 100:
            raise ValueError("user_id must be a nonempty string of at most 100 characters."
                             if owner else "asset_id must be a nonempty string of at most 100 characters.")
        value = value.strip()
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError("Identifiers must contain visible text.")
        return value.lower() if owner else value.upper()

    def import_records(self, records: object, *, replace: bool = False) -> int:
        """Validate the whole batch, then commit all records or none of them."""
        validated = validate_inventory_records(records)
        statement = f"INSERT INTO asset_inventory ({_COLUMNS}, search_text, asset_type_key) VALUES (?,?,?,?,?,?,?,?,?,?)"
        if replace:
            statement += " ON CONFLICT(asset_id) DO UPDATE SET " + ", ".join(
                f"{column}=excluded.{column}"
                for column in (*_FIELDS[1:], "search_text", "asset_type_key")
            )
        rows = []
        for record in validated:
            values = record.model_dump()
            search_text = "\n".join(values[key] or "" for key in (
                "asset_id", "name", "serial_number", "platform", "location"
            )).casefold()
            rows.append((*(values[key] for key in _FIELDS), search_text, record.asset_type.casefold()))
        try:
            with self._connection(write=True) as connection:
                connection.executemany(statement, rows)
        except sqlite3.IntegrityError as exc:
            raise InventoryImportError("Import conflicts with an existing asset; no records were imported.") from exc
        return len(validated)

    def get_asset(self, asset_id: str) -> AssetLookupResult:
        try:
            normalized = self._identifier(asset_id)
        except ValueError as exc:
            return AssetLookupResult(ok=False, status="denied", error=str(exc))
        try:
            with self._connection() as connection:
                row = connection.execute(f"SELECT {_COLUMNS} FROM asset_inventory WHERE asset_id=?", (normalized,)).fetchone()
        except sqlite3.Error:
            return AssetLookupResult(ok=False, status="error", error="Asset inventory storage is unavailable.")
        if row is None:
            return AssetLookupResult(ok=False, status="not_found", error=f"Asset '{normalized}' was not found.")
        return AssetLookupResult(ok=True, status="success", asset=self._asset(row))

    def search_assets(self, query: AssetSearchQuery) -> AssetSearchResult:
        if not isinstance(query, AssetSearchQuery):
            return AssetSearchResult(ok=False, status="denied", error="Invalid asset search query.")
        try:
            # Revalidate even constructed/mutated instances before SQL LIMIT.
            query = AssetSearchQuery.model_validate(query.model_dump())
        except ValueError:
            return AssetSearchResult(ok=False, status="denied", error="Invalid asset search query.")
        clauses, parameters = [], []
        if query.text is not None:
            clauses.append("instr(search_text, ?) > 0")
            parameters.append(query.text.casefold())
        for key, value in (("owner_id", query.owner_id), ("asset_type_key", query.asset_type), ("status", query.status)):
            if value is not None:
                clauses.append(f"{key}=?")
                parameters.append(value.lower() if key == "owner_id" else value.casefold())
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        try:
            with self._connection() as connection:
                rows = connection.execute(
                    f"SELECT {_COLUMNS} FROM asset_inventory{where} ORDER BY asset_id LIMIT ?",
                    (*parameters, query.limit + 1),
                ).fetchall()
        except sqlite3.Error:
            return AssetSearchResult(ok=False, status="error", error="Asset inventory storage is unavailable.")
        selected = [self._asset(row) for row in rows[:query.limit]]
        return AssetSearchResult(ok=True, status="success", assets=selected, count=len(selected), truncated=len(rows) > query.limit)

    def assign_asset(self, asset_id: str, user_id: str) -> AssetMutationResult:
        return self._change_owner(asset_id, user_id, assign=True)

    def unassign_asset(self, asset_id: str) -> AssetMutationResult:
        return self._change_owner(asset_id, None, assign=False)

    def _change_owner(self, asset_id: str, user_id: str | None, *, assign: bool) -> AssetMutationResult:
        try:
            normalized = self._identifier(asset_id)
            owner = self._identifier(user_id, owner=True) if assign else None
        except ValueError as exc:
            return AssetMutationResult(ok=False, status="denied", provider="sqlite", error=str(exc))
        try:
            with self._connection(write=True) as connection:
                row = connection.execute(f"SELECT {_COLUMNS} FROM asset_inventory WHERE asset_id=?", (normalized,)).fetchone()
                if row is None:
                    return AssetMutationResult(ok=False, status="not_found", provider="sqlite", asset_id=normalized,
                                               error=f"Asset '{normalized}' was not found.")
                asset = self._asset(row)
                common = {"provider": "sqlite", "asset_id": normalized, "owner_id": asset.owner_id}
                if asset.status in {"repair", "retired"}:
                    return AssetMutationResult(ok=False, status="denied", **common,
                                               error=f"The asset cannot change assignment while its status is '{asset.status}'.")
                if asset.owner_id == owner:
                    return AssetMutationResult(ok=True, status="executed", changed=False, **common,
                                               message=f"Asset '{normalized}' already has the requested assignment.")
                if assign and asset.owner_id is not None:
                    return AssetMutationResult(ok=False, status="denied", **common,
                                               error="The asset is assigned to another user; unassign it explicitly before reassignment.")
                status = "in_use" if assign else "available"
                connection.execute("UPDATE asset_inventory SET owner_id=?, status=? WHERE asset_id=?", (owner, status, normalized))
                persisted = connection.execute("SELECT owner_id, status FROM asset_inventory WHERE asset_id=?", (normalized,)).fetchone()
                if persisted is None or persisted["owner_id"] != owner or persisted["status"] != status:
                    raise sqlite3.IntegrityError("Asset assignment readback did not match.")
                return AssetMutationResult(ok=True, status="executed", provider="sqlite", changed=True,
                                           asset_id=normalized, owner_id=owner,
                                           message=f"Asset '{normalized}' was assigned to '{owner}'." if assign else f"Asset '{normalized}' was unassigned.")
        except sqlite3.Error:
            return AssetMutationResult(ok=False, status="error", provider="sqlite", asset_id=normalized,
                                       error="Asset inventory storage is unavailable; verify its state before retrying.")
