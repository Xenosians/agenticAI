from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

from pydantic import ValidationError
import pytest

from config import Settings
from scripts import asset_inventory as cli
from services.assets import (
    AssetSearchQuery, InventoryImportError, SQLiteAssetService,
    build_asset_mutation_service, build_asset_service,
)
from tools.assets.mcp import register_asset_tools


def record(asset_id="FIXTURE-1", **overrides):
    return {"asset_id": asset_id, "name": "Synthetic test inventory", "asset_type": "laptop",
            "status": "available", **overrides}


def test_sqlite_settings_require_explicit_persistent_path(tmp_path):
    for value in (None, "", ":memory:"):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, asset_backend="sqlite", asset_sqlite_path=value)
    path = tmp_path / "inventory.sqlite3"
    settings = Settings(_env_file=None, asset_backend=" SQLITE ", asset_sqlite_path=path)
    assets = build_asset_service(settings)
    assert isinstance(assets, SQLiteAssetService)
    assert assets.search_assets(AssetSearchQuery()).count == 0
    assert assets.get_asset("LAP-001").status == "not_found"  # No mock seeding.


def test_assign_and_unassign_survive_provider_reopen_with_readback(tmp_path):
    path = tmp_path / "inventory.sqlite3"
    assets = SQLiteAssetService(path)
    assets.import_records([record()])
    mutations = build_asset_mutation_service(assets)
    first = mutations.assign_asset(" fixture-1 ", " Case.User ")
    assert first.ok and first.changed and first.provider == "sqlite"

    reopened = SQLiteAssetService(path)
    persisted = reopened.get_asset("fixture-1").asset
    assert persisted.owner_id == "case.user" and persisted.status == "in_use"
    repeated = build_asset_mutation_service(reopened).assign_asset("FIXTURE-1", "case.user")
    assert repeated.ok and not repeated.changed
    assert build_asset_mutation_service(reopened).unassign_asset("FIXTURE-1").changed
    final = SQLiteAssetService(path)
    assert final.get_asset("FIXTURE-1").asset.owner_id is None
    assert final.get_asset("FIXTURE-1").asset.status == "available"
    assert not final.unassign_asset("FIXTURE-1").changed


def test_concurrent_providers_cannot_overwrite_another_owner(tmp_path):
    path = tmp_path / "inventory.sqlite3"
    SQLiteAssetService(path).import_records([record()])
    providers = [SQLiteAssetService(path), SQLiteAssetService(path)]
    ready = Barrier(2)

    def assign(index):
        ready.wait(timeout=10)
        return providers[index].assign_asset("FIXTURE-1", f"fixture-owner-{index}")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(assign, range(2)))
    assert sum(result.ok and result.changed for result in results) == 1
    assert sum(result.status == "denied" for result in results) == 1
    winner = next(result.owner_id for result in results if result.ok)
    assert SQLiteAssetService(path).get_asset("FIXTURE-1").asset.owner_id == winner


@pytest.mark.parametrize("status", ["repair", "retired"])
def test_protected_asset_status_denies_assignment_changes(tmp_path, status):
    assets = SQLiteAssetService(tmp_path / "inventory.sqlite3")
    assets.import_records([record(status=status, owner_id="fixture-owner")])
    assert assets.assign_asset("FIXTURE-1", "other").status == "denied"
    assert assets.unassign_asset("FIXTURE-1").status == "denied"
    persisted = assets.get_asset("FIXTURE-1").asset
    assert persisted.status == status and persisted.owner_id == "fixture-owner"


def test_lookup_mutation_errors_do_not_create_inventory_records(tmp_path):
    assets = SQLiteAssetService(tmp_path / "inventory.sqlite3")
    assert assets.get_asset(123).status == "denied"
    assert assets.assign_asset("", "fixture-owner").status == "denied"
    assert assets.assign_asset("missing", "fixture-owner").status == "not_found"
    assets.import_records([record()])
    assert assets.assign_asset("FIXTURE-1", "").status == "denied"
    assert assets.search_assets(AssetSearchQuery()).count == 1


def test_search_filters_are_combined_case_insensitive_and_bounded(tmp_path):
    assets = SQLiteAssetService(tmp_path / "inventory.sqlite3")
    assets.import_records([
        record("FIXTURE-B", name="Straße 100%", owner_id="fixture-owner", status="in_use", location="Test lab"),
        record("FIXTURE-A", name="Straße 100%", owner_id="fixture-owner", status="in_use", location="Test lab"),
        record("FIXTURE-C", asset_type="phone"),
    ])
    found = assets.search_assets(AssetSearchQuery(text="STRASSE", owner_id="FIXTURE-OWNER", asset_type="LAPTOP", status="IN_USE", limit=1))
    assert found.ok and found.count == 1 and found.truncated
    assert found.assets[0].asset_id == "FIXTURE-A"
    assert assets.search_assets(AssetSearchQuery(text="100%")).count == 2
    assert assets.search_assets(AssetSearchQuery(text="%' OR 1=1 --")).count == 0
    assert assets.get_asset("' OR 1=1 --").status == "not_found"
    assert assets.search_assets(AssetSearchQuery(owner_id="other")).count == 0
    assert assets.search_assets(AssetSearchQuery.model_construct(limit=100_000)).status == "denied"


@pytest.mark.parametrize("invalid", [
    record(name=""), record(status="unknown"), record(status="available", owner_id="owner"),
    record(status="in_use"), record(provider="mock"), record(location="test\ncontrol"),
    record(serial_number=123),
])
def test_invalid_import_batch_never_partially_inserts(tmp_path, invalid):
    assets = SQLiteAssetService(tmp_path / "inventory.sqlite3")
    with pytest.raises(InventoryImportError):
        assets.import_records([record("VALID-FIRST"), invalid])
    assert assets.search_assets(AssetSearchQuery()).count == 0


def test_default_import_conflict_rolls_back_prior_insert_and_preserves_existing(tmp_path):
    assets = SQLiteAssetService(tmp_path / "inventory.sqlite3")
    assets.import_records([record(name="Original")])
    with pytest.raises(InventoryImportError):
        assets.import_records([record("NEW-FIRST"), record(name="Overwrite attempt")])
    assert assets.get_asset("NEW-FIRST").status == "not_found"
    assert assets.get_asset("FIXTURE-1").asset.name == "Original"
    with pytest.raises(InventoryImportError):
        assets.import_records([record("DUPLICATE"), record("duplicate")], replace=True)
    assert assets.get_asset("DUPLICATE").status == "not_found"


def test_explicit_replacement_only_changes_supplied_ids(tmp_path):
    path = tmp_path / "inventory.sqlite3"
    assets = SQLiteAssetService(path)
    assets.import_records([record(), record("UNCHANGED")])
    assert assets.import_records([record(name="Operator update")], replace=True) == 1
    reopened = SQLiteAssetService(path)
    assert reopened.get_asset("FIXTURE-1").asset.name == "Operator update"
    assert reopened.get_asset("UNCHANGED").asset.name == "Synthetic test inventory"


def test_import_cli_validation_is_no_write_and_duplicate_fields_are_rejected(tmp_path, monkeypatch, capsys):
    database = tmp_path / "inventory.sqlite3"
    source = tmp_path / "input.json"
    monkeypatch.setattr(cli, "Settings", lambda: SimpleNamespace(asset_backend="sqlite", asset_sqlite_path=database))
    source.write_text(json.dumps([record()]), encoding="utf-8")
    assert cli.main([str(source), "--validate-only"]) == 0
    assert not database.exists()
    assert cli.main([str(source)]) == 0
    assert cli.main([str(source)]) == 1
    assert SQLiteAssetService(database).search_assets(AssetSearchQuery()).count == 1
    source.write_text('[{"asset_id":"FIRST","asset_id":"SECOND"}]', encoding="utf-8")
    assert cli.main([str(source), "--replace"]) == 1
    output = capsys.readouterr()
    assert "Synthetic test inventory" not in output.err


def test_all_existing_mcp_asset_tools_share_persistent_provider(tmp_path):
    class Server:
        def __init__(self):
            self.functions = {}

        def tool(self):
            def register(function):
                self.functions[function.__name__] = function
                return function
            return register

    path = tmp_path / "inventory.sqlite3"
    assets = SQLiteAssetService(path)
    assets.import_records([record()])
    server = Server()
    register_asset_tools(server, assets, build_asset_mutation_service(assets))
    assert set(server.functions) == {"asset_get", "asset_search", "asset_assign", "asset_unassign"}
    assert server.functions["asset_search"](status="available").count == 1
    assert server.functions["asset_assign"]("FIXTURE-1", "fixture-owner").changed
    assert server.functions["asset_get"]("FIXTURE-1").asset.owner_id == "fixture-owner"
    assert SQLiteAssetService(path).get_asset("FIXTURE-1").asset.owner_id == "fixture-owner"
    assert server.functions["asset_unassign"]("FIXTURE-1").changed
    assert SQLiteAssetService(path).get_asset("FIXTURE-1").asset.owner_id is None
