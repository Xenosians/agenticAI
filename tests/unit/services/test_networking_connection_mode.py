import asyncio
from types import SimpleNamespace

import pytest

from services.extensions import service
from services.extensions.context import (
    execution_profile, networking_inference_stage, owner_execution, selected_profile,
)
from services.extensions.options import networking_inference_mode, validate_model_options
from services.extensions.store import ExtensionStore
from subagents.llm.runtime.stages import InferenceStage


@pytest.mark.parametrize("value", ["", "selected", True, False, 0, [], {}, None])
def test_networking_mode_rejects_unknown_or_coerced_values(value):
    with pytest.raises(ValueError, match="networking_inference"):
        validate_model_options({"backend": "local", "networking_inference": value})


@pytest.mark.parametrize("backend", ["http", "pi", "opencode", "local"])
@pytest.mark.parametrize("mode", ["dedicated_local", "selected_model"])
def test_model_creation_admits_explicit_networking_mode_without_selecting_it(tmp_path, monkeypatch, backend, mode):
    store = ExtensionStore(tmp_path)
    monkeypatch.setattr(service, "extension_store", lambda: store)
    monkeypatch.setattr(service, "harness_binary", lambda kind: tmp_path / kind)
    settings = SimpleNamespace(model_profiles={"installed": SimpleNamespace(enabled=True)})
    payload = {"name": "Connection", "backend": backend, "networking_inference": mode,
               "model": "installed" if backend == "local" else "groq/unit-model",
               "api_key": "private-unit-key", "url": "https://api.groq.com/openai/v1"}
    result = asyncio.run(service.command("alice", "model", payload, settings))
    assert result["networking_inference"] == mode
    assert "api_key" not in result
    assert store.get("alice", result["id"], secret=True)["api_key"] == "private-unit-key"
    assert store.selected("alice") == {"model_id": None, "dataset_id": None}


@pytest.mark.parametrize("backend", ["http", "pi", "opencode", "local", "uploaded"])
def test_networking_option_update_preserves_credentials_identity_selection_and_model(tmp_path, monkeypatch, backend):
    store = ExtensionStore(tmp_path)
    monkeypatch.setattr(service, "extension_store", lambda: store)
    saved = store.put("alice", "model", {
        "backend": backend, "model": "existing-model", "name": "Existing",
        "url": "https://api.groq.com/openai/v1", "model_path": "/private/model",
        "state": "ready",
    }, "existing-private-key")
    store.select("alice", saved["id"], None)
    before = store.get("alice", saved["id"], secret=True)
    with store.connect() as db:
        encrypted_before = bytes(db.execute("SELECT secret FROM resource WHERE owner=? AND id=?",
                                           ("alice", saved["id"])).fetchone()[0])
    result = asyncio.run(service.command("alice", "model_options", {
        "id": saved["id"], "networking_inference": "selected_model",
    }, SimpleNamespace()))
    assert result["networking_inference"] == "selected_model"
    assert "api_key" not in result and "model_path" not in result
    after = store.get("alice", saved["id"], secret=True)
    assert {key: value for key, value in after.items() if key != "networking_inference"} == before
    assert store.selected("alice")["model_id"] == saved["id"]
    with store.connect() as db:
        encrypted_after = bytes(db.execute("SELECT secret FROM resource WHERE owner=? AND id=?",
                                          ("alice", saved["id"])).fetchone()[0])
    assert encrypted_after == encrypted_before
    with pytest.raises(ValueError, match="networking_inference"):
        store.update_model_options("alice", saved["id"], {"networking_inference": "arbitrary"})
    assert store.get("alice", saved["id"]) == {key: value for key, value in after.items() if key != "api_key"}
    with pytest.raises(ValueError, match="not found"):
        store.update_model_options("bob", saved["id"], {"networking_inference": "selected_model"})
    cleared = store.update_model_options("alice", saved["id"], {"networking_inference": None})
    assert "networking_inference" not in cleared
    assert networking_inference_mode(cleared) == "dedicated_local"


def test_model_creation_default_and_invalid_override(tmp_path, monkeypatch):
    store = ExtensionStore(tmp_path)
    monkeypatch.setattr(service, "extension_store", lambda: store)
    settings = SimpleNamespace(model_profiles={"installed": SimpleNamespace(enabled=True)})
    payload = {"name": "Existing defaults", "backend": "local", "model": "installed"}
    result = asyncio.run(service.command("alice", "model", payload, settings))
    assert "networking_inference" not in result
    assert networking_inference_mode(result) == "dedicated_local"
    for invalid in (None, "remote", True):
        with pytest.raises(ValueError, match="networking_inference"):
            asyncio.run(service.command("alice", "model", {**payload, "networking_inference": invalid}, settings))
    assert len(store.list("alice")) == 1


def test_catalog_modes_are_detached_and_current_owner_preferences_are_unchanged(tmp_path, monkeypatch):
    store = ExtensionStore(tmp_path)
    monkeypatch.setattr(service, "extension_store", lambda: store)
    monkeypatch.setattr(service, "_installed", lambda kind: False)
    catalog = asyncio.run(service.command("alice", "list", {}, SimpleNamespace(model_profiles={})))
    modes = catalog["networking_inference_options"]
    assert modes["default"] == "dedicated_local"
    assert [row["id"] for row in modes["choices"]] == ["dedicated_local", "selected_model"]
    modes["choices"].clear()
    repeated = asyncio.run(service.command("alice", "list", {}, SimpleNamespace(model_profiles={})))
    assert len(repeated["networking_inference_options"]["choices"]) == 2
    assert catalog["selected"] == store.selected("alice")


def test_concurrent_owners_cannot_cross_networking_modes(tmp_path, monkeypatch):
    store = ExtensionStore(tmp_path)
    monkeypatch.setattr(service, "extension_store", lambda: store)
    for owner, extra in (("alice", {"networking_inference": "selected_model"}), ("bob", {})):
        saved = store.put(owner, "model", {"backend": "http", "model": "same-model", "state": "ready", **extra})
        store.select(owner, saved["id"], None)

    async def check(owner, expected):
        with owner_execution(owner, "Read network system info"):
            await asyncio.sleep(0)
            assert networking_inference_stage() is expected
            assert (selected_profile(stage=expected) is not None) == (owner == "alice")

    async def run():
        await asyncio.gather(check("alice", InferenceStage.NETWORKING_SELECTED),
                             check("bob", InferenceStage.NETWORKING))

    asyncio.run(run())
    assert execution_profile.get() is None


@pytest.mark.parametrize("origin", [None, InferenceStage.NETWORKING_SELECTED])
def test_optional_stage_alone_cannot_disclose_to_an_unopted_connection(origin):
    token = execution_profile.set({"backend": "http", "model": "not-opted-in"})
    try:
        stage = InferenceStage.NETWORKING_SELECTED if origin is None else InferenceStage.REPAIR
        assert selected_profile(stage=stage, repair_of=origin) is None
        assert networking_inference_stage() is InferenceStage.NETWORKING
    finally:
        execution_profile.reset(token)
