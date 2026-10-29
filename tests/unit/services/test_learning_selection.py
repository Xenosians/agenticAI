"""Learning selection: the owner's catalog and a readiness check, never a run.

Both read the owner's real extension registry. The check pins the chosen model
and dataset version and says whether a candidate run can start on this host:
`ready` only for SFT with LoRA/QLoRA on owned local weights the host can read.
Every other selection is `blocked` with the gates it still needs. Checking never
starts, loads or promotes anything.
"""
import json
from pathlib import Path

import pytest

from services.extensions.store import ExtensionStore
from services.learning_control.application import (
    LearningSelectionError,
    SelectionRequest,
    learning_catalog,
    selection_preflight,
)

OWNER, OTHER = "owner-a", "owner-b"
LATER_GATES = ["fixed_evaluation_required", "separate_promotion_required"]
ENTRYPOINTS = {"sft": "scripts/train_recipe.py", "dpo": "scripts/train_preferences.py",
               "ppo": "scripts/train_ppo.py"}


def has_weights(row):
    return Path("/weights") if row.get("backend") in {"local", "uploaded"} else None


def no_weights(row):
    return None


def examples(count=12):
    return "\n".join(json.dumps({"messages": [{"role": "user", "content": f"question {index}"},
                                              {"role": "assistant", "content": f"answer {index}"}]})
                     for index in range(count))


def training(store, owner=OWNER, count=12, replaces=None):
    return store.dataset(owner, name="Ticket triage", text=examples(count), source="Owned tickets",
                         license="Owned", purpose="training", reviewed=True, replaces=replaces)


def reference(store, owner=OWNER):
    return store.dataset(owner, name="Router manual", text="Router maintenance needs a planned window.",
                         source="Owned manual", license="Owned", purpose="rag", reviewed=True)


def model(store, owner=OWNER, backend="local", state="ready", name="Hub model"):
    return store.put(owner, "model", {"name": name, "backend": backend, "model": "fixture", "state": state})


def uploaded(store, owner=OWNER):
    # The record shape written when an uploaded model passes admission.
    return store.put(owner, "model", {"name": "My weights", "backend": "uploaded", "model": "a" * 32,
                                      "model_path": "/state/user-models/owner/a", "files": {"config.json": "1" * 64},
                                      "state": "ready", "quantization": "bnb4", "estimated_vram_bytes": 1})


def choose(model_row, dataset_row, method="sft", role="hub", tuning="qlora", digest=None):
    return SelectionRequest(method=method, role=role, tuning=tuning, model_id=model_row["id"],
                            dataset_id=dataset_row["id"], dataset_sha256=digest or dataset_row["sha256"])


@pytest.fixture
def store(tmp_path):
    return ExtensionStore(tmp_path)


def test_catalog_lists_only_the_owners_models_and_training_datasets(store):
    local, outbound = model(store), model(store, backend="http", name="Outbound model")
    data = training(store)
    reference(store)
    model(store, owner=OTHER, name="Another owner's model")
    training(store, owner=OTHER)

    catalog = learning_catalog(OWNER, store, weights=has_weights)

    assert catalog["schema"] == "learning-catalog.v2"
    assert [row["id"] for row in catalog["methods"]] == ["sft", "dpo", "ppo"]
    assert catalog["models"] == [
        {"id": local["id"], "name": "Hub model", "backend": "local", "state": "ready",
         "source_kind": "local", "training_available": True},
        {"id": outbound["id"], "name": "Outbound model", "backend": "http", "state": "ready",
         "source_kind": "outbound", "training_available": False}]
    assert catalog["datasets"] == [
        {"id": data["id"], "name": "Ticket triage", "purpose": "training", "state": "ready",
         "sha256": data["sha256"], "lineage_id": data["id"], "version": 1,
         "train_count": 10, "validation_count": 2, "superseded_by": None}]
    assert catalog["promotion_enabled"] is False
    # Dataset bodies never leave the registry.
    assert "question 0" not in json.dumps(catalog)


def test_training_is_available_only_for_ready_local_weights_the_host_can_read(store):
    local, pending = model(store), model(store, state="pending", name="Pending")

    for weights, expected in ((has_weights, {local["id"]: True, pending["id"]: False}),
                              (no_weights, {local["id"]: False, pending["id"]: False})):
        catalog = learning_catalog(OWNER, store, weights=weights)
        assert {row["id"]: row["training_available"] for row in catalog["models"]} == expected


def test_an_uploaded_model_is_the_owners_local_weights(store):
    weights = uploaded(store)
    training(store)

    catalog = learning_catalog(OWNER, store, weights=has_weights)

    assert catalog["models"] == [{"id": weights["id"], "name": "My weights", "backend": "uploaded",
                                  "state": "ready", "source_kind": "local", "training_available": True}]
    assert "/state/" not in json.dumps(catalog)
    receipt = selection_preflight(OWNER, choose(weights, catalog["datasets"][0]), store, weights=has_weights)
    assert receipt["model"]["source_kind"] == "local"
    assert receipt["status"] == "ready" and receipt["blockers"] == []


def test_a_runnable_check_is_ready_and_pins_the_selection_without_running(store):
    hub, data = model(store), training(store)

    receipt = selection_preflight(OWNER, choose(hub, data), store, weights=has_weights)

    assert receipt["schema"] == "learning-selection-preflight.v2"
    assert receipt["status"] == "ready" and receipt["blockers"] == []
    assert receipt["later_gates"] == LATER_GATES
    assert (receipt["method"], receipt["role"], receipt["tuning"]) == ("sft", "hub", "qlora")
    assert receipt["entrypoint"] == "scripts/train_recipe.py"
    assert receipt["model"] == {"id": hub["id"], "backend": "local", "source_kind": "local", "state": "ready"}
    assert receipt["dataset"] == {"id": data["id"], "sha256": data["sha256"], "lineage_id": data["id"],
                                  "version": 1, "train_count": 10, "validation_count": 2}
    for flag in ("model_loaded", "live_tool_execution", "promotion_enabled"):
        assert receipt[flag] is False
    # The same selection always has the same inspection identity.
    assert selection_preflight(OWNER, choose(hub, data), store, weights=has_weights)["preflight_sha256"] == \
        receipt["preflight_sha256"]
    # The same selection on a host without its weights is a different, blocked check.
    blocked = selection_preflight(OWNER, choose(hub, data), store, weights=no_weights)
    assert blocked["status"] == "blocked" and blocked["blockers"] == ["local_weights_unavailable"]
    assert blocked["preflight_sha256"] != receipt["preflight_sha256"]


@pytest.mark.parametrize("method, role, tuning, blockers", [
    ("sft", "specialist", "lora", []),
    ("sft", "hub", "full", ["full_tuning_not_offered"]),
    ("dpo", "specialist", "qlora", ["host_dispatch_not_integrated", "reviewed_preference_partitions_required"]),
    ("ppo", "hub", "qlora", ["evaluated_adapter_checkpoint_required", "host_dispatch_not_integrated",
                             "independent_simulator_episodes_required"]),
])
def test_each_method_reports_its_own_prerequisites(store, method, role, tuning, blockers):
    hub, data = model(store), training(store)

    receipt = selection_preflight(OWNER, choose(hub, data, method, role, tuning), store, weights=has_weights)

    assert receipt["blockers"] == blockers
    assert receipt["status"] == ("ready" if not blockers else "blocked")
    assert receipt["entrypoint"] == ENTRYPOINTS[method]


@pytest.mark.parametrize("method, role, tuning", [
    ("dpo", "hub", "qlora"), ("dpo", "specialist", "lora"), ("ppo", "specialist", "qlora"), ("ppo", "hub", "full"),
])
def test_unsupported_method_settings_are_refused(store, method, role, tuning):
    hub, data = model(store), training(store)

    with pytest.raises(LearningSelectionError, match="^invalid_learning_request$"):
        selection_preflight(OWNER, choose(hub, data, method, role, tuning), store)


def test_an_outbound_connection_is_inference_only(store):
    remote, data = model(store, backend="http"), training(store)

    receipt = selection_preflight(OWNER, choose(remote, data), store, weights=has_weights)

    assert receipt["model"]["source_kind"] == "outbound"
    assert receipt["status"] == "blocked" and receipt["blockers"] == ["outbound_connection_is_inference_only"]


def test_a_new_dataset_version_supersedes_the_one_it_replaces(store):
    hub, first = model(store), training(store)
    before = selection_preflight(OWNER, choose(hub, first), store)
    second = training(store, count=13, replaces=first["id"])

    with pytest.raises(LearningSelectionError, match="^learning_dataset_superseded$"):
        selection_preflight(OWNER, choose(hub, first), store)
    listed = {row["id"]: row for row in learning_catalog(OWNER, store)["datasets"]}
    assert listed[first["id"]]["superseded_by"] == second["id"]
    after = selection_preflight(OWNER, choose(hub, second), store)
    assert (after["dataset"]["version"], after["dataset"]["lineage_id"]) == (2, first["id"])
    assert after["preflight_sha256"] != before["preflight_sha256"]


def test_a_dataset_that_changed_since_it_was_listed_is_refused(store):
    hub, data = model(store), training(store)

    with pytest.raises(LearningSelectionError, match="^learning_dataset_changed$"):
        selection_preflight(OWNER, choose(hub, data, digest="0" * 64), store)


def test_another_owners_resources_look_exactly_like_missing_ones(store):
    hub, data = model(store), training(store)
    foreign_model, foreign_data = model(store, owner=OTHER), training(store, owner=OTHER)

    for chosen_model, chosen_data in ((foreign_model, data), (hub, foreign_data), ({"id": "f" * 32}, data)):
        with pytest.raises(LearningSelectionError, match="^learning_resource_unavailable$"):
            selection_preflight(OWNER, choose(chosen_model, chosen_data), store)


def test_reference_data_and_unready_models_cannot_be_chosen(store):
    hub, data = model(store), training(store)

    with pytest.raises(LearningSelectionError, match="^learning_dataset_not_training$"):
        selection_preflight(OWNER, choose(hub, reference(store)), store)
    with pytest.raises(LearningSelectionError, match="^learning_resource_unavailable$"):
        selection_preflight(OWNER, choose(model(store, state="pending"), data), store)


@pytest.mark.parametrize("content", [
    {"train": [], "validation": []},
    {"train": [{"messages": [{"role": "user", "content": "only a question"}]}] * 10, "validation": []},
    {"train": [], "evaluation": []},
])
def test_a_damaged_training_dataset_makes_the_catalog_unavailable_not_partial(store, content):
    model(store)
    store.put(OWNER, "dataset", {"name": "Damaged", "purpose": "training", "state": "ready", "sha256": "1" * 64,
                                 "source": "Owned", "license": "Owned", "content": content})

    with pytest.raises(LearningSelectionError, match="^learning_training_membership_invalid$"):
        learning_catalog(OWNER, store)


def test_an_unknown_model_backend_is_never_offered(store):
    store.put(OWNER, "model", {"name": "Unknown", "backend": "shell", "model": "x", "state": "ready"})

    with pytest.raises(LearningSelectionError, match="^learning_resource_unavailable$"):
        learning_catalog(OWNER, store)


@pytest.mark.parametrize("owner", ["", "   ", "x" * 201])
def test_an_invalid_owner_is_refused(store, owner):
    with pytest.raises(LearningSelectionError, match="^invalid_learning_request$"):
        learning_catalog(owner, store)


def test_weights_are_only_sealed_uploads_and_enabled_full_local_checkpoints(tmp_path, monkeypatch):
    """Training needs a complete base checkpoint the host can read; nothing else counts as weights."""
    from types import SimpleNamespace

    import config.settings
    from config.settings import ModelProfileSettings
    from services.learning_control.application import model_weights

    sealed, unsealed, checkpoint, empty = (tmp_path / name for name in ("sealed", "unsealed", "checkpoint", "empty"))
    for path in (sealed, unsealed, checkpoint, empty):
        path.mkdir()
    (sealed / ".sealed").write_text("")
    (checkpoint / "config.json").write_text("{}")
    profiles = {
        "full": ModelProfileSettings(backend="transformers", model_path=checkpoint),
        "disabled": ModelProfileSettings(backend="transformers", model_path=checkpoint, enabled=False),
        "adapter": ModelProfileSettings(backend="transformers", model_path=checkpoint, adapter_path=tmp_path),
        "no-config": ModelProfileSettings(backend="transformers", model_path=empty),
        "no-path": ModelProfileSettings(backend="transformers"),
    }
    monkeypatch.setattr(config.settings, "get_settings", lambda: SimpleNamespace(model_profiles=profiles))

    assert model_weights({"backend": "uploaded", "model_path": str(sealed)}) == sealed
    assert model_weights({"backend": "local", "model": "full"}) == checkpoint
    for row in ({"backend": "uploaded", "model_path": str(unsealed)}, {"backend": "uploaded", "model_path": ""},
                {"backend": "uploaded"}, {"backend": "local", "model": "disabled"},
                {"backend": "local", "model": "adapter"}, {"backend": "local", "model": "no-config"},
                {"backend": "local", "model": "no-path"}, {"backend": "local", "model": "unknown"},
                {"backend": "http", "model": "full", "model_path": str(sealed)}):
        assert model_weights(row) is None, row
