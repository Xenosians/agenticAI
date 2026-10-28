"""Learning selection: the owner's catalog and a readiness check, never a run.

Both read the owner's real extension registry. The check pins the chosen model
and dataset version, lists every gate that still blocks training, and never
starts, loads or promotes anything.
"""
import json

import pytest

from services.extensions.store import ExtensionStore
from services.learning_control.application import (
    LearningSelectionError,
    SelectionRequest,
    learning_catalog,
    selection_preflight,
)

OWNER, OTHER = "owner-a", "owner-b"
COMMON_BLOCKERS = {"trusted_model_identity_required", "trusted_training_configuration_required",
                   "fixed_evaluation_required", "resource_unknown", "host_dispatch_not_integrated",
                   "separate_promotion_required"}
ENTRYPOINTS = {"sft": "scripts/train_recipe.py", "dpo": "scripts/train_preferences.py",
               "ppo": "scripts/train_ppo.py"}


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

    catalog = learning_catalog(OWNER, store)

    assert catalog["schema"] == "learning-catalog.v1"
    assert [row["id"] for row in catalog["methods"]] == ["sft", "dpo", "ppo"]
    assert catalog["models"] == [
        {"id": local["id"], "name": "Hub model", "backend": "local", "state": "ready",
         "source_kind": "local", "training_available": False},
        {"id": outbound["id"], "name": "Outbound model", "backend": "http", "state": "ready",
         "source_kind": "outbound", "training_available": False}]
    assert catalog["datasets"] == [
        {"id": data["id"], "name": "Ticket triage", "purpose": "training", "state": "ready",
         "sha256": data["sha256"], "lineage_id": data["id"], "version": 1,
         "train_count": 10, "validation_count": 2, "superseded_by": None}]
    assert catalog["execution_enabled"] is False and catalog["promotion_enabled"] is False
    # Dataset bodies never leave the registry.
    assert "question 0" not in json.dumps(catalog)


def test_an_uploaded_model_is_the_owners_local_weights(store):
    weights = uploaded(store)
    training(store)

    catalog = learning_catalog(OWNER, store)

    assert catalog["models"] == [{"id": weights["id"], "name": "My weights", "backend": "uploaded",
                                  "state": "ready", "source_kind": "local", "training_available": False}]
    assert "/state/" not in json.dumps(catalog)
    receipt = selection_preflight(OWNER, choose(weights, catalog["datasets"][0]), store)
    assert receipt["model"]["source_kind"] == "local"
    assert "outbound_connection_is_inference_only" not in receipt["blockers"]


def test_the_check_pins_the_selection_and_lists_every_gate_without_running(store):
    hub, data = model(store), training(store)

    receipt = selection_preflight(OWNER, choose(hub, data), store)

    assert receipt["schema"] == "learning-selection-preflight.v1"
    assert receipt["status"] == "blocked"
    assert receipt["blockers"] == sorted(COMMON_BLOCKERS)
    assert (receipt["method"], receipt["role"], receipt["tuning"]) == ("sft", "hub", "qlora")
    assert receipt["entrypoint"] == "scripts/train_recipe.py"
    assert receipt["model"] == {"id": hub["id"], "backend": "local", "source_kind": "local", "state": "ready"}
    assert receipt["dataset"] == {"id": data["id"], "sha256": data["sha256"], "lineage_id": data["id"],
                                  "version": 1, "train_count": 10, "validation_count": 2}
    assert receipt["resource_state"] == "unknown"
    for flag in ("execution_enabled", "model_loaded", "live_tool_execution", "promotion_enabled"):
        assert receipt[flag] is False
    # The same selection always has the same inspection identity.
    assert selection_preflight(OWNER, choose(hub, data), store)["preflight_sha256"] == receipt["preflight_sha256"]


@pytest.mark.parametrize("method, role, tuning, extra", [
    ("sft", "specialist", "lora", set()),
    ("dpo", "specialist", "qlora", {"reviewed_preference_partitions_required"}),
    ("ppo", "hub", "qlora", {"evaluated_adapter_checkpoint_required", "independent_simulator_episodes_required"}),
])
def test_each_method_reports_its_own_prerequisites(store, method, role, tuning, extra):
    hub, data = model(store), training(store)

    receipt = selection_preflight(OWNER, choose(hub, data, method, role, tuning), store)

    assert set(receipt["blockers"]) == COMMON_BLOCKERS | extra
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

    receipt = selection_preflight(OWNER, choose(remote, data), store)

    assert receipt["model"]["source_kind"] == "outbound"
    assert set(receipt["blockers"]) == COMMON_BLOCKERS | {"outbound_connection_is_inference_only"}


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
