"""Dataset version identity (SRS 1.9 FR-085).

A changed dataset is uploaded as a new version of the one it replaces. Each
version keeps its own identity and digest, earlier versions stay intact so
their citations still resolve, history stays linear and owner-scoped, and a
training example never moves between training and evaluation across versions.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from services.extensions.store import ExtensionStore

REFERENCE = dict(name="Manual", source="Owned manual", license="Owned", purpose="rag", reviewed=True)
TRAINING = dict(name="Examples", source="Owned", license="Owned", purpose="training", reviewed=True)


def example(number, answer="answer"):
    return {"messages": [{"role": "user", "content": f"question {number}"},
                         {"role": "assistant", "content": f"{answer} {number}"}]}


def jsonl(rows):
    return "\n".join(json.dumps(row) for row in rows)


def inputs(dataset, split):
    return {json.dumps(row["messages"][:-1]) for row in dataset["content"][split]}


def test_a_new_version_links_to_the_dataset_it_replaces(tmp_path):
    store = ExtensionStore(tmp_path)
    first = store.dataset("alice", text="Router maintenance needs a window.", **REFERENCE)
    assert first["version"] == 1 and first["lineage_id"] == first["id"]
    assert first.get("previous_id") is None
    second = store.dataset("alice", text="Router maintenance needs an approved window.",
                           replaces=first["id"], **REFERENCE)
    assert second["version"] == 2 and second["lineage_id"] == first["id"]
    assert second["previous_id"] == first["id"] and second["previous_sha256"] == first["sha256"]
    assert second["sha256"] != first["sha256"]
    # The earlier version is kept as it was, so a citation made against it still resolves.
    assert store.resolve("alice", first["id"], 1, first["sha256"])["sha256"] == first["sha256"]
    listed = {row["id"]: row for row in store.list("alice")}
    assert listed[first["id"]]["superseded_by"] == second["id"]
    assert "superseded_by" not in listed[second["id"]]


def test_version_history_stays_linear_owner_scoped_and_purpose_bound(tmp_path):
    store = ExtensionStore(tmp_path)
    first = store.dataset("alice", text="Version one.", **REFERENCE)
    second = store.dataset("alice", text="Version two.", replaces=first["id"], **REFERENCE)
    with pytest.raises(ValueError, match="newer version"):
        store.dataset("alice", text="A branch from version one.", replaces=first["id"], **REFERENCE)
    with pytest.raises(ValueError, match="identical"):
        store.dataset("alice", text="Version two.", replaces=second["id"], **REFERENCE)
    with pytest.raises(ValueError, match="not found"):
        store.dataset("bob", text="Bob's copy.", replaces=second["id"], **REFERENCE)
    with pytest.raises(ValueError, match="purpose"):
        store.dataset("alice", text=jsonl(example(n) for n in range(12)), replaces=second["id"], **TRAINING)
    assert [row["version"] for row in store.list("alice")] == [1, 2]


def test_deleting_a_middle_version_keeps_the_history_linear(tmp_path):
    store = ExtensionStore(tmp_path)
    first = store.dataset("alice", text="Version one.", **REFERENCE)
    second = store.dataset("alice", text="Version two.", replaces=first["id"], **REFERENCE)
    third = store.dataset("alice", text="Version three.", replaces=second["id"], **REFERENCE)
    store.delete("alice", second["id"])
    listed = {row["id"]: row for row in store.list("alice")}
    assert listed[first["id"]]["superseded_by"] == third["id"]
    with pytest.raises(ValueError, match="newer version"):
        store.dataset("alice", text="A branch from version one.", replaces=first["id"], **REFERENCE)
    fourth = store.dataset("alice", text="Version four.", replaces=third["id"], **REFERENCE)
    assert fourth["version"] == 4 and fourth["lineage_id"] == first["id"]


def test_a_new_training_version_never_moves_an_example_between_training_and_evaluation(tmp_path):
    store = ExtensionStore(tmp_path)
    first = store.dataset("alice", text=jsonl(example(n) for n in range(12)), **TRAINING)
    # Version two corrects one answer and adds examples.
    rows = [example(n, "corrected" if n == 0 else "answer") for n in range(40)]
    second = store.dataset("alice", text=jsonl(rows), replaces=first["id"], **TRAINING)
    assert inputs(first, "validation") <= inputs(second, "validation")
    assert inputs(first, "train") <= inputs(second, "train")
    assert inputs(second, "validation").isdisjoint(inputs(second, "train"))
    assert len(inputs(second, "validation")) == 40 // 5
    # An example removed in one version and restored later keeps its first split.
    removed = sorted(inputs(first, "validation"))[0]
    kept = [row for row in rows if json.dumps(row["messages"][:-1]) != removed]
    third = store.dataset("alice", text=jsonl(kept + [example(n) for n in range(40, 45)]),
                          replaces=second["id"], **TRAINING)
    fourth = store.dataset("alice", text=jsonl(rows + [example(n) for n in range(40, 46)]),
                           replaces=third["id"], **TRAINING)
    assert removed in inputs(fourth, "validation")


def test_a_version_without_evaluation_examples_is_refused(tmp_path):
    store = ExtensionStore(tmp_path)
    first = store.dataset("alice", text=jsonl(example(n) for n in range(12)), **TRAINING)
    training_only = [row for row in (example(n) for n in range(12))
                     if json.dumps(row["messages"][:-1]) in inputs(first, "train")]
    with pytest.raises(ValueError, match="evaluation"):
        store.dataset("alice", text=jsonl(training_only), replaces=first["id"], **TRAINING)


def test_the_export_and_the_catalog_carry_the_version_identity(tmp_path, monkeypatch):
    from services.extensions import service

    store = ExtensionStore(tmp_path)
    monkeypatch.setattr(service, "extension_store", lambda: store)
    settings = SimpleNamespace(model_profiles={})
    first = asyncio.run(service.command("alice", "dataset", {
        "text": jsonl(example(n) for n in range(12)), **TRAINING}, settings))
    second = asyncio.run(service.command("alice", "dataset", {
        "text": jsonl(example(n) for n in range(15)), "replaces": first["id"], **TRAINING}, settings))
    assert second["version"] == 2 and "content" not in second
    exported = asyncio.run(service.command("alice", "export_dataset", {"id": second["id"]}, settings))
    assert exported["version"] == 2 and exported["lineage_id"] == first["id"]
    assert exported["previous_sha256"] == first["sha256"]
    with pytest.raises(ValueError, match="replaces"):
        asyncio.run(service.command("alice", "dataset", {
            "text": "Text.", "replaces": ["not", "an", "id"], **REFERENCE}, settings))
