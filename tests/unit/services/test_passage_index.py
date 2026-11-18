"""Reference passages are indexed once per owner, not rebuilt for every search.

Each owner's passages live in that owner's own index, so another owner's data
never affects which passages are found or how they rank.
"""
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from services.extensions.store import NEWER_VERSION_EXISTS, ExtensionStore

FREEZE = "Payroll cluster maintenance. The quarterly change freeze starts on the second Tuesday at 18:00 UTC."
VPN = "VPN troubleshooting. Error code 809 usually means a home router is blocking the tunnel."


def add(store, text, *, owner="alice", name="Handbook", **extra):
    return store.dataset(owner, name=name, source=f"{name} source", license="Owned", purpose="rag",
                         reviewed=True, text=text, **extra)


def contents(hits):
    return [hit["content"] for hit in hits]


def test_a_search_after_admission_never_reads_or_splits_dataset_text_again(tmp_path, monkeypatch):
    store = ExtensionStore(tmp_path)
    row = add(store, FREEZE)

    def refuse(*args, **kwargs):
        raise AssertionError("dataset text was split again")

    monkeypatch.setattr("services.extensions.store._chunks", refuse)
    reopened = ExtensionStore(tmp_path)
    assert contents(reopened.search_reference("alice", "payroll change freeze")) == [FREEZE]
    assert contents(reopened.search("alice", row["id"], "payroll change freeze")) == [FREEZE]


def test_data_saved_before_the_index_existed_is_indexed_on_first_search(tmp_path):
    store = ExtensionStore(tmp_path)
    add(store, FREEZE)
    with store.connect() as db:
        db.execute(f'DROP TABLE "{store.passage_table("alice")}"')
        db.execute("DELETE FROM passage_source")
    assert contents(ExtensionStore(tmp_path).search_reference("alice", "payroll change freeze")) == [FREEZE]


def test_a_store_created_before_the_index_gains_it(tmp_path):
    store = ExtensionStore(tmp_path)
    add(store, FREEZE)
    with store.connect() as db:
        db.execute(f'DROP TABLE "{store.passage_table("alice")}"')
        db.execute("DROP TABLE passage_source")
    assert contents(ExtensionStore(tmp_path).search_reference("alice", "payroll change freeze")) == [FREEZE]


def test_a_deleted_dataset_is_never_found_again(tmp_path):
    store = ExtensionStore(tmp_path)
    row = add(store, FREEZE)
    store.delete("alice", row["id"])
    assert store.search_reference("alice", "payroll change freeze") == []


def test_new_reference_and_index_are_committed_together(tmp_path, monkeypatch):
    store = ExtensionStore(tmp_path)
    index = store._index_dataset

    def fail_after_index(*args, **kwargs):
        index(*args, **kwargs)
        raise sqlite3.OperationalError("simulated index write failure")

    monkeypatch.setattr(store, "_index_dataset", fail_after_index)
    with pytest.raises(sqlite3.OperationalError, match="simulated index write failure"):
        add(store, FREEZE)

    with store.connect() as db:
        assert db.execute("SELECT 1 FROM resource WHERE owner='alice'").fetchone() is None
        assert db.execute("SELECT 1 FROM passage_source WHERE owner='alice'").fetchone() is None
        assert db.execute("SELECT 1 FROM sqlite_master WHERE name=?", (store.passage_table("alice"),)).fetchone() is None


def test_replacement_and_its_index_are_committed_together(tmp_path, monkeypatch):
    store = ExtensionStore(tmp_path)
    old = add(store, "Old guide. The gateway uses the legacy route.")
    index = store._index_dataset

    def fail_after_index(*args, **kwargs):
        index(*args, **kwargs)
        raise sqlite3.OperationalError("simulated replacement index failure")

    monkeypatch.setattr(store, "_index_dataset", fail_after_index)
    with pytest.raises(sqlite3.OperationalError, match="simulated replacement index failure"):
        add(store, "New guide. The gateway uses the current route.", replaces=old["id"])

    assert len(store.list("alice")) == 1
    assert contents(store.search_reference("alice", "gateway legacy route")) == [
        "Old guide. The gateway uses the legacy route."]
    assert contents(store.search_reference("alice", "gateway current route")) == [
        "Old guide. The gateway uses the legacy route."]


def test_concurrent_replacements_publish_only_one_version_and_index(tmp_path):
    store = ExtensionStore(tmp_path)
    old = add(store, "Runbook. The service uses the old gateway.")
    start_together = Barrier(2)

    def replace(text):
        start_together.wait(timeout=5)
        try:
            add(store, text, replaces=old["id"])
            return "saved"
        except ValueError as error:
            if str(error) == NEWER_VERSION_EXISTS:
                return "conflict"
            raise

    with ThreadPoolExecutor(max_workers=2) as workers:
        outcomes = list(workers.map(replace, [
            "Runbook. The service uses the east gateway.",
            "Runbook. The service uses the west gateway.",
        ]))

    assert sorted(outcomes) == ["conflict", "saved"]
    histories = store.list("alice")
    assert len(histories) == 2
    latest = next(item for item in histories if "superseded_by" not in item)
    matches = contents(store.search_reference("alice", "service gateway"))
    assert matches == [latest["content"]["text"]]


def test_concurrent_cold_searches_build_one_correct_owner_index(tmp_path):
    store = ExtensionStore(tmp_path)
    add(store, FREEZE)
    table = store.passage_table("alice")
    with store.connect() as db:
        db.execute(f'DROP TABLE "{table}"')
        db.execute("DELETE FROM passage_source WHERE owner='alice'")

    def search(_):
        hits = store.search_reference("alice", "payroll change freeze")
        return [(hit["dataset_id"], hit["chunk"], hit["sha256"], hit["content"]) for hit in hits]

    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(workers.map(search, range(16)))

    assert results and all(result == results[0] for result in results)
    assert results[0][0][3] == FREEZE
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM passage_source WHERE owner='alice'").fetchone()[0] == 1


def test_a_replaced_version_is_not_searched_for_requests_but_its_source_still_is(tmp_path):
    store = ExtensionStore(tmp_path)
    first = add(store, "Payroll change freeze. The freeze starts on the first Monday.")
    add(store, FREEZE, replaces=first["id"])
    assert contents(store.search_reference("alice", "payroll change freeze")) == [FREEZE]
    assert contents(store.search("alice", first["id"], "payroll change freeze")) == [
        "Payroll change freeze. The freeze starts on the first Monday."]


def test_each_owner_has_a_separate_index(tmp_path):
    store = ExtensionStore(tmp_path)
    add(store, FREEZE, owner="alice")
    add(store, VPN, owner="bob")
    alice, bob = store.passage_table("alice"), store.passage_table("bob")
    assert alice != bob and "alice" not in alice and "bob" not in bob
    with store.connect() as db:
        assert [row[0] for row in db.execute(f'SELECT content FROM "{alice}"')] == [FREEZE]
        assert [row[0] for row in db.execute(f'SELECT content FROM "{bob}"')] == [VPN]
    assert store.search_reference("bob", "payroll change freeze") == []


def test_another_owners_data_never_changes_how_passages_rank(tmp_path):
    store = ExtensionStore(tmp_path)
    # Equal evidence for both words: the order is decided only by this owner's statistics.
    add(store, "Alpha alpha alpha beta.\n\nAlpha beta beta beta.")
    before = contents(store.search_reference("alice", "alpha beta"))
    for index in range(30):
        add(store, f"Alpha note {index}. Alpha alpha.", owner="bob", name=f"Bob {index}")
    assert contents(store.search_reference("alice", "alpha beta")) == before


def test_a_long_question_still_finds_the_passage_with_most_of_its_words(tmp_path):
    store = ExtensionStore(tmp_path)
    words = [f"word{letter}" for letter in "abcdefghijklmnopqrst"]
    answer = "Answer. " + " ".join(words[11:]) + "."
    add(store, "\n\n".join(["Other. " + " ".join(words[:4]) + ".", answer]))
    assert contents(store.search_reference("alice", " ".join(words))) == [answer]


def test_every_hit_reopens_the_passage_it_cites(tmp_path):
    store = ExtensionStore(tmp_path)
    first = add(store, "Printer queue. Restart the spooler.\n\nLaptop loans. Ask the desk.", name="Printers")
    add(store, "Payroll access. Request it from finance.\n\n" + FREEZE, name="Payroll")
    store.delete("alice", first["id"])
    add(store, "Desk hours. Monday to Friday.\n\n" + VPN, name="Remote")
    for question in ("payroll change freeze", "VPN error 809", "payroll access finance"):
        for hit in store.search_reference("alice", question):
            reopened = store.resolve("alice", hit["dataset_id"], hit["chunk"], hit["sha256"])
            assert reopened["content"] == hit["content"]


def test_an_old_version_between_current_ones_is_never_searched(tmp_path):
    store = ExtensionStore(tmp_path)
    add(store, "Backup schedule. Backups run nightly to vault-b.", name="Backups")
    old = add(store, "Retired runbook. The legacy fax gateway reboots on Sundays.", name="Fax")
    add(store, "Wireless guide. Guests use CORP-GUEST.", name="Wireless")
    add(store, "Current runbook. The fax service moved to email.", name="Fax", replaces=old["id"])
    assert store.search_reference("alice", "legacy fax gateway reboots") == []
    assert contents(store.search_reference("alice", "backups nightly vault-b")) == [
        "Backup schedule. Backups run nightly to vault-b."]
    assert contents(store.search_reference("alice", "guests CORP-GUEST")) == ["Wireless guide. Guests use CORP-GUEST."]


def test_training_datasets_are_never_indexed(tmp_path):
    store = ExtensionStore(tmp_path)
    rows = [{"messages": [{"role": "user", "content": f"payroll change freeze {i}?"},
                          {"role": "assistant", "content": "Second Tuesday."}]} for i in range(12)]
    store.dataset("alice", name="Train", source="Owned", license="Owned", purpose="training", reviewed=True,
                  text="\n".join(json.dumps(row) for row in rows))
    assert store.search_reference("alice", "payroll change freeze") == []
    with store.connect() as db:
        tables = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE name LIKE 'passages_%'")]
    assert all(store.passage_table("alice") not in name for name in tables)


def test_an_index_table_name_is_never_built_from_untrusted_text(tmp_path):
    store = ExtensionStore(tmp_path)
    name = store.passage_table('alice"; DROP TABLE resource; --')
    assert name.startswith("passages_") and name[len("passages_"):].isalnum()
    with pytest.raises(sqlite3.OperationalError):
        with store.connect() as db:
            db.execute(f'SELECT 1 FROM "{name}"')
