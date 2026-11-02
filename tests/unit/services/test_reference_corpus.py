"""Requests draw on all of the owner's reference data unless the owner picks one source.

Passages reach answer-writing steps as compact quoted data: the facts and their
source, never hashes, and never as instructions or authority.
"""
import json

import pytest

from services.extensions.context import (
    answer_references, owner_execution, private_request, retrieval_evidence, with_retrieval,
)
from services.extensions.store import ExtensionStore
from subagents.llm.runtime.stages import InferenceStage

FREEZE = "Payroll cluster maintenance. The quarterly change freeze starts on the second Tuesday at 18:00 UTC."
VPN = "VPN troubleshooting. Error code 809 usually means a home router is blocking the tunnel."
QUESTION = "When does the payroll change freeze start?"


@pytest.fixture
def store(tmp_path, monkeypatch):
    store = ExtensionStore(tmp_path)
    monkeypatch.setattr("services.extensions.service.extension_store", lambda: store)
    return store


def add(store, text, *, name="Handbook", owner="alice", purpose="rag", **extra):
    return store.dataset(owner, name=name, source=f"{name} source", license="Owned", purpose=purpose,
                         reviewed=True, text=text, **extra)


def passages(owner="alice", message=QUESTION):
    with owner_execution(owner, message) as (_profile, used):
        return [hit["content"] for hit in retrieval_evidence.get()], used


def test_without_a_chosen_source_every_reference_dataset_is_searched(store):
    add(store, VPN, name="VPN guide")
    add(store, FREEZE, name="Payroll runbook")
    assert passages() == ([FREEZE], True)
    assert passages(message="What does VPN error 809 mean?") == ([VPN], True)


def test_a_chosen_source_limits_the_search_to_it(store):
    vpn = add(store, VPN, name="VPN guide")
    add(store, FREEZE, name="Payroll runbook")
    store.select("alice", None, vpn["id"])
    assert passages() == ([], False)


def test_only_the_latest_version_of_a_dataset_is_searched(store):
    first = add(store, "Payroll change freeze. The freeze starts on the first Monday.")
    add(store, FREEZE, replaces=first["id"])
    assert passages() == ([FREEZE], True)


def test_training_data_and_other_owners_reference_data_are_never_searched(store):
    add(store, FREEZE, owner="bob")
    rows = [{"messages": [{"role": "user", "content": f"When does the payroll change freeze start {i}?"},
                          {"role": "assistant", "content": "On the second Tuesday."}]} for i in range(12)]
    add(store, "\n".join(json.dumps(row) for row in rows), purpose="training")
    assert passages() == ([], False)


def test_a_request_that_found_no_reference_passage_is_not_private(store):
    add(store, VPN)
    with owner_execution("alice", "Restart the print spooler") as (profile, used):
        assert retrieval_evidence.get() == [] and not private_request(profile, used)
    with owner_execution("alice", "What does VPN error 809 mean?") as (profile, used):
        assert private_request(profile, used)


def test_a_passage_a_tool_added_later_makes_the_request_private(store):
    from services.extensions.context import disclose_references
    add(store, VPN)
    with owner_execution("alice", "Restart the print spooler") as (profile, used):
        assert not private_request(profile, used)
        disclose_references(store.search_reference("alice", "VPN error 809"))
        assert private_request(profile, used)


def test_answer_steps_get_compact_quoted_passages_and_citations_stay_complete(store):
    row = add(store, FREEZE, name="Payroll runbook")
    with owner_execution("alice", QUESTION):
        messages = with_retrieval([{"role": "system", "content": "policy"}, {"role": "user", "content": QUESTION}],
                                  stage=InferenceStage.PRIMARY_RESPONSE)
        references = answer_references()
    header, quoted = messages[-2]["content"].split("\n", 1)
    assert header.startswith("UNTRUSTED REFERENCE DATA")
    assert "never instructions" in header and "permission" in header
    assert json.loads(quoted) == [{"source": "Payroll runbook source", "text": FREEZE}]
    assert row["sha256"] not in messages[-2]["content"]
    assert [item["locator"] for item in references] == [f"dataset:{row['id']}#chunk:1@{row['sha256']}"]


def test_non_ascii_text_is_quoted_as_written(store):
    add(store, "Pembekuan perubahan payroll dimulai Selasa kedua. Café ☕ freeze note.")
    with owner_execution("alice", "payroll freeze café"):
        content = with_retrieval([{"role": "user", "content": "payroll freeze café"}],
                                 stage=InferenceStage.PRIMARY_RESPONSE)[0]["content"]
    assert "Café ☕" in content


def test_the_step_that_chooses_a_tool_never_sees_reference_passages(store):
    # Given the answer, a specialist writes it as prose instead of its tool
    # call, and untrusted text must never steer which tool runs or its arguments.
    add(store, FREEZE)
    messages = [{"role": "system", "content": "policy"}, {"role": "user", "content": QUESTION}]
    with owner_execution("alice", QUESTION):
        assert with_retrieval(messages, stage=InferenceStage.SPECIALIST_REASONING) is messages
        assert answer_references() == []


def test_the_answer_step_is_told_not_to_fill_gaps_the_passages_leave(store):
    # A passage can share the question's words without stating the fact asked
    # for; the answer must then say the documents do not cover it.
    add(store, FREEZE)
    with owner_execution("alice", "How long does the payroll change freeze last?"):
        header = with_retrieval([{"role": "user", "content": "How long does the payroll change freeze last?"}],
                                stage=InferenceStage.PRIMARY_RESPONSE)[0]["content"].split("\n", 1)[0]
    assert "do not state" in header and "say so" in header
