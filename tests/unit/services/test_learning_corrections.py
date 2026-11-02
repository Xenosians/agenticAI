"""Accepted corrections as training data (FR-085/089).

A correction the owner wrote on one of their tasks, once a reviewer accepted it,
pairs the owner's request with the result it should have had. The owner can save
their accepted corrections as the next version of one training dataset. Saving
goes through the same training admission as any upload (privacy scan, fixed
training/validation splits, version identity) and never starts a run.
"""
import json

import pytest

from services.extensions.store import TRAINING_MINIMUM_INPUTS, ExtensionStore
from services.learning_control.application import LearningSelectionError
from services.learning_control.corrections import AcceptedCorrection, accepted_corrections

OWNER, OTHER = "owner-a", "owner-b"


@pytest.fixture
def store(tmp_path):
    return ExtensionStore(tmp_path)


def correction(index, *, request=None, text=None, reviewed_at=None):
    return AcceptedCorrection(
        feedback_id=f"{index:032x}",
        request=request or f"Which queue handles VPN ticket {index}?",
        correction=text or f"VPN ticket {index} belongs to the Network Operations queue.",
        reviewed_at=reviewed_at or f"2026-11-01T{10 + index // 60:02d}:{index % 60:02d}:00.000000Z",
    )


def enough(start=0):
    return [correction(index) for index in range(start, start + TRAINING_MINIMUM_INPUTS + 2)]


def training_rows(row):
    return [record["messages"] for split in row["content"].values() for record in split]


def test_accepted_corrections_become_a_training_dataset_version_with_their_provenance(store):
    corrections = enough()
    saved = accepted_corrections(OWNER, corrections, store, apply=True)
    assert saved["status"] == "saved" and saved["usable"] == len(corrections) and saved["withheld"] == 0
    dataset = saved["dataset"]
    assert dataset["version"] == 1
    assert dataset["train_count"] + dataset["validation_count"] == len(corrections)
    row = store.get(OWNER, dataset["id"])
    assert row["purpose"] == "training" and row["sha256"] == dataset["sha256"]
    assert row["corrections"]["feedback_ids"] == sorted(item.feedback_id for item in corrections)
    first = corrections[0]
    assert [{"role": "user", "content": first.request},
            {"role": "assistant", "content": first.correction}] in training_rows(row)


def test_a_preview_saves_nothing(store):
    preview = accepted_corrections(OWNER, enough(), store, apply=False)
    assert preview["status"] == "ready" and preview["dataset"] is None
    assert preview["minimum"] == TRAINING_MINIMUM_INPUTS
    assert store.list(OWNER) == []


def test_too_few_requests_cannot_be_saved(store):
    corrections = [correction(index) for index in range(TRAINING_MINIMUM_INPUTS - 1)]
    assert accepted_corrections(OWNER, corrections, store, apply=False)["status"] == "too_few"
    with pytest.raises(LearningSelectionError, match="learning_corrections_too_few"):
        accepted_corrections(OWNER, corrections, store, apply=True)
    assert store.list(OWNER) == []


def test_the_latest_reviewed_correction_of_a_request_is_its_target(store):
    corrections = enough()
    request = corrections[0].request
    corrections.append(correction(90, request=request, text="It belongs to the Service Desk queue.",
                                  reviewed_at="2026-11-02T09:00:00.000000Z"))
    saved = accepted_corrections(OWNER, corrections, store, apply=True)
    assert saved["usable"] == len(corrections) - 1
    targets = [messages[-1]["content"] for messages in training_rows(store.get(OWNER, saved["dataset"]["id"]))
               if messages[0]["content"] == request]
    assert targets == ["It belongs to the Service Desk queue."]


def test_a_correction_with_a_personal_identifier_is_withheld_and_counted(store):
    corrections = [*enough(), correction(91, text="Ask jane.doe@example.test to reset it.")]
    saved = accepted_corrections(OWNER, corrections, store, apply=True)
    assert saved["withheld"] == 1 and saved["usable"] == len(corrections) - 1
    assert "jane.doe" not in json.dumps(store.get(OWNER, saved["dataset"]["id"])["content"])


def test_saving_again_adds_a_version_that_keeps_validation_out_of_training(store):
    first = accepted_corrections(OWNER, enough(), store, apply=True)["dataset"]
    validation = {json.dumps(record["messages"][:-1])
                  for record in store.get(OWNER, first["id"])["content"]["validation"]}
    unchanged = accepted_corrections(OWNER, enough(), store, apply=False)
    assert unchanged["status"] == "unchanged" and unchanged["dataset"]["id"] == first["id"]
    with pytest.raises(LearningSelectionError, match="learning_corrections_unchanged"):
        accepted_corrections(OWNER, enough(), store, apply=True)
    second = accepted_corrections(OWNER, [*enough(), correction(92), correction(93)], store, apply=True)["dataset"]
    assert second["version"] == 2 and second["lineage_id"] == first["lineage_id"]
    row = store.get(OWNER, second["id"])
    assert row["previous_id"] == first["id"]
    assert validation <= {json.dumps(record["messages"][:-1]) for record in row["content"]["validation"]}


def test_each_owner_has_their_own_corrections_dataset(store):
    mine = accepted_corrections(OWNER, enough(), store, apply=True)["dataset"]
    theirs = accepted_corrections(OTHER, enough(), store, apply=True)["dataset"]
    assert theirs["version"] == 1 and theirs["lineage_id"] != mine["lineage_id"]
    assert [row["id"] for row in store.list(OTHER)] == [theirs["id"]]


def test_an_uploaded_training_dataset_is_never_replaced_by_corrections(store):
    text = "\n".join(json.dumps({"messages": [{"role": "user", "content": f"q{index}"},
                                              {"role": "assistant", "content": f"a{index}"}]})
                     for index in range(TRAINING_MINIMUM_INPUTS))
    upload = store.dataset(OWNER, name="Ticket triage", text=text, source="Owned tickets", license="Owned",
                           purpose="training", reviewed=True)
    saved = accepted_corrections(OWNER, enough(), store, apply=True)["dataset"]
    assert saved["version"] == 1 and saved["lineage_id"] != upload["lineage_id"]
    assert store.get(OWNER, upload["id"])["sha256"] == upload["sha256"]
