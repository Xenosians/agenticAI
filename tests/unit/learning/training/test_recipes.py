import hashlib
import json
import pytest
from learning.training.recipes import AdaptationRecipe, admit_records, encode_record, train_recipe


def recipe(tmp_path, rows, **kwargs):
    path = tmp_path / "data.jsonl"
    raw = "".join(json.dumps(row) + "\n" for row in rows).encode()
    path.write_bytes(raw)
    return AdaptationRecipe(objective="sft", model_path=tmp_path / "base", dataset=path,
        dataset_sha256=hashlib.sha256(raw).hexdigest(), allowed_sources=["authored"],
        allowed_licenses=["CC0"], output_root=tmp_path / "output", **kwargs)


def records():
    return [dict(id=str(i), source="authored", license="CC0", reviewed=True, split=split,
        messages=[{"role":"user", "content":f"Read interface port{i}."}, {"role":"assistant","content":f"port{i}"}])
        for i, split in enumerate(["train", "validation"])]


def test_admission_and_hash(tmp_path):
    config = recipe(tmp_path, records())
    assert [len(rows) for rows in admit_records(config)] == [1, 1]
    config.dataset.write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        admit_records(config)


@pytest.mark.parametrize("defect", ["duplicate", "unreviewed", "malformed", "secret"])
def test_rejects_bad_training_evidence(tmp_path, defect):
    rows = records()
    if defect == "duplicate": rows[1]["messages"] = rows[0]["messages"]
    if defect == "unreviewed": rows[0]["reviewed"] = False
    if defect == "malformed": rows[0]["messages"][-1] = "bad"
    if defect == "secret": rows[0]["messages"][0]["content"] = "password=veryprivate"
    with pytest.raises(ValueError): admit_records(recipe(tmp_path, rows))


def test_no_implicit_training(tmp_path):
    with pytest.raises(PermissionError): train_recipe(recipe(tmp_path, records()))


def test_assistant_only_loss_and_no_truncation(tmp_path):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [1, 2] if kwargs["add_generation_prompt"] else [1, 2, 3, 4]
    config = recipe(tmp_path, records())
    assert encode_record(Tokenizer(), records()[0], config)["labels"] == [-100, -100, 3, 4]
    config.max_length = 3
    with pytest.raises(ValueError, match="truncate"):
        encode_record(Tokenizer(), records()[0], config)


def test_causal_adaptation_labels_all_tokens_including_eos(tmp_path):
    class Tokenizer:
        eos_token_id = 9
        def encode(self, text, **kwargs):
            return [1, 2, 3]
    config = recipe(tmp_path, records())
    config.objective = "causal_adaptation"
    encoded = encode_record(Tokenizer(), {"text":"Reviewed network documentation."}, config)
    assert encoded["input_ids"] == encoded["labels"] == [1, 2, 3, 9]
