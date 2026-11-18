"""The training job description every worker runs: one schema for every specialist and stage."""
import json

import pytest
from pydantic import ValidationError

from learning.worker.job import TrainingJob, load_job


def job(**overrides):
    payload = {
        "schema": "janus-training-job.v1",
        "job_id": "networking-20261119T000000Z",
        "model": {"source": "huggingface", "name": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "main"},
        "dataset": {"source": "local", "path": "data"},
        "stages": ["merge", "sft", "dpo"],
        "evaluation": {"cases": "data/heldout.jsonl"},
        "device": "auto",
    }
    payload.update(overrides)
    return payload


def test_a_job_names_a_model_a_dataset_and_its_stages_in_order():
    parsed = TrainingJob.model_validate(job())
    assert parsed.stages == ["merge", "sft", "dpo"]
    assert parsed.model.source == "huggingface"
    assert parsed.sft.tuning == "lora"          # defaults keep CPU workers able to train
    assert parsed.guard.patience == 2


@pytest.mark.parametrize("stages", [["dpo", "sft"], ["sft", "sft"], [], ["ppo"], ["rl"]])
def test_stages_must_run_in_the_fixed_order_once_each(stages):
    with pytest.raises(ValidationError):
        TrainingJob.model_validate(job(stages=stages))


def test_ppo_is_only_for_the_routing_hub():
    with pytest.raises(ValidationError, match="routing hub"):
        TrainingJob.model_validate(job(stages=["sft", "ppo"], role="specialist"))
    assert TrainingJob.model_validate(job(stages=["sft", "dpo", "ppo"], role="hub")).role == "hub"


@pytest.mark.parametrize("model", [
    {"source": "huggingface", "name": "../../etc/passwd"},
    {"source": "huggingface", "name": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "main; rm -rf /"},
    {"source": "ollama", "name": "qwen2.5:0.5b; curl x"},
    {"source": "local", "path": "/etc"},
    {"source": "ftp", "name": "x"},
])
def test_a_model_reference_is_a_plain_name_or_a_path_inside_the_job(model):
    with pytest.raises(ValidationError):
        TrainingJob.model_validate(job(model=model))


def test_an_ollama_model_is_a_name_and_tag_and_a_local_model_stays_inside_the_job():
    assert TrainingJob.model_validate(job(model={"source": "ollama", "name": "qwen2.5:0.5b"})).model.name == "qwen2.5:0.5b"
    assert TrainingJob.model_validate(job(model={"source": "local", "path": "base"})).model.path == "base"


def test_the_merge_stage_needs_adapters_of_the_same_base():
    with pytest.raises(ValidationError, match="adapters"):
        TrainingJob.model_validate(job(stages=["merge", "sft"], merge={"adapters": []}))
    parsed = TrainingJob.model_validate(job(merge={"adapters": ["adapters/a", "adapters/b"], "method": "ties",
                                                   "density": 0.5}))
    assert parsed.merge.method == "ties"


def test_unknown_fields_are_refused_so_a_typo_never_silently_trains_with_defaults():
    with pytest.raises(ValidationError):
        TrainingJob.model_validate(job(lerning_rate=1))


def test_load_job_reads_the_job_file_and_resolves_paths_inside_its_directory(tmp_path):
    (tmp_path / "job.json").write_text(json.dumps(job(model={"source": "local", "path": "base"})))
    parsed, root = load_job(tmp_path / "job.json")
    assert root == tmp_path.resolve()
    assert parsed.resolve(root, parsed.model.path) == (tmp_path / "base").resolve()
    with pytest.raises(ValueError, match="inside the job"):
        parsed.resolve(root, "../outside")
