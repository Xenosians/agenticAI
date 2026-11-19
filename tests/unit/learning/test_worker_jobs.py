"""The host turns a reviewed specialist into a worker job: corpus, held-out cases, admissions."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from learning.continual.automation.specialists import SpecialistTarget
from learning.worker.job import load_job
from learning.worker.loading import require_admissions
from learning.worker_jobs import build_specialist_job, specialist_heldout_rows

CASE = {"name": "ping one host", "user": "ping web-01", "allowed_tools": ["network_ping"],
        "allowed_arguments": {"host": ["web-01"]}, "expected_tool": "network_ping",
        "expected_arguments": {"host": "web-01"}}


def fixture_target(tmp_path: Path) -> SpecialistTarget:
    config = tmp_path / "networking.json"
    config.write_text(json.dumps({"specialist": "networking-specialist", "agent_definition": "agents/net.md",
                                  "base_model_key": "small", "candidate_model_key": "net-candidate",
                                  "examples": [], "eval_cases": [CASE], "max_new_tokens": 96}), encoding="utf-8")
    return SpecialistTarget(name="networking", config_path=config, specialist="networking-specialist",
                            agent_definition=Path("agents/net.md"), base_model_key="small",
                            candidate_model_key="net-candidate")


def fake_corpus(directory: Path, *, system_prompt="SYSTEM PROMPT") -> None:
    directory.mkdir(parents=True)
    record = {"record_id": "r1", "split": "train", "tool_name": "network_ping", "user_request": "ping db-01",
              "semantic_context": {"allowed_tools": ["network_ping"], "allowed_arguments": {"host": ["db-01"]},
                                   "forbidden_tools": [], "forbidden_arguments": {}},
              "target_response": '[{"arguments":{"host":"db-01"},"name":"network_ping"}]', "tags": []}
    (directory / "train.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (directory / "validation.jsonl").write_text(json.dumps(record | {"record_id": "r2", "split": "validation"}) + "\n",
                                               encoding="utf-8")


@pytest.fixture
def host(tmp_path, monkeypatch):
    import learning.worker_jobs as jobs

    base = tmp_path / "models" / "small"
    base.mkdir(parents=True)
    (base / "model.safetensors").write_bytes(b"weights")
    serving = tmp_path / "models" / "net-serving-adapter"
    serving.mkdir(parents=True)
    (serving / "adapter_config.json").write_text("{}")
    profiles = {"small": SimpleNamespace(model_path=base, adapter_path=None, worker_prompt_profile="compact",
                                         backend="hf-causal"),
                "net-candidate": SimpleNamespace(model_path=base, adapter_path=serving,
                                                 worker_prompt_profile="compact", backend="hf-causal")}
    admitted = []
    monkeypatch.setattr(jobs, "_build_corpus", lambda target, directory: fake_corpus(directory))
    monkeypatch.setattr(jobs, "_system_prompt", lambda target, profile: "SYSTEM PROMPT")
    monkeypatch.setattr(jobs, "_admit", lambda artifact, details: admitted.append((artifact, details)) or "a" * 64)
    return SimpleNamespace(profiles=profiles, admitted=admitted, base=base, serving=serving)


def test_held_out_cases_carry_the_exact_runtime_prompt(tmp_path):
    rows = specialist_heldout_rows(fixture_target(tmp_path), system_prompt="SYSTEM PROMPT")
    assert len(rows) == 1
    assert rows[0]["expected_tool"] == "network_ping" and rows[0]["max_new_tokens"] == 96
    messages = rows[0]["prompt_messages"]
    assert messages[0] == {"role": "system", "content": "SYSTEM PROMPT"}
    assert messages[1] == {"role": "user", "content": "ping web-01"}
    assert '"allowed_arguments": {"host": ["web-01"]}' in messages[2]["content"]


def test_a_specialist_job_is_a_self_contained_folder_the_worker_accepts(tmp_path, host):
    target = fixture_target(tmp_path)
    folder = build_specialist_job(target, profiles=host.profiles, directory=tmp_path / "job",
                                  stages=("sft", "dpo"), stamp="20261119T010203Z")
    job, root = load_job(folder / "job.json")
    assert job.job_id == "networking-20261119T010203Z"
    assert job.stages == ["sft", "dpo"] and job.role == "specialist"
    assert job.model.source == "local" and (root / job.model.path / "model.safetensors").is_file()
    assert (root / "data/train.jsonl").is_file() and (root / "data/validation.jsonl").is_file()
    first = json.loads((root / "data/train.jsonl").read_text().splitlines()[0])
    assert first["prompt_messages"][0]["content"] == "SYSTEM PROMPT"
    assert first["chosen"] == '[{"arguments":{"host":"db-01"},"name":"network_ping"}]'
    assert job.baseline_adapter == "serving-adapter"       # the serving specialist is what must be beaten
    require_admissions(job)                                 # every input has a receipt
    assert {artifact for artifact, _ in host.admitted} == set(job.input_artifacts())


def test_a_specialist_without_a_serving_adapter_is_measured_against_its_base(tmp_path, host):
    host.profiles["net-candidate"] = None
    folder = build_specialist_job(fixture_target(tmp_path), profiles=host.profiles, directory=tmp_path / "job",
                                  stages=("sft",), stamp="20261119T010203Z")
    job, _ = load_job(folder / "job.json")
    assert job.baseline_adapter is None


def test_a_lua_refusal_stops_the_job_before_any_file_is_handed_to_a_worker(tmp_path, host, monkeypatch):
    import learning.worker_jobs as jobs

    def refuse(artifact, details):
        raise PermissionError("operator adapter refuses")

    monkeypatch.setattr(jobs, "_admit", refuse)
    with pytest.raises(PermissionError):
        build_specialist_job(fixture_target(tmp_path), profiles=host.profiles, directory=tmp_path / "job",
                             stages=("sft",), stamp="20261119T010203Z")
    assert not (tmp_path / "job" / "job.json").exists()


@pytest.mark.parametrize("backend, options", [("qwen3", {"enable_thinking": False}), ("hf-causal", {}),
                                              ("qwen-funccall", {}), ("ministral", {})])
def test_the_job_renders_prompts_exactly_as_the_serving_backend_does(tmp_path, host, backend, options):
    """Qwen3 is served with enable_thinking=False; a candidate scored without it scored 13/13 in the worker
    and 11/13 in Janus's gate on the same adapter (live, 19 Nov)."""
    host.profiles["small"].backend = backend
    folder = build_specialist_job(fixture_target(tmp_path), profiles=host.profiles, directory=tmp_path / "job",
                                  stages=("sft",), stamp="20261119T010203Z")
    assert load_job(folder / "job.json")[0].evaluation.template_options == options


def test_a_serving_adapter_on_other_base_weights_is_not_the_workers_baseline(tmp_path, host):
    """networking: its config's base is hub-main (3B) while the serving adapter sits on a 0.5B model."""
    other = tmp_path / "models" / "other-base"
    other.mkdir(parents=True)
    host.profiles["net-candidate"].model_path = other
    folder = build_specialist_job(fixture_target(tmp_path), profiles=host.profiles, directory=tmp_path / "job",
                                  stages=("sft",), stamp="20261119T010203Z")
    job, root = load_job(folder / "job.json")
    assert job.baseline_adapter is None and not (root / "serving-adapter").exists()


def test_a_backend_with_unknown_prompt_rendering_is_refused(tmp_path, host):
    host.profiles["small"].backend = "something-new"
    with pytest.raises(ValueError, match="prompt"):
        build_specialist_job(fixture_target(tmp_path), profiles=host.profiles, directory=tmp_path / "job",
                             stages=("sft",), stamp="20261119T010203Z")


def test_the_cpu_proxmox_worker_gets_a_lora_job_and_a_gpu_worker_qlora(tmp_path, host):
    cpu = build_specialist_job(fixture_target(tmp_path), profiles=host.profiles, directory=tmp_path / "cpu",
                               stages=("sft",), stamp="20261119T010203Z", device="cpu")
    gpu = build_specialist_job(fixture_target(tmp_path), profiles=host.profiles, directory=tmp_path / "gpu",
                               stages=("sft",), stamp="20261119T010203Z", device="cuda")
    assert load_job(cpu / "job.json")[0].sft.tuning == "lora"
    assert load_job(gpu / "job.json")[0].sft.tuning == "qlora"


def write_provenance(base: Path, *, files: dict[str, bytes], repository="mistralai/Tiny", revision="a" * 40):
    import hashlib

    for name, content in files.items():
        (base / name).write_bytes(content)
    (base / "acquisition-provenance.json").write_text(json.dumps({
        "repository": repository, "revision": revision, "license": "Apache-2.0",
        "sha256": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}}))


def test_a_remote_gpu_job_names_a_verified_base_by_its_pinned_repository_instead_of_carrying_it(tmp_path, host):
    """A 7.2 GiB base in every Kaggle upload is not needed when the local files are the pinned public files."""
    write_provenance(host.base, files={"model.safetensors": b"weights", "config.json": b"{}"})
    folder = build_specialist_job(fixture_target(tmp_path), profiles=host.profiles, directory=tmp_path / "job",
                                  stamp="20261119T010203Z", device="cuda", base_by_reference=True)
    job, root = load_job(folder / "job.json")
    assert (job.model.source, job.model.name, job.model.revision) == ("huggingface", "mistralai/Tiny", "a" * 40)
    assert not (root / "base").exists()
    require_admissions(job)
    assert "huggingface:mistralai/Tiny@" + "a" * 40 in {artifact for artifact, _ in host.admitted}


@pytest.mark.parametrize("trouble", ["no provenance", "changed file", "not a commit"])
def test_a_base_that_cannot_be_proven_identical_travels_with_the_job(tmp_path, host, trouble):
    if trouble != "no provenance":
        write_provenance(host.base, files={"model.safetensors": b"weights"},
                         revision="main" if trouble == "not a commit" else "a" * 40)
    if trouble == "changed file":
        (host.base / "model.safetensors").write_bytes(b"other weights")
    folder = build_specialist_job(fixture_target(tmp_path), profiles=host.profiles, directory=tmp_path / "job",
                                  stamp="20261119T010203Z", device="cuda", base_by_reference=True)
    job, root = load_job(folder / "job.json")
    assert job.model.source == "local" and (root / "base" / "model.safetensors").is_file()
