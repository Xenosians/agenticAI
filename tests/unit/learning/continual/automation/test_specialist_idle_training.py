"""Idle training of the small local specialists.

Each idle cycle picks one small specialist whose reviewed training set changed
since its last candidate, trains a QLoRA candidate under the same breathing-room
limits as the Hub cycle, checks it on the specialist's held-out cases and keeps
it as an unregistered candidate. Nothing here changes what serves requests.
"""
import asyncio
import json
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from learning.continual.automation import specialists as specialists_module
from learning.continual.automation.config import ContinualAutomationSettings
from learning.continual.automation.limiter import TrainingYielded
from learning.continual.automation.specialists import (
    SpecialistCandidateStore,
    SpecialistIdleTrainer,
    SpecialistTarget,
    small_specialist_targets,
)

MODEL_PROFILES = {
    "small-base": SimpleNamespace(backend="hf-causal", model_path=Path("small"), adapter_path=None),
    "small-trained": SimpleNamespace(backend="hf-causal", model_path=Path("small"), adapter_path=Path("adapter")),
    "hub-main": SimpleNamespace(backend="ministral", model_path=Path("hub"), adapter_path=None),
}


def write_config(directory: Path, name: str, *, base: str, candidate: str, agent: str) -> Path:
    path = directory / f"{name}.json"
    path.write_text(json.dumps({
        "schema": "specialist-training-config.v1", "specialist": agent,
        "agent_definition": f"subagents/agents/{agent}.md", "base_model_key": base,
        "candidate_model_key": candidate, "worker_prompt_profile": "compact", "max_new_tokens": 96,
        "examples": [], "eval_cases": [],
    }), encoding="utf-8")
    return path


def weights(megabytes: int):
    return lambda _profile: megabytes * 2**20


def test_only_small_local_models_are_idle_training_targets(tmp_path):
    write_config(tmp_path, "account", base="small-base", candidate="small-trained", agent="account-specialist")
    write_config(tmp_path, "developer", base="hub-main", candidate="developer-func-trained", agent="developer-specialist")
    write_config(tmp_path, "missing", base="no-such-profile", candidate="x", agent="ticket-specialist")
    sizes = {"small-base": 900, "hub-main": 7300}
    targets = small_specialist_targets(
        config_directory=tmp_path, profiles=MODEL_PROFILES, max_weights_gib=2.0,
        weights_bytes=lambda profile: sizes[next(k for k, v in MODEL_PROFILES.items() if v is profile)] * 2**20)
    assert [(target.name, target.base_model_key) for target in targets] == [("account", "small-base")]


def test_external_training_files_beside_a_config_belong_to_it(tmp_path):
    write_config(tmp_path, "networking", base="small-base", candidate="small-trained", agent="networking-specialist")
    (tmp_path / "networking-safety-train.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "networking-unhinted-train.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "access-train.jsonl").write_text("{}\n", encoding="utf-8")
    [target] = small_specialist_targets(config_directory=tmp_path, profiles=MODEL_PROFILES, max_weights_gib=2.0,
                                        weights_bytes=weights(900))
    assert [path.name for path in target.external_sft_jsonl] == [
        "networking-safety-train.jsonl", "networking-unhinted-train.jsonl"]


def test_a_target_has_new_material_only_when_its_inputs_changed(tmp_path):
    config = write_config(tmp_path, "account", base="small-base", candidate="small-trained", agent="account-specialist")
    [target] = small_specialist_targets(config_directory=tmp_path, profiles=MODEL_PROFILES, max_weights_gib=2.0,
                                        weights_bytes=weights(900))
    store = SpecialistCandidateStore(tmp_path / "state")
    fingerprint = target.fingerprint(catalog="catalog-a")
    assert store.needs_training(target, fingerprint)
    store.record(target, fingerprint=fingerprint, outcome="candidate", candidate=None)
    assert not store.needs_training(target, target.fingerprint(catalog="catalog-a"))
    # A reviewed example added to the config, or a changed tool catalog, is new material.
    config.write_text(config.read_text(encoding="utf-8").replace('"examples": []', '"examples": [{"tool_name": "a", '
                                                                 '"user_request": "b", "arguments": {}}]'),
                      encoding="utf-8")
    assert store.needs_training(target, target.fingerprint(catalog="catalog-a"))


def test_a_changed_tool_catalog_alone_is_new_material(tmp_path):
    write_config(tmp_path, "account", base="small-base", candidate="small-trained", agent="account-specialist")
    [target] = small_specialist_targets(config_directory=tmp_path, profiles=MODEL_PROFILES, max_weights_gib=2.0,
                                        weights_bytes=weights(900))
    store = SpecialistCandidateStore(tmp_path / "state")
    store.record(target, fingerprint=target.fingerprint(catalog="catalog-a"), outcome="candidate", candidate=None)
    assert not store.needs_training(target, target.fingerprint(catalog="catalog-a"))
    assert store.needs_training(target, target.fingerprint(catalog="catalog-b"))


def test_an_interrupted_run_does_not_count_as_trained(tmp_path):
    write_config(tmp_path, "account", base="small-base", candidate="small-trained", agent="account-specialist")
    [target] = small_specialist_targets(config_directory=tmp_path, profiles=MODEL_PROFILES, max_weights_gib=2.0,
                                        weights_bytes=weights(900))
    store = SpecialistCandidateStore(tmp_path / "state")
    fingerprint = target.fingerprint(catalog="catalog")
    store.record(target, fingerprint=fingerprint, outcome="interrupted", candidate=None)
    assert store.needs_training(target, fingerprint)


def test_targets_take_turns_starting_with_the_longest_waiting(tmp_path):
    for name in ("access", "account", "networking"):
        write_config(tmp_path, name, base="small-base", candidate=f"{name}-trained", agent=f"{name}-specialist")
    targets = small_specialist_targets(config_directory=tmp_path, profiles=MODEL_PROFILES, max_weights_gib=2.0,
                                       weights_bytes=weights(900))
    store = SpecialistCandidateStore(tmp_path / "state")
    fingerprints = {target.name: target.fingerprint(catalog="catalog") for target in targets}
    assert store.next_target(targets, fingerprints).name == "access"
    store.record(targets[0], fingerprint=fingerprints["access"], outcome="interrupted", candidate=None)
    # An interrupted target goes to the back of the line instead of blocking the others.
    assert store.next_target(targets, fingerprints).name == "account"
    for target in targets:
        store.record(target, fingerprint=fingerprints[target.name], outcome="candidate", candidate=None)
    assert store.next_target(targets, fingerprints) is None


class Scheduler:
    active = False
    queue_depth = 0

    async def run(self, *, priority, work):
        return await asyncio.to_thread(work)


def build_trainer(tmp_path: Path, monkeypatch, *, train, evaluate):
    write_config(tmp_path, "account", base="small-base", candidate="small-trained", agent="account-specialist")
    [target] = small_specialist_targets(config_directory=tmp_path, profiles=MODEL_PROFILES, max_weights_gib=2.0,
                                        weights_bytes=weights(900))
    unloaded = []
    runtime = SimpleNamespace(active_jobs={}, gpu_scheduler=Scheduler(),
                              model_manager=SimpleNamespace(unload_all=lambda: unloaded.append(1)))
    settings = ContinualAutomationSettings(enabled=True, auto_train=True, step_pause_seconds=0.0)
    budgets = []
    monkeypatch.setattr(specialists_module, "gpu_budget", lambda **kw: budgets.append(kw) or nullcontext())
    monkeypatch.setattr(specialists_module, "train_candidate", train)
    monkeypatch.setattr(specialists_module, "evaluate_candidate", evaluate)
    monkeypatch.setattr(specialists_module, "release_training_memory", lambda: None)
    idle = SpecialistIdleTrainer(runtime=runtime, settings=settings, targets=[target],
                                 store=SpecialistCandidateStore(tmp_path / "state"),
                                 work_root=tmp_path / "work", catalog_fingerprint=lambda _target: "catalog",
                                 yield_reason=lambda: None)
    idle.budgets = budgets
    return idle, runtime, unloaded


def test_a_cycle_trains_one_candidate_checks_it_and_registers_nothing(tmp_path, monkeypatch):
    calls = {}

    def train(*, target, output_root, between_steps, limits):
        calls["train"] = target.name
        between_steps()
        return SimpleNamespace(adapter_directory=str(output_root / "runs" / "r1" / "adapter"),
                               output_directory=str(output_root / "runs" / "r1"))

    def evaluate(*, target, manifest, report_path):
        if manifest is None:
            return None  # no specialist serving yet
        calls["evaluate"] = manifest.adapter_directory
        return SimpleNamespace(passed=9, total=9, promotion_gate_passed=True)

    idle, _runtime, unloaded = build_trainer(tmp_path, monkeypatch, train=train, evaluate=evaluate)
    result = asyncio.run(idle.run_cycle())
    assert (result.specialist, result.outcome, result.gate_passed) == ("account", "candidate", True)
    assert (result.passed, result.total, result.serving_passed) == (9, 9, None)
    assert calls["train"] == "account" and calls["evaluate"].endswith("adapter")
    assert unloaded == [1]
    # The candidate stays where training wrote it; serving settings are untouched.
    assert result.candidate.endswith("adapter")
    # Trained once for these inputs: the next idle cycle has nothing to do.
    assert asyncio.run(idle.run_cycle()) is None


def test_a_candidate_is_compared_with_the_specialist_serving_now(tmp_path, monkeypatch):
    train = lambda **kwargs: SimpleNamespace(adapter_directory="a", output_directory="o")  # noqa: E731

    def evaluate(*, target, manifest, report_path):
        # The serving specialist is checked on the same held-out cases.
        if manifest is None:
            return SimpleNamespace(passed=9, total=9, promotion_gate_passed=True)
        return SimpleNamespace(passed=8, total=9, promotion_gate_passed=False)

    idle, _runtime, _unloaded = build_trainer(tmp_path, monkeypatch, train=train, evaluate=evaluate)
    result = asyncio.run(idle.run_cycle())
    assert (result.outcome, result.passed, result.serving_passed) == ("rejected", 8, 9)


def test_a_candidate_that_fails_its_held_out_cases_is_kept_as_rejected(tmp_path, monkeypatch):
    train = lambda **kwargs: SimpleNamespace(adapter_directory="a", output_directory="o")  # noqa: E731
    evaluate = lambda **kw: None if kw["manifest"] is None else SimpleNamespace(  # noqa: E731
        passed=7, total=9, promotion_gate_passed=False)
    idle, _runtime, _unloaded = build_trainer(tmp_path, monkeypatch, train=train, evaluate=evaluate)
    result = asyncio.run(idle.run_cycle())
    assert (result.outcome, result.gate_passed, result.passed, result.total) == ("rejected", False, 7, 9)
    assert asyncio.run(idle.run_cycle()) is None


def test_a_request_during_training_stops_it_and_the_specialist_is_tried_again_later(tmp_path, monkeypatch):
    def train(*, target, output_root, between_steps, limits):
        raise TrainingYielded("request_waiting")

    idle, _runtime, _unloaded = build_trainer(tmp_path, monkeypatch, train=train,
                                              evaluate=lambda **_: pytest.fail("must not evaluate"))
    result = asyncio.run(idle.run_cycle())
    assert (result.outcome, result.reason) == ("interrupted", "request_waiting")
    assert idle.store.needs_training(idle.targets[0], idle.targets[0].fingerprint(catalog="catalog"))


def test_a_request_waiting_at_handover_keeps_the_serving_models(tmp_path, monkeypatch):
    idle, _runtime, unloaded = build_trainer(tmp_path, monkeypatch, train=lambda **_: pytest.fail("must not train"),
                                             evaluate=lambda **_: pytest.fail("must not evaluate"))
    idle.yield_reason = lambda: "request_waiting"
    result = asyncio.run(idle.run_cycle())
    assert (result.outcome, result.reason) == ("interrupted", "request_waiting")
    assert unloaded == []


def test_training_uses_the_limiter_between_optimizer_steps(tmp_path, monkeypatch):
    seen = []

    def train(*, target, output_root, between_steps, limits):
        between_steps()
        seen.append("stepped")
        idle.yield_reason = lambda: "on_battery"
        between_steps()
        pytest.fail("training must stop on battery")

    idle, _runtime, _unloaded = build_trainer(tmp_path, monkeypatch, train=train,
                                              evaluate=lambda **_: pytest.fail("must not evaluate"))
    result = asyncio.run(idle.run_cycle())
    assert seen == ["stepped"]
    assert (result.outcome, result.reason) == ("interrupted", "on_battery")


def test_a_failed_run_is_recorded_and_not_retried_for_the_same_inputs(tmp_path, monkeypatch):
    def train(**_):
        raise ValueError("Target exceeds max_length: account-sft-0001")

    idle, _runtime, _unloaded = build_trainer(tmp_path, monkeypatch, train=train,
                                              evaluate=lambda **_: pytest.fail("must not evaluate"))
    result = asyncio.run(idle.run_cycle())
    assert result.outcome == "failed" and "max_length" in result.reason
    assert asyncio.run(idle.run_cycle()) is None


def test_work_files_of_a_stopped_run_are_removed(tmp_path, monkeypatch):
    def train(*, target, output_root, between_steps, limits):
        (output_root / "runs" / "r1").mkdir(parents=True)
        raise TrainingYielded("request_waiting")

    idle, _runtime, _unloaded = build_trainer(tmp_path, monkeypatch, train=train,
                                              evaluate=lambda **_: pytest.fail("must not evaluate"))
    asyncio.run(idle.run_cycle())
    assert list((tmp_path / "work").glob("*/runs/*")) == []


def test_unreadable_state_starts_fresh_instead_of_stopping_training(tmp_path):
    write_config(tmp_path, "account", base="small-base", candidate="small-trained", agent="account-specialist")
    [target] = small_specialist_targets(config_directory=tmp_path, profiles=MODEL_PROFILES, max_weights_gib=2.0,
                                        weights_bytes=weights(900))
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "specialists.json").write_text("{not json", encoding="utf-8")
    store = SpecialistCandidateStore(tmp_path / "state")
    assert store.needs_training(target, target.fingerprint(catalog="catalog"))


def test_training_memory_is_released_after_every_run(tmp_path, monkeypatch):
    released = []
    write_config(tmp_path, "account", base="small-base", candidate="small-trained", agent="account-specialist")

    def train(*, target, output_root, between_steps, limits):
        raise TrainingYielded("request_waiting")

    idle, _runtime, _unloaded = build_trainer(tmp_path, monkeypatch, train=train,
                                              evaluate=lambda **_: pytest.fail("must not evaluate"))
    monkeypatch.setattr(specialists_module, "release_training_memory", lambda: released.append(1))
    asyncio.run(idle.run_cycle())
    assert released == [1]


def test_releasing_training_memory_forgets_the_8bit_optimizer_overrides(monkeypatch):
    # The Trainer registers the embedding with bitsandbytes' process-wide optimizer
    # manager and never removes it, which keeps the whole model alive after training.
    manager = SimpleNamespace(module_weight_config_triple=[("embedding", "weight", {"optim_bits": 32})],
                              pid2config={1: {}}, index2config={(0, 0): {}}, uses_config_override=True)
    monkeypatch.setattr(specialists_module, "_bitsandbytes_manager", lambda: manager)
    specialists_module.release_training_memory()
    assert manager.module_weight_config_triple == [] and manager.pid2config == {} and manager.index2config == {}


def test_training_runs_inside_the_gpu_share_of_the_breathing_room_settings(tmp_path, monkeypatch):
    train = lambda **kwargs: SimpleNamespace(adapter_directory="a", output_directory="o")  # noqa: E731
    evaluate = lambda **kw: None if kw["manifest"] is None else SimpleNamespace(  # noqa: E731
        passed=9, total=9, promotion_gate_passed=True)
    idle, _runtime, _unloaded = build_trainer(tmp_path, monkeypatch, train=train, evaluate=evaluate)
    asyncio.run(idle.run_cycle())
    assert idle.budgets == [{"fraction": idle.settings.gpu_memory_fraction,
                             "min_free_gib": idle.settings.min_free_gpu_gib}]


class FakeProcess:
    def __init__(self):
        self.returncode, self.killed = None, False

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("worker", timeout)
        return self.returncode

    def kill(self):
        self.killed, self.returncode = True, -9


class FakeWorker:
    """LocalTransport's interface. The job 'runs' for a few polls, then writes the result it was given,
    as the real worker does; ``stops=False`` is a worker that never reaches its next step."""

    def __init__(self, root: Path, *, result, polls=2, stops=True):
        self.root, self.result, self.polls, self.stops = Path(root), result, polls, stops
        self.process, self.calls = FakeProcess(), []

    def send(self, job_folder, job_id):
        self.calls.append("send")

    def start(self, job_id):
        self.calls.append("start")
        self.job_id = job_id
        (self.root / "runs").mkdir(parents=True, exist_ok=True)  # as LocalTransport.start, with the log
        (self.root / "runs" / f"{job_id}.log").write_text("worker log", encoding="utf-8")
        return self.process

    def running(self, job_id):
        if self.process.returncode is not None:
            return False
        if self.polls:
            self.polls -= 1
            return True
        self.finish(self.result)
        return False

    def finish(self, result):
        from learning.worker.stages import fingerprint_directory

        run = self.root / "runs" / self.job_id
        run.mkdir(parents=True, exist_ok=True)
        if result is None:  # killed before it could write a result (the kernel's OOM killer)
            self.process.returncode = -9
            return
        if result["outcome"] == "trained":
            (run / "final").mkdir()
            (run / "final" / "adapter_model.safetensors").write_bytes(b"weights")
            result = result | {"final_adapter": "final", "final_adapter_sha256": fingerprint_directory(run / "final")}
        (run / "result.json").write_text(json.dumps({"job_id": self.job_id} | result), encoding="utf-8")
        self.process.returncode = 0

    def stop(self, job_id):
        self.calls.append("stop")
        (self.root / "runs" / f"{job_id}.stop").write_text("yield\n", encoding="utf-8")
        if self.stops:
            self.finish({"outcome": "interrupted", "error": "the host asked the worker to yield"})

    def fetch(self, job_id, destination=None):
        return self.root / "runs" / job_id

    def clean(self, job_id):
        self.calls.append("clean")


def train_with(tmp_path, monkeypatch, worker, *, between_steps=lambda: None):
    jobs = []

    def build_job(target, *, profiles, directory, stages, stamp, device, settings):
        jobs.append({"target": target.name, "stages": tuple(stages), "device": device, "settings": settings,
                     "directory": Path(directory), "stamp": stamp, "profiles": profiles})
        Path(directory).mkdir(parents=True)
        (Path(directory) / "job.json").write_text("{}", encoding="utf-8")
        return directory

    monkeypatch.setattr(specialists_module, "model_profiles", lambda: MODEL_PROFILES)
    monkeypatch.setattr(specialists_module, "build_specialist_job", build_job)
    monkeypatch.setattr(specialists_module, "local_transport", lambda root: worker)
    monkeypatch.setattr(specialists_module, "candidate_manifest",
                        lambda target, candidate, run_directory: SimpleNamespace(
                            adapter_directory=candidate["adapter_directory"], output_directory=str(run_directory)))
    monkeypatch.setattr(specialists_module, "WORKER_POLL_SECONDS", 0)
    write_config(tmp_path, "account", base="small-base", candidate="small-trained", agent="account-specialist")
    [target] = small_specialist_targets(config_directory=tmp_path, profiles=MODEL_PROFILES, max_weights_gib=2.0,
                                        weights_bytes=weights(900))
    manifest = specialists_module.train_candidate(
        target=target, output_root=worker.root, between_steps=between_steps,
        limits={"step_pause_seconds": 2.0, "gpu_memory_fraction": 0.85})
    return manifest, jobs


def test_the_one_portable_worker_trains_the_candidate_in_its_own_process(tmp_path, monkeypatch):
    """Owner decision (18 November): every specialist trains under one trainer; only the job differs."""
    worker = FakeWorker(tmp_path / "work" / "account", result={"outcome": "trained", "final_stage": "sft"})
    manifest, jobs = train_with(tmp_path, monkeypatch, worker)
    [job] = jobs
    assert (job["target"], job["stages"], job["device"]) == ("account", ("sft", "dpo"), "cuda")
    assert job["profiles"] is MODEL_PROFILES
    # The worker shares this machine: the idle cycle's breathing room travels with the job.
    assert job["settings"]["limits"] == {"step_pause_seconds": 2.0, "gpu_memory_fraction": 0.85}
    assert job["settings"]["sft"] == {"epochs": 4, "max_length": 1536}
    assert job["directory"] == worker.root / "jobs" / f"account-{job['stamp']}"
    assert worker.calls == ["send", "start", "clean"]
    # The candidate is the adapter the worker reported, taken only after its digest matched.
    run = worker.root / "runs" / f"account-{job['stamp']}"
    assert Path(manifest.adapter_directory) == (run / "final").resolve()
    assert manifest.output_directory == str(run)


def test_a_worker_run_in_which_no_stage_beat_the_start_gives_no_candidate(tmp_path, monkeypatch):
    worker = FakeWorker(tmp_path / "work" / "account",
                        result={"outcome": "no_improvement", "baseline": {"passed": 13, "total": 13}})
    manifest, _jobs = train_with(tmp_path, monkeypatch, worker)
    assert manifest is None


def test_a_failed_worker_run_reports_the_workers_own_error(tmp_path, monkeypatch):
    worker = FakeWorker(tmp_path / "work" / "account",
                        result={"outcome": "failed", "error": "OutOfMemoryError: CUDA out of memory"})
    with pytest.raises(RuntimeError, match="OutOfMemoryError"):
        train_with(tmp_path, monkeypatch, worker)
    assert worker.calls[-1] == "clean"


def test_a_worker_killed_before_it_wrote_a_result_fails_with_its_exit_code_and_log(tmp_path, monkeypatch):
    worker = FakeWorker(tmp_path / "work" / "account", result=None)
    with pytest.raises(RuntimeError, match=r"exit code -9 .*runs/account-\d{8}T\d{6}Z\.log"):
        train_with(tmp_path, monkeypatch, worker)


def test_a_request_while_the_worker_trains_stops_it_and_keeps_nothing(tmp_path, monkeypatch):
    worker = FakeWorker(tmp_path / "work" / "account", result={"outcome": "trained"}, polls=5)
    checks = []

    def between_steps():
        checks.append(1)
        if len(checks) == 2:
            raise TrainingYielded("request_waiting")

    with pytest.raises(TrainingYielded):
        train_with(tmp_path, monkeypatch, worker, between_steps=between_steps)
    assert worker.calls == ["send", "start", "stop", "clean"]
    assert not worker.process.killed
    # Nothing of the stopped run stays: not its output, its stop request or its log.
    assert list((worker.root / "runs").iterdir()) == []


def test_a_worker_that_does_not_reach_its_next_step_in_time_is_killed(tmp_path, monkeypatch):
    worker = FakeWorker(tmp_path / "work" / "account", result={"outcome": "trained"}, polls=5, stops=False)
    monkeypatch.setattr(specialists_module, "WORKER_GRACE_SECONDS", 0)

    def between_steps():
        raise TrainingYielded("request_waiting")

    with pytest.raises(TrainingYielded):
        train_with(tmp_path, monkeypatch, worker, between_steps=between_steps)
    assert worker.process.killed


def test_the_worker_process_gets_no_service_secrets_and_downloads_nothing(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("TAILSCALE_CLIENT_SECRET", "not-for-the-worker")
    monkeypatch.setenv("ITSM_LINK_SECRETS_FILE", "/not/for/the/worker")
    monkeypatch.setenv("HF_TOKEN", "not-needed-for-local-inputs")
    environment = specialists_module.worker_environment()
    assert environment["PATH"] == "/usr/bin:/bin"
    assert not {"TAILSCALE_CLIENT_SECRET", "ITSM_LINK_SECRETS_FILE", "HF_TOKEN"} & set(environment)
    assert environment["HF_HUB_OFFLINE"] == "1" and environment["TRANSFORMERS_OFFLINE"] == "1"


def test_a_cycle_whose_worker_improved_nothing_is_rejected_without_evaluation(tmp_path, monkeypatch):
    idle, _runtime, _unloaded = build_trainer(tmp_path, monkeypatch, train=lambda **_: None,
                                              evaluate=lambda **_: pytest.fail("nothing to evaluate"))
    result = asyncio.run(idle.run_cycle())
    assert (result.outcome, result.reason) == ("rejected", "no stage beat the specialist it started from")
    # Done for these inputs, like any other rejected candidate.
    assert asyncio.run(idle.run_cycle()) is None


def test_the_cycle_hands_the_worker_its_breathing_room(tmp_path, monkeypatch):
    seen = []

    def train(*, target, output_root, between_steps, limits):
        seen.append(limits)
        return SimpleNamespace(adapter_directory="a", output_directory="o")

    evaluate = lambda **kw: None if kw["manifest"] is None else SimpleNamespace(  # noqa: E731
        passed=9, total=9, promotion_gate_passed=True)
    idle, _runtime, _unloaded = build_trainer(tmp_path, monkeypatch, train=train, evaluate=evaluate)
    # The pause is the worker's, between its own steps; the cycle itself never sleeps it out.
    idle.settings = ContinualAutomationSettings(enabled=True, auto_train=True, step_pause_seconds=90.0,
                                                gpu_memory_fraction=0.5)
    started = time.monotonic()
    asyncio.run(idle.run_cycle())
    assert time.monotonic() - started < 10
    # A job allows at most a minute between steps.
    assert seen == [{"step_pause_seconds": 60.0, "gpu_memory_fraction": 0.5}]


def test_a_cycle_first_removes_work_a_stopped_service_left_behind(tmp_path, monkeypatch):
    output_root = tmp_path / "work" / "account"
    runs = output_root / "runs"
    (runs / "account-20261119T000000Z").mkdir(parents=True)              # stopped with the service
    (runs / "account-20261119T000000Z.log").write_text("log", encoding="utf-8")
    (runs / "account-20261118T000000Z" / "final").mkdir(parents=True)    # a candidate
    (output_root / "jobs" / "account-20261119T000000Z.partial").mkdir(parents=True)
    seen = {}

    def train(*, target, output_root, between_steps, limits):
        seen["runs"] = sorted(path.name for path in (output_root / "runs").iterdir())
        seen["jobs"] = (output_root / "jobs").exists()
        return SimpleNamespace(adapter_directory="a", output_directory="o")

    evaluate = lambda **kw: None if kw["manifest"] is None else SimpleNamespace(  # noqa: E731
        passed=9, total=9, promotion_gate_passed=True)
    idle, _runtime, _unloaded = build_trainer(tmp_path, monkeypatch, train=train, evaluate=evaluate)
    asyncio.run(idle.run_cycle())
    # Candidates and the log of the stopped run stay (for diagnosis); its half-written output goes.
    assert seen == {"runs": ["account-20261118T000000Z", "account-20261119T000000Z.log"], "jobs": False}


def test_a_candidate_that_passes_but_does_worse_than_the_serving_one_is_rejected(tmp_path, monkeypatch):
    train = lambda **kwargs: SimpleNamespace(adapter_directory="a", output_directory="o")  # noqa: E731

    def evaluate(*, target, manifest, report_path):
        # Both pass their own gate rule, but the serving specialist passed more cases.
        if manifest is None:
            return SimpleNamespace(passed=10, total=10, promotion_gate_passed=True)
        return SimpleNamespace(passed=9, total=9, promotion_gate_passed=True)

    idle, _runtime, _unloaded = build_trainer(tmp_path, monkeypatch, train=train, evaluate=evaluate)
    result = asyncio.run(idle.run_cycle())
    assert (result.outcome, result.passed, result.serving_passed) == ("rejected", 9, 10)
