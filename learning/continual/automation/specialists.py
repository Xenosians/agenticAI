"""Idle training of the small local specialists.

The specialists choose and fill in the tool calls, so they gain the most from
training. The small ones (about 0.5B parameters) also fit the GPU of an
ordinary workstation, where the Hub model does not.

One idle cycle trains one specialist whose reviewed training set changed since
its last candidate:

- targets are the reviewed configs in ``learning/config/specialists`` whose base
  model is small enough to train next to normal use of the machine;
- the one training worker (``learning/worker``) trains it in a child process,
  as it does on Proxmox and other machines: QLoRA SFT under the fit guard,
  then DPO on the model's own wrong answers; a stage is kept only when it
  beats the specialist it started from on the held-out cases;
- the cycle runs under the same breathing-room limits as the Hub cycle
  (limiter.py): the job carries the step pause and GPU share, and the worker
  stops at its next step, held-out case or pause for a request, battery or
  shutdown; the specialist is tried again after the next idle window;
- Janus's own held-out gate then checks the candidate next to the specialist
  serving now, and the candidate stays where the worker wrote it.

Nothing here registers or serves a candidate. Putting one into service stays an
explicit decision.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from learning.continual.automation.limiter import TrainingYielded, gpu_budget
from learning.paths import REPOSITORY_ROOT, RUNTIME_LEARNING_ROOT
from learning.worker_jobs import build_specialist_job
from learning.worker_transport import LocalTransport, collect_candidate

SPECIALIST_CONFIG_DIRECTORY = REPOSITORY_ROOT / "learning" / "config" / "specialists"
SPECIALIST_WORK_ROOT = RUNTIME_LEARNING_ROOT / "continuous" / "specialists"
WEIGHT_SUFFIXES = (".safetensors", ".bin")

# What an idle cycle asks of the worker: the recipe with which the worker took the access specialist
# from 0 to 13 of 13 held-out cases on a 6 GiB laptop GPU.
WORKER_STAGES = ("sft", "dpo")
WORKER_RECIPE = {"sft": {"epochs": 4, "max_length": 1536}, "dpo": {"max_length": 1536}}
WORKER_POLL_SECONDS = 0.5
# How long a worker asked to stop may take to reach its next step, case or pause before it is killed.
WORKER_GRACE_SECONDS = 120.0


@dataclass(frozen=True)
class SpecialistTarget:
    name: str
    config_path: Path
    specialist: str
    agent_definition: Path
    base_model_key: str
    candidate_model_key: str
    external_sft_jsonl: tuple[Path, ...] = field(default=())

    def fingerprint(self, *, catalog: str) -> str:
        """Everything the candidate is trained from: reviewed examples, extra files, tools, prompt."""
        digest = hashlib.sha256()
        for path in (self.config_path, *self.external_sft_jsonl):
            digest.update(path.name.encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        digest.update(catalog.encode())
        return digest.hexdigest()


def _weights_bytes(profile) -> int:
    model_path = Path(profile.model_path)
    if not model_path.is_dir():
        return 0
    return sum(path.stat().st_size for path in model_path.iterdir() if path.suffix in WEIGHT_SUFFIXES)


def small_specialist_targets(*, config_directory: Path = SPECIALIST_CONFIG_DIRECTORY, profiles: dict,
                             max_weights_gib: float,
                             weights_bytes: Callable[[object], int] = _weights_bytes) -> list[SpecialistTarget]:
    """Reviewed specialist configs whose base model is a local model small enough to train here."""
    from learning.training.specialist_sft import load_training_config

    targets = []
    for config_path in sorted(config_directory.glob("*.json")):
        config = load_training_config(config_path)
        profile = profiles.get(config.base_model_key)
        if profile is None or profile.model_path is None:
            continue
        size = weights_bytes(profile)
        if size <= 0 or size > max_weights_gib * 2**30:
            continue
        targets.append(SpecialistTarget(
            name=config_path.stem, config_path=config_path, specialist=config.specialist,
            agent_definition=Path(config.agent_definition), base_model_key=config.base_model_key,
            candidate_model_key=config.candidate_model_key,
            external_sft_jsonl=tuple(sorted(config_directory.glob(f"{config_path.stem}-*.jsonl"))),
        ))
    return targets


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SpecialistCandidateStore:
    """Per specialist: what its last candidate was trained from and how that run ended.

    Only a run that produced a candidate (accepted or rejected by its held-out
    cases) or failed for its inputs counts as done for those inputs; an
    interrupted run is tried again.
    """

    DONE = {"candidate", "rejected", "failed"}

    def __init__(self, directory: Path):
        self.path = Path(directory) / "specialists.json"

    def _load(self) -> dict:
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return state if isinstance(state, dict) else {}

    def last(self, target: SpecialistTarget) -> dict | None:
        entry = self._load().get(target.name)
        return entry if isinstance(entry, dict) else None

    def needs_training(self, target: SpecialistTarget, fingerprint: str) -> bool:
        last = self.last(target)
        return not (last and last.get("fingerprint") == fingerprint and last.get("outcome") in self.DONE)

    def next_target(self, targets: Iterable[SpecialistTarget],
                    fingerprints: dict[str, str]) -> SpecialistTarget | None:
        """The specialist with new material that has waited longest since its last attempt."""
        waiting = [target for target in targets if self.needs_training(target, fingerprints[target.name])]
        return min(waiting, key=lambda target: (self.last(target) or {}).get("attempted_at", ""), default=None)

    def record(self, target: SpecialistTarget, *, fingerprint: str, outcome: str, candidate: str | None,
               **details) -> None:
        state = self._load()
        state[target.name] = {"fingerprint": fingerprint, "outcome": outcome, "candidate": candidate,
                              "attempted_at": _utc_now(), **details}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + "." + uuid.uuid4().hex + ".tmp")
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)


@dataclass
class SpecialistCycleResult:
    specialist: str
    outcome: str  # candidate | rejected | interrupted | failed
    reason: str | None = None
    candidate: str | None = None
    gate_passed: bool | None = None
    passed: int | None = None
    total: int | None = None
    serving_passed: int | None = None
    seconds: float = 0.0


def model_profiles() -> dict:
    from config import Settings

    return Settings().model_profiles


def worker_environment() -> dict[str, str]:
    """The training process's environment: no service secrets, and no downloads (its inputs are local)."""
    keep = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH",
            "PYTORCH_CUDA_ALLOC_CONF", "JANUS_WORKER_CACHE")
    environment = {key: os.environ[key] for key in keep if key in os.environ}
    return environment | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                          "TOKENIZERS_PARALLELISM": "false"}


def local_transport(root: Path) -> LocalTransport:
    return LocalTransport(root=root, python=sys.executable, environment=worker_environment())


def candidate_manifest(target: SpecialistTarget, candidate: dict, run_directory: Path) -> SimpleNamespace:
    """What Janus's held-out gate needs to load the worker's adapter on the specialist's base model."""
    from config import Settings
    from learning.training.specialist_sft import load_training_config, resolve_worker_prompt_profile

    base = Settings().require_model_profile(target.base_model_key)
    profile = resolve_worker_prompt_profile(config=load_training_config(target.config_path), base_profile=base)
    return SimpleNamespace(base_model_path=str(base.model_path), adapter_directory=candidate["adapter_directory"],
                           worker_prompt_profile=profile, output_directory=str(run_directory))


def _discard_run(output_root: Path, job_id: str) -> None:
    """A stopped run leaves nothing: not its output, its stop request or its log."""
    runs = output_root / "runs"
    shutil.rmtree(runs / job_id, ignore_errors=True)
    for suffix in (".stop", ".log"):
        (runs / f"{job_id}{suffix}").unlink(missing_ok=True)


def _halt(transport, process, job_id: str) -> None:
    transport.stop(job_id)  # seen at the worker's next optimizer step, held-out case or pause slice
    try:
        process.wait(timeout=WORKER_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _supervise(transport, process, job_id: str, between_steps: Callable[[], None]) -> None:
    """Wait for the worker; when ``between_steps`` raises (a request, battery, shutdown) stop it first."""
    try:
        while transport.running(job_id):
            between_steps()
            time.sleep(WORKER_POLL_SECONDS)
    except BaseException:
        if transport.running(job_id):
            _halt(transport, process, job_id)
        raise


def train_candidate(*, target: SpecialistTarget, output_root: Path, between_steps: Callable[[], None],
                    limits: dict) -> SimpleNamespace | None:
    """Train one candidate with the portable worker in a child process; None when no stage beat the start.

    The worker process returns all of its memory when it exits, and a crash in
    training cannot take the serving process with it.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    job_id = f"{target.name}-{stamp}"
    folder = build_specialist_job(target, profiles=model_profiles(), directory=output_root / "jobs" / job_id,
                                  stages=WORKER_STAGES, stamp=stamp, device="cuda",
                                  settings={**WORKER_RECIPE, "limits": limits})
    transport = local_transport(output_root)
    try:
        transport.send(folder, job_id)
        process = transport.start(job_id)
        _supervise(transport, process, job_id, between_steps)
    except TrainingYielded:
        _discard_run(output_root, job_id)
        raise
    finally:
        transport.clean(job_id)  # the job folder: links to the base weights, and the data
    run_directory = transport.fetch(job_id)
    if not (run_directory / "result.json").is_file():
        raise RuntimeError(f"the worker ended with exit code {process.returncode} and wrote no result "
                           f"(log: runs/{job_id}.log)")
    result = json.loads((run_directory / "result.json").read_text(encoding="utf-8"))
    outcome = result.get("outcome")
    if outcome == "interrupted":
        _discard_run(output_root, job_id)
        raise TrainingYielded("worker_stopped")
    if outcome not in {"trained", "no_improvement"}:
        raise RuntimeError(f"worker outcome {outcome}: {result.get('error')}")
    candidate = collect_candidate(run_directory)
    return None if candidate is None else candidate_manifest(target, candidate, run_directory)


def evaluate_candidate(*, target: SpecialistTarget, manifest, report_path: Path):
    """Held-out cases of one specialist; ``manifest=None`` checks the specialist serving now.

    The candidate is evaluated under a model profile that exists only in this
    call, built from its training manifest; serving settings are not changed.
    Returns None when there is no serving specialist to compare with.
    """
    from config import Settings
    from learning.evaluation.config_specialist_gate import evaluate_specialist_config_gate

    settings = Settings()
    model_key = target.candidate_model_key
    if manifest is not None:
        base = settings.require_model_profile(target.base_model_key)
        model_key = f"{target.candidate_model_key}-idle-candidate"
        settings.model_profiles[model_key] = base.model_copy(update={
            "model_path": Path(manifest.base_model_path), "adapter_path": Path(manifest.adapter_directory),
            "worker_prompt_profile": manifest.worker_prompt_profile, "enabled": True})
    elif settings.model_profile(model_key) is None:
        return None
    return asyncio.run(evaluate_specialist_config_gate(
        project_root=REPOSITORY_ROOT, config_paths=[target.config_path], model_key=model_key,
        report_path=report_path, gate_name=f"idle-{target.name}", settings=settings))


def _bitsandbytes_manager():
    try:
        from bitsandbytes.optim import GlobalOptimManager
    except ImportError:
        return None
    return GlobalOptimManager.get_instance()


def release_training_memory() -> None:
    """Give the GPU back after a run, including memory the libraries would keep.

    Transformers registers the embedding with bitsandbytes' process-wide
    optimizer manager for every 8-bit optimizer run and never removes it; that
    reference keeps the whole trained model on the GPU after training.
    """
    manager = _bitsandbytes_manager()
    if manager is not None:
        manager.module_weight_config_triple = []
        manager.pid2config = {}
        manager.index2config = {}
    from subagents.llm.runtime.memory import release_unused_accelerator_memory

    release_unused_accelerator_memory()


def tool_catalog_fingerprint(target: SpecialistTarget) -> str:
    """The tools and worker prompt the specialist is trained against."""
    from config import Settings
    from learning.training.specialist_sft import load_training_config, resolve_worker_prompt_profile
    from subagents.core.definitions.loader import load_agent_definition
    from subagents.core.tooling.capabilities import build_agent_capability_catalog
    from subagents.core.tooling.prompt import build_worker_system_prompt

    config = load_training_config(target.config_path)
    profile = Settings().require_model_profile(target.base_model_key)
    agent = load_agent_definition(REPOSITORY_ROOT / target.agent_definition)
    catalog = build_agent_capability_catalog(agent, include_arguments=True)
    prompt = build_worker_system_prompt(agent, capability_catalog=catalog,
                                        prompt_profile=resolve_worker_prompt_profile(config=config,
                                                                                     base_profile=profile))
    return hashlib.sha256(json.dumps([catalog, prompt], sort_keys=True, default=str).encode()).hexdigest()


class SpecialistIdleTrainer:
    """Trains one small specialist candidate per idle cycle, inside the breathing-room limits."""

    TRAINING_PRIORITY = 1000

    def __init__(self, *, runtime, settings, targets: list[SpecialistTarget], store: SpecialistCandidateStore,
                 work_root: Path = SPECIALIST_WORK_ROOT,
                 catalog_fingerprint: Callable[[SpecialistTarget], str] = tool_catalog_fingerprint,
                 yield_reason: Callable[[], str | None]):
        self.runtime = runtime
        self.settings = settings
        self.targets = targets
        self.store = store
        self.work_root = Path(work_root)
        self.catalog_fingerprint = catalog_fingerprint
        self.yield_reason = yield_reason

    def _fingerprints(self) -> dict[str, str]:
        return {target.name: target.fingerprint(catalog=self.catalog_fingerprint(target))
                for target in self.targets}

    async def run_cycle(self) -> SpecialistCycleResult | None:
        fingerprints = await asyncio.to_thread(self._fingerprints)
        target = self.store.next_target(self.targets, fingerprints)
        if target is None:
            return None
        fingerprint = fingerprints[target.name]
        started = time.monotonic()
        output_root = self.work_root / target.name
        print(f"[CONTINUAL] specialist training started specialist={target.name}")

        def check() -> None:
            # While the worker trains (it pauses between its own steps) and before each held-out check.
            reason = self.yield_reason()
            if reason is not None:
                raise TrainingYielded(reason)

        def work() -> SpecialistCycleResult:
            # A request that queued while this cycle waited keeps the serving models.
            check()
            self.runtime.model_manager.unload_all()
            # Work a stopped service left behind (the job folders hold links to the base weights).
            self._remove_unfinished_runs(output_root, keep_logs=True)
            shutil.rmtree(output_root / "jobs", ignore_errors=True)
            try:
                with gpu_budget(fraction=self.settings.gpu_memory_fraction,
                                min_free_gib=self.settings.min_free_gpu_gib):
                    manifest = train_candidate(target=target, output_root=output_root, between_steps=check,
                                               limits=self._worker_limits())
                    if manifest is None:
                        return SpecialistCycleResult(specialist=target.name, outcome="rejected",
                                                     reason="no stage beat the specialist it started from")
                    check()
                    reports = Path(manifest.output_directory)
                    candidate = evaluate_candidate(target=target, manifest=manifest,
                                                   report_path=reports / "heldout-candidate.json")
                    check()
                    serving = evaluate_candidate(target=target, manifest=None,
                                                 report_path=reports / "heldout-serving.json")
            finally:
                release_training_memory()
            better = serving is None or candidate.passed >= serving.passed
            return SpecialistCycleResult(
                specialist=target.name,
                outcome="candidate" if candidate.promotion_gate_passed and better else "rejected",
                candidate=str(manifest.adapter_directory), gate_passed=bool(candidate.promotion_gate_passed),
                passed=candidate.passed, total=candidate.total,
                serving_passed=None if serving is None else serving.passed)

        try:
            result = await self.runtime.gpu_scheduler.run(priority=self.TRAINING_PRIORITY, work=work)
        except TrainingYielded as stopped:
            self._remove_unfinished_runs(output_root, keep_logs=False)
            result = SpecialistCycleResult(specialist=target.name, outcome="interrupted", reason=stopped.reason)
        except Exception as exc:  # recorded so the same inputs are not retried every idle window
            self._remove_unfinished_runs(output_root, keep_logs=True)
            result = SpecialistCycleResult(specialist=target.name, outcome="failed",
                                           reason=f"{type(exc).__name__}: {str(exc)[:300]}")
        result.seconds = round(time.monotonic() - started, 1)
        self.store.record(target, fingerprint=fingerprint, outcome=result.outcome, candidate=result.candidate,
                          reason=result.reason, gate_passed=result.gate_passed, passed=result.passed,
                          total=result.total, serving_passed=result.serving_passed, seconds=result.seconds)
        print(f"[CONTINUAL] specialist training {result.outcome} specialist={target.name} "
              f"reason={result.reason} heldout={result.passed}/{result.total} seconds={result.seconds}")
        return result

    def _worker_limits(self) -> dict:
        """The breathing room the worker keeps on this shared machine (a job allows at most 60 s)."""
        return {"step_pause_seconds": min(self.settings.step_pause_seconds, 60.0),
                "gpu_memory_fraction": self.settings.gpu_memory_fraction}

    @staticmethod
    def _remove_unfinished_runs(output_root: Path, *, keep_logs: bool) -> None:
        """Remove runs that left no candidate: no worker ``final`` adapter and no training manifest.

        A failed run keeps its log for diagnosis; an interrupted one is tried again, so its log goes too.
        """
        runs = output_root / "runs"
        if not runs.is_dir():
            return
        for run in list(runs.iterdir()):
            if run.is_dir() and not (run / "final").is_dir() and not (run / "training-manifest.json").is_file():
                shutil.rmtree(run, ignore_errors=True)
                if not keep_logs:
                    run.with_name(run.name + ".log").unlink(missing_ok=True)
        for stop in runs.glob("*.stop"):  # no worker runs while this is called
            stop.unlink(missing_ok=True)
