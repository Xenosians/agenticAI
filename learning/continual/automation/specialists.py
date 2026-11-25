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

Proxmox comes first (owner decisions 18 November 2026, 23:30 WIB): when a
registered remote worker (the ``janus-trainer`` container) is free and the
specialist fits it, ``tend_remote`` sends it the same job over SSH; that needs
neither the GPU nor an idle machine. A finished run is fetched, also kept on
the worker, and judged by Janus's gate in the next idle window, like a local
one. What does not fit the worker trains on the local GPU when it is idle.

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
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from learning.continual.automation.limiter import TrainingYielded, gpu_budget
from learning.paths import REPOSITORY_ROOT, RUNTIME_LEARNING_ROOT
from learning.worker_dispatch import MIN_KAGGLE_HOURS, Worker, fits
from learning.worker_jobs import build_specialist_job
from learning.worker_transport import (LocalTransport, SshTransport, TransportError, build_worker_package,
                                       collect_candidate, worker_source_digest)

SPECIALIST_CONFIG_DIRECTORY = REPOSITORY_ROOT / "learning" / "config" / "specialists"
SPECIALIST_WORK_ROOT = RUNTIME_LEARNING_ROOT / "continuous" / "specialists"
WEIGHT_SUFFIXES = (".safetensors", ".bin")

# What an idle cycle asks of the worker: the recipe with which the worker took the access specialist
# from 0 to 13 of 13 held-out cases on a 6 GiB laptop GPU. ``max_length`` is a floor, not a cut: the worker
# raises it to fit the longest prompt and answer (up to its own ceiling) and reports the effective value.
WORKER_STAGES = ("sft", "dpo")
WORKER_RECIPE = {"sft": {"epochs": 4, "max_length": 1536}, "dpo": {"max_length": 1536}}
WORKER_POLL_SECONDS = 0.5
# How long a worker asked to stop may take to reach its next step, case or pause before it is killed.
WORKER_GRACE_SECONDS = 120.0
# The controller ticks every few seconds; each look at a remote worker is an SSH connection (15 s when it is off).
REMOTE_POLL_SECONDS = 60.0
REMOTE_RETRY_SECONDS = 600.0
# How often the remote workers are tended while the local GPU trains or judges a candidate.
REMOTE_TEND_SECONDS = 5.0
# A failed run is retried for the same inputs after this long even when the worker code did not change.
FAILED_RETRY_SECONDS = 6 * 60 * 60
RSYNC_PARTIAL = frozenset({23, 24})   # rsync: some or all of the files were not there


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


@dataclass(frozen=True)
class RemoteWorker:
    """A registered remote training machine (``~/.config/janus-live/<name>.json``): reached over SSH, or
    Kaggle's API (``kind`` kaggle, with a 0600 token file instead of SSH details)."""
    name: str
    kind: str
    device: str
    ram_gib: float
    threads: int
    host: str = ""
    user: str = ""
    key: Path | None = None
    known_hosts: Path | None = None
    gpu_gib: float = 0.0
    token_file: Path | None = None

    def capacity(self) -> Worker:
        return Worker(name=self.name, kind=self.kind, device=self.device, ram_gib=self.ram_gib, gpu_gib=self.gpu_gib)


def load_remote_worker(path: Path | None) -> RemoteWorker | None:
    """The worker file the first boot wrote, or None when there is none or it is not in the known shape."""
    if path is None:
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        worker = RemoteWorker(name=str(data["name"]), kind=str(data["kind"]), device=str(data["device"]),
                              host=str(data["host"]), user=str(data["user"]), key=Path(data["key"]),
                              known_hosts=Path(data["known_hosts"]), ram_gib=float(data["ram_gib"]),
                              threads=int(data["threads"]), gpu_gib=float(data.get("gpu_gib", 0.0)))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if worker.kind not in ("proxmox", "local") or worker.device not in ("cpu", "cuda") \
            or not (1 <= worker.threads <= 256):
        return None
    return worker


def load_kaggle_worker(path: Path | None) -> RemoteWorker | None:
    """The Kaggle worker file (``kind`` kaggle, a CUDA device and a token file), or None."""
    if path is None:
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        worker = RemoteWorker(name=str(data["name"]), kind=str(data["kind"]), device=str(data["device"]),
                              ram_gib=float(data["ram_gib"]), threads=int(data["threads"]),
                              gpu_gib=float(data["gpu_gib"]), token_file=Path(data["token_file"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if worker.kind != "kaggle" or worker.device != "cuda" or worker.gpu_gib <= 0 \
            or not worker.token_file.is_absolute() or not (1 <= worker.threads <= 256):
        return None
    return worker


def read_token_file(path: Path) -> str:
    """A secret kept in a file only its owner reads (mode 0600); never an environment variable or argument."""
    import stat

    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_size > 4096:
                raise TransportError("the token file must be a private regular file (mode 0600)")
            return handle.read().strip()
    except OSError:
        raise TransportError("the token file cannot be read") from None


def kaggle_transport(worker: RemoteWorker):
    from learning.kaggle_worker import connect

    # No package is built here: the trainer installs the worker package it built for this dispatch
    # (see SpecialistIdleTrainer._install_worker) before the first send. A second build in the shared
    # folder could race that one, and a connect is not a dispatch.
    return connect(token=read_token_file(worker.token_file))


def kaggle_hours_left(transport) -> float | None:
    from learning.kaggle_worker import weekly_gpu_hours_left

    return weekly_gpu_hours_left(transport)


def remote_transport(worker: RemoteWorker) -> SshTransport:
    return SshTransport(host=worker.host, user=worker.user, key=worker.key, known_hosts=worker.known_hosts,
                        environment={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                                     "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": str(worker.threads)})


def _attempted_at(entry: dict) -> datetime | None:
    value = entry.get("attempted_at")
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


class SpecialistCandidateStore:
    """Per specialist: what its last candidate was trained from and how that run ended.

    Only a run that produced a candidate (accepted or rejected by its held-out cases) counts as done for
    those inputs; an interrupted run is tried again. A failed run is retried when the worker code changed
    (its digest differs) or after ``FAILED_RETRY_SECONDS``, so a fixed worker is not blocked forever by an
    old failure. A run dispatched to a remote worker, or fetched from one and waiting for Janus's gate, is
    not trained again meanwhile. The store is safe for the local cycle and the remote/Kaggle tending to
    read and write it at the same time.
    """

    DONE = {"candidate", "rejected"}
    PENDING = {"dispatched", "fetched"}
    FAILED = "failed"

    def __init__(self, directory: Path):
        self.path = Path(directory) / "specialists.json"
        self._lock = threading.RLock()

    def _load(self) -> dict:
        with self._lock:
            try:
                state = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {}
            return state if isinstance(state, dict) else {}

    def last(self, target: SpecialistTarget) -> dict | None:
        with self._lock:
            entry = self._load().get(target.name)
            return entry if isinstance(entry, dict) else None

    def needs_training(self, target: SpecialistTarget, fingerprint: str, *, worker_digest: str | None = None,
                       now: datetime | None = None) -> bool:
        with self._lock:
            last = self.last(target)
            if not last or last.get("fingerprint") != fingerprint:
                return True
            outcome = last.get("outcome")
            if outcome in self.DONE or outcome in self.PENDING:
                return False
            if outcome != self.FAILED:
                return True                      # interrupted or an unknown outcome: try again
            recorded = last.get("worker_digest")
            if worker_digest and recorded and recorded != worker_digest:
                return True                      # the worker changed: the old failure may be fixed
            attempted = _attempted_at(last)
            if attempted is None:
                return True
            return ((now or datetime.now(timezone.utc)) - attempted).total_seconds() >= FAILED_RETRY_SECONDS

    def pending(self, targets: Iterable[SpecialistTarget], outcome: str) -> list[tuple[SpecialistTarget, dict]]:
        with self._lock:
            return [(target, last) for target in targets
                    if (last := self.last(target)) and last.get("outcome") == outcome]

    def next_target(self, targets: Iterable[SpecialistTarget], fingerprints: dict[str, str], *,
                    worker_digest: str | None = None, now: datetime | None = None) -> SpecialistTarget | None:
        """The specialist with new material that has waited longest since its last attempt."""
        with self._lock:
            waiting = [target for target in targets
                       if self.needs_training(target, fingerprints[target.name], worker_digest=worker_digest, now=now)]
            return min(waiting, key=lambda target: (self.last(target) or {}).get("attempted_at", ""), default=None)

    def record(self, target: SpecialistTarget, *, fingerprint: str, outcome: str, candidate: str | None,
               **details) -> None:
        with self._lock:
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
    outcome: str  # candidate | rejected | interrupted | failed | dispatched | fetched
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


def _no_improvement_reason(result: dict) -> str:
    """Why a run kept nothing: one reason shared by every stage that did not run (e.g. a perfect start)."""
    skipped = {str(stage.get("reason")) for stage in result.get("stages") or []
               if isinstance(stage, dict) and not stage.get("ran", True) and stage.get("reason")}
    return skipped.pop()[:300] if len(skipped) == 1 else "no stage beat the specialist it started from"


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
    """Trains one small specialist candidate per idle cycle, inside the breathing-room limits.

    With a registered remote worker, ``tend_remote`` (called on every controller tick) sends the worker
    what fits it and brings finished runs back; ``run_cycle`` judges fetched runs first, then trains locally.
    """

    TRAINING_PRIORITY = 1000

    def __init__(self, *, runtime, settings, targets: list[SpecialistTarget], store: SpecialistCandidateStore,
                 work_root: Path = SPECIALIST_WORK_ROOT,
                 catalog_fingerprint: Callable[[SpecialistTarget], str] = tool_catalog_fingerprint,
                 yield_reason: Callable[[], str | None], remote: RemoteWorker | None = None,
                 remote_transport_factory: Callable[[RemoteWorker], object] = remote_transport,
                 kaggle: RemoteWorker | None = None,
                 kaggle_transport_factory: Callable[[RemoteWorker], object] = kaggle_transport,
                 local_max_weights_gib: float = float("inf"),
                 worker_digest: Callable[[], str] = worker_source_digest):
        self.runtime = runtime
        self.settings = settings
        self.targets = targets
        self.store = store
        self.work_root = Path(work_root)
        self.catalog_fingerprint = catalog_fingerprint
        self.yield_reason = yield_reason
        self.remote = remote
        self.remote_transport_factory = remote_transport_factory
        self.weights_gib = lambda target: _weights_bytes(model_profiles()[target.base_model_key]) / 2**30
        self.clock: Callable[[], float] = time.monotonic
        self._next_remote_look = 0.0
        self._remote_reachable = remote is not None   # until a look says otherwise
        self.kaggle = kaggle
        self.kaggle_transport_factory = kaggle_transport_factory
        self.kaggle_hours = kaggle_hours_left
        # The laptop's own limit (CONTINUAL_SPECIALIST_MAX_WEIGHTS_GIB): what is bigger is Kaggle's to train.
        self.local_max_weights_gib = local_max_weights_gib
        self._next_kaggle_look = 0.0
        # A failed run is retried when the worker code that ran changed (see worker_source_digest).
        self.worker_digest = worker_digest
        # One worker package for every remote/Kaggle dispatch; rebuilt only when the worker source changes.
        self._package_lock = threading.RLock()
        self._package_path: Path | None = None
        self._package_digest: str | None = None
        # The specialist this machine is training right now: remote tending must not dispatch it too.
        self._training: str | None = None
        self.remote_tend_seconds = REMOTE_TEND_SECONDS

    def _fingerprints(self) -> dict[str, str]:
        return {target.name: target.fingerprint(catalog=self.catalog_fingerprint(target))
                for target in self.targets}

    def _digest(self) -> str:
        return self.worker_digest()

    def _worker_package(self) -> Path:
        """The worker package every remote dispatch installs; built once per worker source digest."""
        digest = self._digest()
        with self._package_lock:
            if (self._package_digest != digest or self._package_path is None
                    or not self._package_path.is_dir()):
                self._package_path = build_worker_package(self.work_root / "worker-package")
                self._package_digest = digest
            return self._package_path

    def _install_worker(self, transport) -> None:
        """Install the shared package under the lock, so a rebuild cannot race a running install."""
        with self._package_lock:
            transport.install(self._worker_package())

    async def _tend_while(self, work):
        """Await a long local step while tending Proxmox and Kaggle, so neither waits for the laptop."""
        task = asyncio.ensure_future(work)
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=self.remote_tend_seconds)
                if task in done:
                    return task.result()
                try:
                    await self.tend_remote()
                except asyncio.CancelledError:
                    raise
                except Exception as error:  # a worker tick never stops the local step
                    print("[CONTINUAL] tending while training failed "
                          f"error={type(error).__name__}: {str(error)[:200]}")
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    # -- the remote worker (Proxmox first) --------------------------------------------------------------------

    async def tend_remote(self) -> SpecialistCycleResult | None:
        """Bring back a finished remote run, or send a remote worker the next job that fits it.

        The Proxmox trainer first (owner decision 1); Kaggle's GPU for what fits nowhere else.
        """
        result = None
        if self.remote is not None:
            result = await asyncio.to_thread(self._tend_remote)
        if result is None and self.kaggle is not None:
            result = await asyncio.to_thread(self._tend_kaggle)
        return result

    def _tend_remote(self) -> SpecialistCycleResult | None:
        now = self.clock()
        if now < self._next_remote_look:
            return None
        self._next_remote_look = now + REMOTE_POLL_SECONDS
        transport = self.remote_transport_factory(self.remote)
        try:
            result = self._look_at_remote(transport)
        except TransportError as error:
            # Unreachable for now (powered off, Wi-Fi): nothing changes; look again in a while, not every tick.
            self._next_remote_look = now + REMOTE_RETRY_SECONDS
            self._remote_reachable = False              # unreachable or broken: the idle laptop may take its work
            print(f"[CONTINUAL] remote worker unavailable worker={self.remote.name} reason={str(error)[:200]}")
            return None
        self._remote_reachable = True
        return result

    def _look_at_remote(self, transport) -> SpecialistCycleResult | None:
        for target, last in self.store.pending(self.targets, "dispatched"):
            if last.get("worker") != self.remote.name:
                continue
            if transport.running(last["job_id"]):
                return None
            return self._collect_remote(transport, target, last)
        fingerprints = self._fingerprints()
        digest = self._digest()
        waiting = [target for target in self.targets
                   if target.name != self._training
                   and self.store.needs_training(target, fingerprints[target.name], worker_digest=digest)
                   and fits(self.remote.capacity(), weights_gib=self.weights_gib(target), stages=list(WORKER_STAGES))]
        target = self.store.next_target(waiting, fingerprints, worker_digest=digest)
        if target is None or transport.busy():      # nothing to send: the trainer is not contacted
            return None
        return self._dispatch_remote(transport, target, fingerprints[target.name], self.remote, digest)

    # -- Kaggle's free GPU (owner decision 8): what fits neither the trainer nor this machine ------------------

    def _kaggle_only(self, target: SpecialistTarget) -> bool:
        weights = self.weights_gib(target)
        on_trainer = self.remote is not None and fits(self.remote.capacity(), weights_gib=weights,
                                                      stages=list(WORKER_STAGES))
        return (not on_trainer and weights > self.local_max_weights_gib
                and fits(self.kaggle.capacity(), weights_gib=weights, stages=list(WORKER_STAGES)))

    def _tend_kaggle(self) -> SpecialistCycleResult | None:
        now = self.clock()
        if now < self._next_kaggle_look:
            return None
        self._next_kaggle_look = now + REMOTE_POLL_SECONDS
        digest = self._digest()
        fingerprints = None
        pending = [(target, last) for target, last in self.store.pending(self.targets, "dispatched")
                   if last.get("worker") == self.kaggle.name]
        if not pending:
            fingerprints = self._fingerprints()
            waiting = [target for target in self.targets
                       if target.name != self._training
                       and self.store.needs_training(target, fingerprints[target.name], worker_digest=digest)
                       and self._kaggle_only(target)]
            if not waiting:
                return None                          # nothing for Kaggle: no API call at all
        try:
            transport = self.kaggle_transport_factory(self.kaggle)
            for target, last in pending:
                if transport.running(last["job_id"]):
                    return None
                return self._collect_remote(transport, target, last)
            target = self.store.next_target(waiting, fingerprints, worker_digest=digest)
            hours = self.kaggle_hours(transport)
            if target is None or hours is None or hours < MIN_KAGGLE_HOURS or transport.busy():
                return None
            return self._dispatch_remote(transport, target, fingerprints[target.name], self.kaggle, digest)
        except Exception as error:  # noqa: BLE001 - Kaggle unreachable or refusing: try again later
            self._next_kaggle_look = now + REMOTE_RETRY_SECONDS
            print(f"[CONTINUAL] kaggle unavailable reason={type(error).__name__}: {str(error)[:200]}")
            return None

    def _dispatch_remote(self, transport, target: SpecialistTarget, fingerprint: str,
                         worker: RemoteWorker, worker_digest: str) -> SpecialistCycleResult:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        job_id = f"{target.name}-{stamp}"
        output_root = self.work_root / target.name
        folder = build_specialist_job(target, profiles=model_profiles(), directory=output_root / "jobs" / job_id,
                                      stages=WORKER_STAGES, stamp=stamp, device=worker.device,
                                      settings={**WORKER_RECIPE, "threads": worker.threads},
                                      base_by_reference=worker.kind == "kaggle")
        try:
            self._install_worker(transport)
            transport.send(folder, job_id)
            transport.start(job_id)
        finally:
            shutil.rmtree(folder, ignore_errors=True)   # links to the base weights and the data; sent or not
        previous = self.store.last(target) or {}
        losses = int(previous.get("losses") or 0) if previous.get("fingerprint") == fingerprint else 0
        self.store.record(target, fingerprint=fingerprint, outcome="dispatched", candidate=None, job_id=job_id,
                          worker=worker.name, worker_digest=worker_digest, losses=losses)
        print(f"[CONTINUAL] specialist job dispatched specialist={target.name} worker={worker.name} "
              f"job={job_id}")
        return SpecialistCycleResult(specialist=target.name, outcome="dispatched", reason=worker.name)

    def _collect_remote(self, transport, target: SpecialistTarget, last: dict) -> SpecialistCycleResult:
        job_id = last["job_id"]
        details = {key: last.get(key) for key in ("job_id", "worker", "worker_digest")} | {
            "losses": int(last.get("losses") or 0)}
        destination = self.work_root / target.name / "runs" / job_id
        try:
            run_directory = Path(transport.fetch(job_id, destination))
        except TransportError as error:
            if error.returncode not in RSYNC_PARTIAL:
                raise                                   # unreachable or unknown, not lost: the job keeps waiting
            run_directory = destination                 # what did arrive, if anything, is checked below
        transport.clean(job_id, keep_result=True)   # results are stored on Proxmox too (owner decision 1)
        result_file = run_directory / "result.json"
        if not result_file.is_file():
            return self._lost_remote(target, last, details)
        try:
            result = json.loads(result_file.read_text(encoding="utf-8"))
            candidate = collect_candidate(Path(run_directory)) if result.get("outcome") == "trained" else None
        except (OSError, ValueError, TransportError) as error:
            result, candidate = {"outcome": "failed", "error": f"{type(error).__name__}: {error}"}, None
        outcome = result.get("outcome")
        if candidate is not None:
            recorded = SpecialistCycleResult(specialist=target.name, outcome="fetched",
                                             candidate=candidate["adapter_directory"])
            self.store.record(target, fingerprint=last["fingerprint"], outcome="fetched",
                              candidate=candidate["adapter_directory"], run_directory=str(run_directory), **details)
        else:
            mapped = {"no_improvement": "rejected", "interrupted": "interrupted"}.get(outcome, "failed")
            reason = (_no_improvement_reason(result) if mapped == "rejected"
                      else str(result.get("error") or outcome)[:300])
            recorded = SpecialistCycleResult(specialist=target.name, outcome=mapped, reason=reason)
            self.store.record(target, fingerprint=last["fingerprint"], outcome=mapped, candidate=None, reason=reason,
                              **details)
        print(f"[CONTINUAL] specialist job {recorded.outcome} specialist={target.name} worker={last.get('worker')} "
              f"job={job_id}")
        return recorded

    def _lost_remote(self, target: SpecialistTarget, last: dict, details: dict) -> SpecialistCycleResult:
        """The run ended without a result: a power loss stops the unit for good, and so does the kernel's
        out-of-memory killer. The first loss is an interruption (tried again); a second one for the same
        inputs is a failure, so a job that cannot fit is not sent forever."""
        losses = details["losses"] + 1
        if losses >= 2:
            outcome, reason = "failed", "the run on the trainer ended without a result twice"
        else:
            outcome, reason = "interrupted", "the run on the trainer ended without a result"
        self.store.record(target, fingerprint=last["fingerprint"], outcome=outcome, candidate=None, reason=reason,
                          **(details | {"losses": losses}))
        print(f"[CONTINUAL] specialist job {outcome} specialist={target.name} worker={last.get('worker')} "
              f"job={last['job_id']} reason={reason}")
        return SpecialistCycleResult(specialist=target.name, outcome=outcome, reason=reason)

    def _local(self, target: SpecialistTarget) -> bool:
        """Within this machine's own limit (CONTINUAL_SPECIALIST_MAX_WEIGHTS_GIB); bigger is Kaggle's."""
        if self.local_max_weights_gib == float("inf"):
            return True                                   # no limit set: nothing needs measuring
        return self.weights_gib(target) <= self.local_max_weights_gib

    def _defer_to_remote(self, target: SpecialistTarget) -> bool:
        """Proxmox first: what fits the trainer waits for it while its last look succeeded."""
        return (self.remote is not None and self._remote_reachable
                and fits(self.remote.capacity(), weights_gib=self.weights_gib(target), stages=list(WORKER_STAGES)))

    # -- the idle cycle on this machine -------------------------------------------------------------------------

    async def run_cycle(self) -> SpecialistCycleResult | None:
        fetched = self.store.pending(self.targets, "fetched")
        if fetched:
            return await self._judge_fetched(*fetched[0])
        fingerprints = await asyncio.to_thread(self._fingerprints)
        digest = await asyncio.to_thread(self._digest)
        target = self.store.next_target([target for target in self.targets if not self._defer_to_remote(target)
                                         and self._local(target)], fingerprints, worker_digest=digest)
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
                    return self._judge(target, manifest)
            finally:
                release_training_memory()

        # While this machine trains, keep sending/collecting remote work: the trainer and Kaggle must not
        # wait for a local run that can take an hour. ``_training`` keeps the same specialist from being
        # dispatched remotely meanwhile.
        self._training = target.name
        try:
            try:
                result = await self._tend_while(
                    self.runtime.gpu_scheduler.run(priority=self.TRAINING_PRIORITY, work=work))
            except TrainingYielded as stopped:
                self._remove_unfinished_runs(output_root, keep_logs=False)
                result = SpecialistCycleResult(specialist=target.name, outcome="interrupted", reason=stopped.reason)
            except Exception as exc:  # recorded so the same inputs are not retried every idle window
                self._remove_unfinished_runs(output_root, keep_logs=True)
                result = SpecialistCycleResult(specialist=target.name, outcome="failed",
                                               reason=f"{type(exc).__name__}: {str(exc)[:300]}")
            return self._finish(target, fingerprint, result, started, worker_digest=digest)
        finally:
            self._training = None

    def _finish(self, target: SpecialistTarget, fingerprint: str, result: SpecialistCycleResult, started: float,
                **details) -> SpecialistCycleResult:
        result.seconds = round(time.monotonic() - started, 1)
        details.setdefault("worker_digest", self._digest())
        self.store.record(target, fingerprint=fingerprint, outcome=result.outcome, candidate=result.candidate,
                          reason=result.reason, gate_passed=result.gate_passed, passed=result.passed,
                          total=result.total, serving_passed=result.serving_passed, seconds=result.seconds, **details)
        print(f"[CONTINUAL] specialist training {result.outcome} specialist={target.name} "
              f"reason={result.reason} heldout={result.passed}/{result.total} seconds={result.seconds}")
        return result

    def _judge(self, target: SpecialistTarget, manifest) -> SpecialistCycleResult:
        """Janus's held-out gate on a candidate, next to the specialist serving now."""
        reports = Path(manifest.output_directory)
        candidate = evaluate_candidate(target=target, manifest=manifest, report_path=reports / "heldout-candidate.json")
        serving = evaluate_candidate(target=target, manifest=None, report_path=reports / "heldout-serving.json")
        better = serving is None or candidate.passed >= serving.passed
        return SpecialistCycleResult(
            specialist=target.name, outcome="candidate" if candidate.promotion_gate_passed and better else "rejected",
            candidate=str(manifest.adapter_directory), gate_passed=bool(candidate.promotion_gate_passed),
            passed=candidate.passed, total=candidate.total, serving_passed=None if serving is None else serving.passed)

    async def _judge_fetched(self, target: SpecialistTarget, last: dict) -> SpecialistCycleResult:
        """A run a remote worker trained: Janus's own gate on this machine, in an idle window."""
        started = time.monotonic()
        details = {key: last.get(key) for key in ("job_id", "worker", "worker_digest", "run_directory")}

        def work() -> SpecialistCycleResult:
            if self.yield_reason() is not None:
                raise TrainingYielded(self.yield_reason() or "request_waiting")
            self.runtime.model_manager.unload_all()
            try:
                run_directory = Path(last["run_directory"])
                candidate = collect_candidate(run_directory)
                if candidate is None:
                    raise TransportError("the fetched run holds no trained adapter")
                return self._judge(target, candidate_manifest(target, candidate, run_directory))
            finally:
                release_training_memory()

        try:
            result = await self._tend_while(
                self.runtime.gpu_scheduler.run(priority=self.TRAINING_PRIORITY, work=work))
        except TrainingYielded:
            return SpecialistCycleResult(specialist=target.name, outcome="interrupted", reason="request_waiting")
        except Exception as exc:
            result = SpecialistCycleResult(specialist=target.name, outcome="failed",
                                           reason=f"{type(exc).__name__}: {str(exc)[:300]}")
        return self._finish(target, last["fingerprint"], result, started, **details)

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
