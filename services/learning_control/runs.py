"""Candidate SFT runs started from the Learning page.

One run at a time on the host. A run starts only from the exact ready check the
owner saw, in a separate trainer process with a minimal environment (no provider
or service credentials, offline model loading). The trainer reports progress in
its own status file; this registry keeps the authoritative record. A run never
changes what serves requests and is never promoted here.

Records are files under the run root: ``<run_id>/run.json`` (written only by this
registry), ``recipe.json`` and ``dataset.jsonl`` (inputs the server chose), and
``status.json`` (written only by the trainer).
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
from types import SimpleNamespace
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pydantic import Field

from .application import (
    LearningSelectionError,
    ResourceId,
    SelectionRequest,
    model_weights,
    selection_preflight,
)
from .contracts import Digest
from .process_protocol import open_process_handle, process_identity, signal_process_handle, write_json

RUN_ID = re.compile(r"^[a-f0-9]{32}$")
DIGEST = re.compile(r"^[a-f0-9]{64}$")
ACTIVE = {"starting", "running", "cancelling", "outcome_unknown"}
TERMINAL = {"completed", "failed", "cancelled", "interrupted"}
LISTED_RUNS = 20
# Trainer failures the owner may see; anything else is reported as training_failed.
TRAINER_ERRORS = {"insufficient_gpu_memory", "chat_template_unsupported", "example_too_long",
                  "cuda_unavailable", "dataset_admission_failed"}
# Variables a trainer process may inherit. Credentials never cross. The Lua adapter
# setting (a path and its digest) does, so the trainer's weight loads use the same script.
TRAINER_ENVIRONMENT = ("PATH", "HOME", "LANG", "LC_ALL", "HF_HOME", "XDG_CACHE_HOME",
                       "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF",
                       "ITSM_LUA_ADAPTER_FILE", "ITSM_LUA_ADAPTER_SHA256")


class RunRequest(SelectionRequest):
    preflight_sha256: Digest
    submission_id: ResourceId


class TrainerUnavailable(OSError):
    """The launcher knows no trainer was started (for example exec failed)."""


@dataclass(frozen=True)
class RunPolicy:
    epochs: int = 3
    max_steps: int = 200
    gradient_accumulation_steps: int = 4
    max_length: int = 768

    def steps(self, train_count: int) -> int:
        return max(1, min(self.max_steps, math.ceil(self.epochs * train_count / self.gradient_accumulation_steps)))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _owner_key(owner: str) -> str:
    return hashlib.sha256(owner.encode()).hexdigest()


def request_digest(request: RunRequest) -> str:
    return hashlib.sha256(json.dumps(request.model_dump(mode="json"), ensure_ascii=False,
                                    sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def run_identity(owner_key: str, submission: str) -> str:
    return hashlib.sha256(f"learning-run.v2:{owner_key}:{submission}".encode()).hexdigest()[:32]


def launch_trainer(run_dir: Path, environment: dict) -> int:
    """Start the trainer detached, so it outlives a restart of this service."""
    script = Path(__file__).with_name("runner.py")
    try:
        log = (run_dir / "trainer.log").open("ab")
    except OSError as exc:
        raise TrainerUnavailable from exc
    try:
        try:
            process = subprocess.Popen([sys.executable, str(script), str(run_dir)], cwd=script.parents[2],
                                       env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       start_new_session=True, close_fds=True)
        except OSError as exc:
            raise TrainerUnavailable from exc
    finally:
        log.close()
    return process.pid


def stop_trainer(pid: int, expected_identity: str) -> None:
    """Signal the exact trainer using a pidfd; never signal a reused PID/group."""
    try:
        descriptor = open_process_handle(pid)
    except ProcessLookupError:
        return
    try:
        if process_identity(pid) == expected_identity:
            try:
                signal_process_handle(descriptor, signal.SIGTERM)
            except ProcessLookupError:
                pass
    finally:
        os.close(descriptor)


def trainer_environment() -> dict:
    environment = {key: os.environ[key] for key in TRAINER_ENVIRONMENT if key in os.environ}
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    return environment


def training_records(row: dict) -> str:
    """The dataset version as reviewed recipe records, split exactly as admitted."""
    lines = []
    for split in ("train", "validation"):
        for record in row["content"][split]:
            identity = hashlib.sha256(json.dumps(record["messages"], sort_keys=True).encode()).hexdigest()
            lines.append(json.dumps({"id": identity, "source": row["source"], "license": row["license"],
                                     "reviewed": True, "split": split, "messages": record["messages"]},
                                    ensure_ascii=False, sort_keys=True))
    return "\n".join(lines) + "\n"


class RunRegistry:
    def __init__(self, root: Path, *, policy: RunPolicy | None = None,
                 launch: Callable[[Path, dict], int] = launch_trainer,
                 identity: Callable[[int], str | None] = process_identity,
                 stop: Callable[[int, str], None] = stop_trainer):
        self.root = Path(root)
        self.policy = policy or RunPolicy()
        self._launch, self._identity, self._stop = launch, identity, stop

    @contextmanager
    def _locked(self):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (self.root / ".lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _records(self) -> list[tuple[Path, dict]]:
        rows = []
        for path in self.root.iterdir():
            if RUN_ID.fullmatch(path.name):
                rows.append((path, self._read(path)))
        return rows

    @staticmethod
    def _read(run_dir: Path) -> dict:
        try:
            if run_dir.is_symlink() or (run_dir / "run.json").is_symlink():
                raise ValueError
            record = json.loads((run_dir / "run.json").read_text())
            required = {"schema", "run_id", "owner", "state", "model", "dataset", "preflight_sha256",
                        "method", "role", "tuning", "progress", "metrics", "candidate_sha256", "error",
                        "started_at", "finished_at", "sequence", "pid", "process_identity",
                        "serving_changed", "promotion_enabled"}
            if (type(record) is not dict or not required <= set(record) or
                    record["schema"] not in {"learning-run.v1", "learning-run.v2"} or
                    record["run_id"] != run_dir.name or not DIGEST.fullmatch(record["owner"]) or
                    record["state"] not in ACTIVE | TERMINAL or
                    record["method"] != "sft" or record["role"] not in {"hub", "specialist"} or
                    record["tuning"] not in {"lora", "qlora"} or
                    not DIGEST.fullmatch(record["preflight_sha256"]) or
                    type(record["model"]) is not dict or not RUN_ID.fullmatch(record["model"].get("id", "")) or
                    set(record["model"]) != {"id", "backend", "source_kind"} or
                    record["model"]["backend"] not in {"local", "uploaded"} or record["model"]["source_kind"] != "local" or
                    type(record["dataset"]) is not dict or not RUN_ID.fullmatch(record["dataset"].get("id", "")) or
                    set(record["dataset"]) != {"id", "sha256", "version", "train_count", "validation_count"} or
                    any(type(record["dataset"][key]) is not int or record["dataset"][key] < 1
                        for key in ("version", "train_count", "validation_count")) or
                    not DIGEST.fullmatch(record["dataset"].get("sha256", "")) or
                    type(record["progress"]) is not dict or type(record["metrics"]) is not dict or
                    set(record["progress"]) != {"step", "max_steps"} or
                    set(record["metrics"]) != {"train_loss", "validation_loss"} or
                    any(value is not None and RunRegistry._loss(value) is None for value in record["metrics"].values()) or
                    type(record["progress"].get("step")) is not int or
                    type(record["progress"].get("max_steps")) is not int or
                    not 0 <= record["progress"]["step"] <= record["progress"]["max_steps"] or
                    record["progress"]["max_steps"] < 1 or
                    (record["pid"] is not None and (type(record["pid"]) is not int or record["pid"] < 1)) or
                    (record["process_identity"] is not None and
                     (not isinstance(record["process_identity"], str) or not record["process_identity"])) or
                    (record["state"] in TERMINAL and not isinstance(record["finished_at"], str)) or
                    (record["state"] in ACTIVE and record["finished_at"] is not None) or
                    (record["state"] == "completed" and not DIGEST.fullmatch(record["candidate_sha256"])) or
                    (record["state"] != "completed" and record["candidate_sha256"] is not None) or
                    record["error"] not in TRAINER_ERRORS | {None, "training_failed", "trainer_unavailable", "launch_outcome_unknown"} or
                    record["serving_changed"] is not False or record["promotion_enabled"] is not False or
                    type(record["sequence"]) is not int or not isinstance(record["started_at"], str)):
                raise ValueError
            if record["schema"] == "learning-run.v2":
                request = RunRequest.model_validate(record["request"])
                if (record["submission_id"] != request.submission_id or
                        record["request_sha256"] != request_digest(request) or
                        run_identity(record["owner"], request.submission_id) != run_dir.name or
                        not RUN_ID.fullmatch(record["launch_token"]) or
                        not DIGEST.fullmatch(record["recipe_sha256"]) or
                        not DIGEST.fullmatch(record["input_sha256"]) or
                        type(record["cancel_requested"]) is not bool or
                        any(record[key] != getattr(request, key) for key in ("method", "role", "tuning", "preflight_sha256")) or
                        record["model"]["id"] != request.model_id or record["dataset"]["id"] != request.dataset_id or
                        record["dataset"]["sha256"] != request.dataset_sha256):
                    raise ValueError
            return record
        except (OSError, ValueError, TypeError, KeyError):
            raise LearningSelectionError("learning_registry_unavailable") from None

    @staticmethod
    def _save(run_dir: Path, record: dict) -> None:
        try:
            write_json(run_dir / "run.json", record)
        except OSError:
            raise LearningSelectionError("learning_registry_unavailable") from None

    def _live(self, record: dict) -> bool | None:
        """True/False is a known process observation; None must retain its lease."""
        pid, identity = record.get("pid"), record.get("process_identity")
        if type(pid) is not int or pid < 1 or not isinstance(identity, str) or not identity:
            return None
        # Legacy identities did not pin the kernel boot, so an active legacy
        # claim cannot be safely attributed after a host restart.
        try:
            observed = self._identity(pid)
            if record["schema"] == "learning-run.v1" and observed is not None:
                return None
            return observed == identity
        except OSError:
            return None

    def _recover_process(self, run_dir: Path, record: dict) -> None:
        if type(record.get("pid")) is int and record.get("process_identity"):
            return
        try:
            receipt = json.loads((run_dir / "process.json").read_text())
            if (type(receipt) is not dict or
                    any(receipt.get(key) != record.get(key) for key in ("run_id", "request_sha256", "launch_token")) or
                    type(receipt.get("pid")) is not int or receipt["pid"] < 1 or
                    not isinstance(receipt.get("process_identity"), str) or not receipt["process_identity"]):
                return
            record.update(pid=receipt["pid"], process_identity=receipt["process_identity"])
        except (OSError, ValueError):
            return

    def _occupies_host(self, record: dict) -> bool:
        if record["state"] in ACTIVE:
            return True
        # A terminal report can precede process exit and CUDA cleanup. Preserve
        # the public outcome, but do not overlap a new trainer with that process.
        return type(record.get("pid")) is int and self._live(record) is not False

    def _status(self, run_dir: Path, record: dict) -> dict | None:
        try:
            status = json.loads((run_dir / "status.json").read_text())
        except (OSError, ValueError):
            return None
        if type(status) is not dict or status.get("state") not in {"running", "completed", "failed"}:
            return None
        step, limit = status.get("step"), status.get("max_steps")
        if type(step) is not int or type(limit) is not int or limit != record["progress"]["max_steps"] or not 0 <= step <= limit:
            return None
        if status["state"] == "completed" and step != limit:
            return None
        return status

    @staticmethod
    def _loss(value) -> float | None:
        return float(value) if type(value) in {int, float} and math.isfinite(value) and value >= 0 else None

    @staticmethod
    def _candidate_verified(run_dir: Path, record: dict, digest: str) -> bool:
        """A plausible status hash alone is not an existing candidate artifact."""
        from learning.continual.storage import fingerprint_directory
        from learning.training.recipes import AdaptationRecipe
        try:
            recipe_path = run_dir / "recipe.json"
            if hashlib.sha256(recipe_path.read_bytes()).hexdigest() != record.get("recipe_sha256"):
                return False
            recipe = AdaptationRecipe.model_validate_json(recipe_path.read_text())
            root = run_dir / "candidate"
            if root.is_symlink() or not root.resolve().is_relative_to(run_dir.resolve()):
                return False
            manifests = list(root.glob("*/manifest.json"))
            if len(manifests) != 1 or any(path.is_symlink() for path in root.rglob("*")):
                return False
            manifest_path = manifests[0]
            manifest = json.loads(manifest_path.read_text())
            artifact = Path(manifest["artifact"]).resolve()
            if (manifest.get("schema") != "learning-recipe-run.v1" or
                    manifest.get("promotion_status") != "candidate_only" or
                    manifest.get("recipe") != recipe.model_dump(mode="json") or
                    manifest.get("artifact_sha256") != digest or
                    not artifact.is_relative_to(manifest_path.parent.resolve()) or not artifact.is_dir()):
                return False
            return fingerprint_directory(artifact) == digest
        except (OSError, ValueError, TypeError, KeyError):
            return False

    def _refresh(self, run_dir: Path, record: dict) -> dict:
        """Fold the trainer's report into an active record; finished records never change."""
        if record["state"] not in ACTIVE:
            return record
        before = json.dumps(record, sort_keys=True)
        self._recover_process(run_dir, record)
        live = self._live(record)
        if live is None:
            record.update(state="outcome_unknown", error="launch_outcome_unknown", finished_at=None)
            if json.dumps(record, sort_keys=True) != before:
                self._save(run_dir, record)
            return record
        if record["state"] in {"starting", "outcome_unknown"}:
            record.update(state="cancelling" if record.get("cancel_requested") else "running", error=None)
        status = self._status(run_dir, record)
        if status is not None:
            record["progress"]["step"] = status["step"]
            record["metrics"] = {"train_loss": self._loss(status.get("train_loss")),
                                 "validation_loss": self._loss(status.get("validation_loss"))}
            if status["state"] == "completed":
                digest = status.get("candidate_sha256")
                if isinstance(digest, str) and DIGEST.fullmatch(digest) and self._candidate_verified(run_dir, record, digest):
                    record.update(state="completed", candidate_sha256=digest)
                else:
                    record.update(state="failed", error="training_failed")
            elif status["state"] == "failed":
                error = status.get("error")
                record.update(state="failed", error=error if error in TRAINER_ERRORS else "training_failed")
        if record["state"] in ACTIVE and live is False:
            record["state"] = "cancelled" if record.get("cancel_requested") else "interrupted"
        if record["state"] not in ACTIVE:
            record["finished_at"] = _now()
        if json.dumps(record, sort_keys=True) != before:
            self._save(run_dir, record)
        return record

    @staticmethod
    def public(record: dict) -> dict:
        public = {key: record[key] for key in (
            "run_id", "state", "method", "role", "tuning", "model", "dataset", "preflight_sha256",
            "progress", "metrics", "candidate_sha256", "error", "started_at", "finished_at",
            "serving_changed", "promotion_enabled")}
        return {"schema": "learning-run.v2", "submission_id": record.get("submission_id"),
                "request_sha256": record.get("request_sha256"), **public}

    def list(self, owner: str) -> list[dict]:
        key = _owner_key(owner)
        with self._locked():
            owned = [self._refresh(run_dir, record) for run_dir, record in self._records() if record["owner"] == key]
        owned.sort(key=lambda record: (record["started_at"], record["sequence"]), reverse=True)
        return [self.public(record) for record in owned[:LISTED_RUNS]]

    def resolve(self, owner: str, submission_id: str) -> dict:
        """Read-only recovery even when the bounded recent list no longer contains a run."""
        if not isinstance(submission_id, str) or not RUN_ID.fullmatch(submission_id):
            raise LearningSelectionError("learning_run_unavailable")
        key = _owner_key(owner)
        with self._locked():
            run_dir = self.root / run_identity(key, submission_id)
            if not run_dir.exists():
                raise LearningSelectionError("learning_run_unavailable")
            record = self._read(run_dir)
            if record["owner"] != key or record.get("submission_id") != submission_id:
                raise LearningSelectionError("learning_run_unavailable")
            return self.public(self._refresh(run_dir, record))

    def start(self, owner: str, request: RunRequest, store, *,
              weights: Callable[[dict], Path | None] = model_weights) -> dict:
        request = RunRequest.model_validate(request.model_dump())
        key, digest = _owner_key(owner), request_digest(request)
        with self._locked():
            run_id = run_identity(key, request.submission_id)
            run_dir = self.root / run_id
            # Resolve before current resources: a retry must recover its own
            # original run even after dataset supersession, removal or a restart.
            if run_dir.exists():
                existing = self._read(run_dir)
                if (existing["owner"] != key or existing.get("submission_id") != request.submission_id or
                        existing.get("request_sha256") != digest):
                    raise LearningSelectionError("learning_submission_conflict")
                return self.public(self._refresh(run_dir, existing))
            records = self._records()
            for run_dir, record in records:
                self._refresh(run_dir, record)
            if any(self._occupies_host(record) for _, record in records):
                raise LearningSelectionError("learning_host_busy")
            selection = SelectionRequest(**request.model_dump(exclude={"preflight_sha256", "submission_id"}))
            # Admit and serialize the same registry snapshot. A second store
            # read after preflight could otherwise select changed memberships.
            snapshot = store.list(owner)
            rows = {row["id"]: row for row in snapshot}
            pinned_weights = {}
            def pin_weights(row):
                path = weights(row)
                pinned_weights[row["id"]] = path
                return path
            view = SimpleNamespace(list=lambda given_owner: snapshot if given_owner == owner else [])
            receipt = selection_preflight(owner, selection, view, weights=pin_weights)
            if receipt["preflight_sha256"] != request.preflight_sha256:
                raise LearningSelectionError("learning_check_changed")
            if receipt["status"] != "ready":
                raise LearningSelectionError("learning_run_not_ready")
            model_row, dataset_row = rows[request.model_id], rows[request.dataset_id]
            base = pinned_weights.get(request.model_id)
            if base is None:
                raise LearningSelectionError("learning_run_not_ready")
            run_dir = self.root / run_id
            run_dir.mkdir(mode=0o700)
            dataset_path = run_dir / "dataset.jsonl"
            raw = training_records(dataset_row).encode()
            with dataset_path.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            dataset = receipt["dataset"]
            steps = self.policy.steps(dataset["train_count"])
            recipe = {"objective": "sft", "role": request.role, "tuning": request.tuning, "model_path": str(base),
                      "dataset": str(dataset_path), "dataset_sha256": hashlib.sha256(raw).hexdigest(),
                      "allowed_sources": [dataset_row["source"]], "allowed_licenses": [dataset_row["license"]],
                      "output_root": str(run_dir / "candidate"), "max_steps": steps,
                      "max_length": self.policy.max_length,
                      "gradient_accumulation_steps": self.policy.gradient_accumulation_steps}
            write_json(run_dir / "recipe.json", recipe)
            record = {"schema": "learning-run.v2", "run_id": run_id, "owner": key, "state": "starting",
                      "submission_id": request.submission_id, "request_sha256": digest,
                      "request": request.model_dump(mode="json"), "launch_token": os.urandom(16).hex(),
                      "recipe_sha256": hashlib.sha256((run_dir / "recipe.json").read_bytes()).hexdigest(),
                      "input_sha256": hashlib.sha256(raw).hexdigest(), "cancel_requested": False,
                      "method": request.method, "role": request.role, "tuning": request.tuning,
                      "model": {key: receipt["model"][key] for key in ("id", "backend", "source_kind")},
                      "dataset": {key: dataset[key] for key in ("id", "sha256", "version", "train_count",
                                                                 "validation_count")},
                      "preflight_sha256": request.preflight_sha256,
                      "progress": {"step": 0, "max_steps": steps},
                      "metrics": {"train_loss": None, "validation_loss": None},
                      "candidate_sha256": None, "error": None, "started_at": _now(), "finished_at": None,
                      "sequence": 1 + max((r.get("sequence", 0) for _, r in records), default=0),
                      "serving_changed": False, "promotion_enabled": False, "pid": None, "process_identity": None}
            self._save(run_dir, record)
            try:
                pid = self._launch(run_dir, trainer_environment())
                if type(pid) is not int or pid < 1:
                    raise OSError("Invalid trainer identity")
                identity = self._identity(pid)
                record.update(pid=pid, process_identity=identity, state="running" if identity else "outcome_unknown",
                              error=None if identity else "launch_outcome_unknown")
            except TrainerUnavailable:
                record.update(state="failed", error="trainer_unavailable", finished_at=_now())
            except OSError:
                record.update(state="outcome_unknown", error="launch_outcome_unknown")
            self._save(run_dir, record)
            return self.public(record)

    def cancel(self, owner: str, run_id: str) -> dict:
        if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
            raise LearningSelectionError("learning_run_unavailable")
        with self._locked():
            run_dir = self.root / run_id
            if not run_dir.exists():
                raise LearningSelectionError("learning_run_unavailable")
            record = self._read(run_dir)
            if record.get("owner") != _owner_key(owner):
                raise LearningSelectionError("learning_run_unavailable")
            record = self._refresh(run_dir, record)
            if record["state"] in ACTIVE:
                if self._live(record) is not True:
                    raise LearningSelectionError("learning_run_outcome_unknown")
                record.update(state="cancelling", cancel_requested=True)
                self._save(run_dir, record)
                # Stop only the trainer this run started, never a process that reused its id.
                try:
                    self._stop(record["pid"], record["process_identity"])
                except OSError:
                    raise LearningSelectionError("learning_run_outcome_unknown") from None
                record = self._refresh(run_dir, record)
            return self.public(record)


def run_registry() -> RunRegistry:
    from config.settings import get_settings
    return RunRegistry(get_settings().learning_run_root)
