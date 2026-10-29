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
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pydantic import Field

from .application import (
    LearningSelectionError,
    SelectionRequest,
    model_weights,
    selection_preflight,
)
from .contracts import Digest

RUN_ID = re.compile(r"^[a-f0-9]{32}$")
DIGEST = re.compile(r"^[a-f0-9]{64}$")
ACTIVE = {"running"}
LISTED_RUNS = 20
# Trainer failures the owner may see; anything else is reported as training_failed.
TRAINER_ERRORS = {"insufficient_gpu_memory", "chat_template_unsupported", "example_too_long",
                  "cuda_unavailable", "dataset_admission_failed"}
# Variables a trainer process may inherit. Credentials never cross.
TRAINER_ENVIRONMENT = ("PATH", "HOME", "LANG", "LC_ALL", "HF_HOME", "XDG_CACHE_HOME",
                       "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF")


class RunRequest(SelectionRequest):
    preflight_sha256: Digest


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


def process_identity(pid: int) -> str | None:
    """A process's start time from /proc, so a reused process id is never mistaken for a trainer."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19] if fields[0] not in {"Z", "X"} else None
    except (OSError, IndexError):
        return None


def launch_trainer(run_dir: Path, environment: dict) -> int:
    """Start the trainer detached, so it outlives a restart of this service."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_candidate_training.py"
    log = (run_dir / "trainer.log").open("ab")
    try:
        process = subprocess.Popen([sys.executable, str(script), str(run_dir)], cwd=script.parents[1],
                                   env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   start_new_session=True, close_fds=True)
    finally:
        log.close()
    return process.pid


def stop_trainer(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


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
                 stop: Callable[[int], None] = stop_trainer):
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
        for path in self.root.glob("*/run.json"):
            if RUN_ID.fullmatch(path.parent.name):
                try:
                    rows.append((path.parent, json.loads(path.read_text())))
                except (OSError, ValueError):
                    continue
        return rows

    @staticmethod
    def _save(run_dir: Path, record: dict) -> None:
        temporary = run_dir / "run.json.tmp"
        temporary.write_text(json.dumps(record, sort_keys=True))
        os.replace(temporary, run_dir / "run.json")

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
        return status

    @staticmethod
    def _loss(value) -> float | None:
        return float(value) if type(value) in {int, float} and math.isfinite(value) and value >= 0 else None

    def _refresh(self, run_dir: Path, record: dict) -> dict:
        """Fold the trainer's report into an active record; finished records never change."""
        if record["state"] not in ACTIVE:
            return record
        status = self._status(run_dir, record)
        if status is not None:
            record["progress"]["step"] = status["step"]
            record["metrics"] = {"train_loss": self._loss(status.get("train_loss")),
                                 "validation_loss": self._loss(status.get("validation_loss"))}
            if status["state"] == "completed":
                digest = status.get("candidate_sha256")
                if isinstance(digest, str) and DIGEST.fullmatch(digest):
                    record.update(state="completed", candidate_sha256=digest)
                else:
                    record.update(state="failed", error="training_failed")
            elif status["state"] == "failed":
                error = status.get("error")
                record.update(state="failed", error=error if error in TRAINER_ERRORS else "training_failed")
        if record["state"] in ACTIVE and self._identity(record["pid"]) != record["process_identity"]:
            record["state"] = "interrupted"
        if record["state"] not in ACTIVE:
            record["finished_at"] = _now()
        self._save(run_dir, record)
        return record

    @staticmethod
    def public(record: dict) -> dict:
        return {key: record[key] for key in (
            "schema", "run_id", "state", "method", "role", "tuning", "model", "dataset", "preflight_sha256",
            "progress", "metrics", "candidate_sha256", "error", "started_at", "finished_at",
            "serving_changed", "promotion_enabled")}

    def list(self, owner: str) -> list[dict]:
        key = _owner_key(owner)
        with self._locked():
            owned = [self._refresh(run_dir, record) for run_dir, record in self._records() if record["owner"] == key]
        owned.sort(key=lambda record: (record["started_at"], record["sequence"]), reverse=True)
        return [self.public(record) for record in owned[:LISTED_RUNS]]

    def start(self, owner: str, request: RunRequest, store, *,
              weights: Callable[[dict], Path | None] = model_weights) -> dict:
        request = RunRequest.model_validate(request.model_dump())
        selection = SelectionRequest(**request.model_dump(exclude={"preflight_sha256"}))
        receipt = selection_preflight(owner, selection, store, weights=weights)
        if receipt["preflight_sha256"] != request.preflight_sha256:
            raise LearningSelectionError("learning_check_changed")
        if receipt["status"] != "ready":
            raise LearningSelectionError("learning_run_not_ready")
        rows = {row["id"]: row for row in store.list(owner)}
        model_row, dataset_row = rows[request.model_id], rows[request.dataset_id]
        base = weights(model_row)
        if base is None:
            raise LearningSelectionError("learning_run_not_ready")
        with self._locked():
            records = self._records()
            for run_dir, record in records:
                self._refresh(run_dir, record)
            if any(record["state"] in ACTIVE for _, record in records):
                raise LearningSelectionError("learning_host_busy")
            run_id = uuid.uuid4().hex
            run_dir = self.root / run_id
            run_dir.mkdir(mode=0o700)
            dataset_path = run_dir / "dataset.jsonl"
            raw = training_records(dataset_row).encode()
            dataset_path.write_bytes(raw)
            dataset = receipt["dataset"]
            steps = self.policy.steps(dataset["train_count"])
            recipe = {"objective": "sft", "role": request.role, "tuning": request.tuning, "model_path": str(base),
                      "dataset": str(dataset_path), "dataset_sha256": hashlib.sha256(raw).hexdigest(),
                      "allowed_sources": [dataset_row["source"]], "allowed_licenses": [dataset_row["license"]],
                      "output_root": str(run_dir / "candidate"), "max_steps": steps,
                      "max_length": self.policy.max_length,
                      "gradient_accumulation_steps": self.policy.gradient_accumulation_steps}
            (run_dir / "recipe.json").write_text(json.dumps(recipe, sort_keys=True))
            record = {"schema": "learning-run.v1", "run_id": run_id, "owner": _owner_key(owner), "state": "running",
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
            try:
                pid = self._launch(run_dir, trainer_environment())
                record.update(pid=pid, process_identity=self._identity(pid))
            except OSError:
                record.update(state="failed", error="trainer_unavailable", finished_at=_now())
            self._save(run_dir, record)
            return self.public(record)

    def cancel(self, owner: str, run_id: str) -> dict:
        if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
            raise LearningSelectionError("learning_run_unavailable")
        with self._locked():
            run_dir = self.root / run_id
            try:
                record = json.loads((run_dir / "run.json").read_text())
            except (OSError, ValueError):
                raise LearningSelectionError("learning_run_unavailable") from None
            if record.get("owner") != _owner_key(owner):
                raise LearningSelectionError("learning_run_unavailable")
            record = self._refresh(run_dir, record)
            if record["state"] in ACTIVE:
                # Stop only the trainer this run started, never a process that reused its id.
                self._stop(record["pid"])
                record.update(state="cancelled", finished_at=_now())
                self._save(run_dir, record)
            return self.public(record)


def run_registry() -> RunRegistry:
    from config.settings import get_settings
    return RunRegistry(get_settings().learning_run_root)
