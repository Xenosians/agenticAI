"""Move a job folder to a worker, start it, and bring back its result.

``SshTransport`` reaches a Linux worker (the Proxmox ``janus-trainer`` container, a second PC) with a
dedicated key and a pinned host key; ``LocalTransport`` runs the same worker in a child process on this
machine. Both run exactly ``python -m learning.worker.stages``; only the place differs.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

JOB_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}-\d{8}T\d{6}Z$")
WORKER_HOME_CODE = "janus-worker"       # the worker package lives in ~/janus-worker on remote workers
ENVIRONMENT_NAME = re.compile(r"[A-Z][A-Z0-9_]{0,63}")
ENVIRONMENT_VALUE = re.compile(r"[A-Za-z0-9_.:/+=,-]{0,256}")
REPOSITORY = Path(__file__).resolve().parents[1]


def build_worker_package(directory: Path, *, repository: Path = REPOSITORY) -> Path:
    """The worker as a remote machine runs it: ``learning/worker`` and its requirements, nothing else.

    The service's ``learning/__init__.py`` imports the whole service, so the package carries an empty one:
    ``python -m learning.worker.stages`` must not import anything a worker does not install.
    """
    directory = Path(directory)
    shutil.rmtree(directory, ignore_errors=True)
    shutil.copytree(repository / "learning" / "worker", directory / "learning" / "worker",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (directory / "learning" / "__init__.py").write_text(
        '"""The training worker package (see learning/worker/__init__.py)."""\n', encoding="utf-8")
    shutil.copy2(repository / "requirements-worker.txt", directory / "requirements-worker.txt")
    return directory


def worker_source_digest(*, repository: Path = REPOSITORY) -> str:
    """A digest of the worker code and its requirements: a failed job is retried when this changes."""
    root = Path(repository)
    files = sorted([*(root / "learning" / "worker").rglob("*.py"), root / "requirements-worker.txt"],
                   key=lambda path: path.relative_to(root).as_posix())
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


class TransportError(RuntimeError):
    """A step on the worker failed; ``returncode`` is the command's exit code when one ran."""

    def __init__(self, message: str, *, returncode: int | None = None):
        super().__init__(message)
        self.returncode = returncode

    @property
    def unreachable(self) -> bool:
        """ssh exits 255 when it cannot reach the worker: its state is unknown, not failed."""
        return self.returncode == 255


def _job_id(job_id: str) -> str:
    if not isinstance(job_id, str) or not JOB_ID.match(job_id):
        raise ValueError("a job id is <name>-<UTC stamp>")
    return job_id


class SshTransport:
    def __init__(self, *, host: str, user: str, key: Path, known_hosts: Path, port: int = 22,
                 python: str = "$HOME/janus-worker/.venv/bin/python",
                 runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
                 environment: dict[str, str] | None = None):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{0,252}", host) or not re.fullmatch(r"[a-z_][a-z0-9_-]{0,31}",
                                                                                         user):
            raise ValueError("an SSH worker is a host name and a user name")
        environment = dict(environment or {})
        if any(not ENVIRONMENT_NAME.fullmatch(name) or not ENVIRONMENT_VALUE.fullmatch(value)
               for name, value in environment.items()):
            raise ValueError("the worker's environment is plain NAME=value pairs")
        self.host, self.user, self.port, self.python = host, user, port, python
        self.key, self.known_hosts, self.runner = Path(key), Path(known_hosts), runner
        self.environment = environment

    def _ssh_options(self) -> list[str]:
        return ["-i", str(self.key), "-p", str(self.port), "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
                "-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={self.known_hosts}",
                "-o", "ConnectTimeout=15"]

    def _run(self, argv: list[str], what: str, *, check: bool = True) -> subprocess.CompletedProcess:
        result = self.runner(argv, capture_output=True, text=True, timeout=3600)
        if check and result.returncode != 0:
            raise TransportError(f"{what} failed on {self.host} with exit code {result.returncode}: "
                                 f"{(result.stderr or '').strip()[-300:]}", returncode=result.returncode)
        return result

    def _remote(self, command: str, what: str, *, check: bool = True) -> subprocess.CompletedProcess:
        return self._run(["ssh", *self._ssh_options(), f"{self.user}@{self.host}", command], what, check=check)

    def _rsync(self, source: str, destination: str, what: str, extra: tuple[str, ...] = ()) -> None:
        shell = "ssh " + " ".join(shlex.quote(option) for option in self._ssh_options())
        self._run(["rsync", "-a", "--delete", "--partial", *extra, "-e", shell, source, destination], what)

    def install(self, package: Path) -> None:
        """Copy the worker package (learning/worker + requirements-worker.txt) to the worker."""
        self._rsync(str(package).rstrip("/") + "/", f"{self.user}@{self.host}:{WORKER_HOME_CODE}/",
                    "installing the worker", ("--exclude=.venv/", "--exclude=__pycache__/"))

    def send(self, job_folder: Path, job_id: str) -> None:
        self._rsync(str(job_folder).rstrip("/") + "/", f"{self.user}@{self.host}:jobs/{_job_id(job_id)}/",
                    "sending the job")

    def start(self, job_id: str) -> None:
        """Start the job as a user unit, so it outlives this SSH session and is stopped by systemd.

        The remote login shell expands ``$HOME``: systemd-run would make a relative ``%h`` absolute against
        the session's directory instead. The unit's output goes to the run's log, which ``fetch`` brings back.
        """
        job_id = _job_id(job_id)
        unit = f"janus-train-{job_id}"
        settings = " ".join(f"--setenv={name}={value}" for name, value in sorted(self.environment.items()))
        command = (f'mkdir -p "$HOME/runs/{job_id}" && '
                   f"systemd-run --user --unit={unit} --collect --property=Nice=10 "
                   f'--property=StandardOutput=append:"$HOME/runs/{job_id}/worker.log" '
                   f'--property=StandardError=append:"$HOME/runs/{job_id}/worker.log" '
                   f'--working-directory="$HOME/{WORKER_HOME_CODE}" {settings} '
                   f'{self.python} -m learning.worker.stages "$HOME/jobs/{job_id}/job.json" "$HOME/runs/{job_id}" '
                   f'--stop-file "$HOME/runs/{job_id}.stop"')
        self._remote(" ".join(command.split()), "starting the worker")

    def running(self, job_id: str) -> bool:
        result = self._remote(f"systemctl --user is-active janus-train-{_job_id(job_id)}", "reading the worker",
                              check=False)
        if result.returncode == 255:            # ssh itself failed: the state is unknown, not "finished"
            raise TransportError(f"the worker {self.host} could not be reached", returncode=255)
        return result.stdout.strip() in {"active", "activating"}

    def busy(self) -> bool:
        """Whether any training unit runs on the worker (one job at a time per machine)."""
        result = self._remote("systemctl --user list-units --type=service --state=active,activating --no-legend "
                              "'janus-train-*'", "reading the worker", check=False)
        if result.returncode == 255:
            raise TransportError(f"the worker {self.host} could not be reached", returncode=255)
        return bool(result.stdout.strip())

    def stop(self, job_id: str) -> None:
        self._remote(f"touch runs/{_job_id(job_id)}.stop", "asking the worker to yield")

    def fetch(self, job_id: str, destination: Path) -> Path:
        job_id = _job_id(job_id)
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        self._rsync(f"{self.user}@{self.host}:runs/{job_id}/", str(destination) + "/", "fetching the result",
                    ("--exclude=cache/", "--exclude=checkpoint-*/"))
        return destination

    def clean(self, job_id: str, *, keep_result: bool = False) -> None:
        """Remove the job (and its stop request); the run stays on the worker when ``keep_result``."""
        job_id = _job_id(job_id)
        paths = f"jobs/{job_id} runs/{job_id}.stop" if keep_result else f"jobs/{job_id} runs/{job_id} runs/{job_id}.stop"
        self._remote(f"rm -rf {paths}", "removing the job", check=False)


class LocalTransport:
    """The same worker in a child process of this machine (the local GPU, or a CPU when nothing else is free)."""

    def __init__(self, *, root: Path, python: str, environment: dict[str, str],
                 launcher: Callable[..., Any] = subprocess.Popen, code: Path | None = None):
        self.root, self.python, self.environment, self.launcher = Path(root), python, environment, launcher
        self.code = code or Path(__file__).resolve().parents[1]
        self.processes: dict[str, Any] = {}

    def send(self, job_folder: Path, job_id: str) -> None:
        target = self.root / "jobs" / _job_id(job_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        if Path(job_folder).resolve() != target.resolve():
            if target.exists():
                raise TransportError(f"job {job_id} was already sent")
            os.replace(job_folder, target)

    def start(self, job_id: str):
        job_id = _job_id(job_id)
        runs = self.root / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        # The child inherits the log; this process closes its own handle once the child has started.
        with (runs / f"{job_id}.log").open("ab") as log:
            process = self.launcher([self.python, "-m", "learning.worker.stages",
                                     str(self.root / "jobs" / job_id / "job.json"), str(runs / job_id),
                                     "--stop-file", str(runs / f"{job_id}.stop")],
                                    cwd=str(self.code), env=self.environment, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=log, start_new_session=True)
        self.processes[job_id] = process
        return process

    def running(self, job_id: str) -> bool:
        process = self.processes.get(_job_id(job_id))
        return process is not None and process.poll() is None

    def stop(self, job_id: str) -> None:
        (self.root / "runs" / f"{_job_id(job_id)}.stop").write_text("yield\n")

    def fetch(self, job_id: str, destination: Path | None = None) -> Path:
        return self.root / "runs" / _job_id(job_id)

    def clean(self, job_id: str) -> None:
        import shutil

        shutil.rmtree(self.root / "jobs" / _job_id(job_id), ignore_errors=True)


def collect_candidate(run_directory: Path) -> dict | None:
    """The trained adapter of a finished run, after checking its digest; None when nothing improved."""
    from learning.worker.stages import fingerprint_directory

    run_directory = Path(run_directory)
    result = json.loads((run_directory / "result.json").read_text(encoding="utf-8"))
    if result.get("outcome") != "trained" or not result.get("final_adapter"):
        return None
    adapter = (run_directory / result["final_adapter"]).resolve()
    if run_directory.resolve() not in adapter.parents or not adapter.is_dir():
        raise TransportError("the result names an adapter outside its run")
    if fingerprint_directory(adapter) != result.get("final_adapter_sha256"):
        raise TransportError("the fetched adapter does not match the digest the worker reported")
    return {"job_id": result["job_id"], "adapter_directory": str(adapter), "adapter_sha256": result["final_adapter_sha256"],
            "final_stage": result.get("final_stage"), "stages": result.get("stages", []),
            "baseline": result.get("baseline"), "worker": result.get("worker")}
