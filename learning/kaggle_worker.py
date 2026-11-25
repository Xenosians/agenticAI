"""Kaggle's free GPU as a remote worker (owner decision 8, 18 Nov 2026), running the one portable worker.

A job goes up as ONE private dataset (``job.zip``: the job folder and the worker package); a private script
kernel with a GPU unpacks it, installs the pinned worker requirements, runs ``learning.worker.stages`` and
zips the run into its output. Janus fetches the run, removes the kernel and the dataset (nothing stays on
Kaggle), and judges the candidate with its own held-out gate, as for every other worker.

Kaggle's public RPC endpoints are used directly (``POST https://api.kaggle.com/v1/<service>/<method>``, the
calls the official ``kaggle`` CLI makes). The API token is a Bearer header only: never in a URL, the job,
the kernel's text, a log line, an error or a repr. Signed upload and download URLs take no token.
"""
from __future__ import annotations

import io
import json
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Callable

import httpx

from learning.worker_transport import JOB_ID, TransportError

API = "https://api.kaggle.com/v1"
KAGGLE_USER = re.compile(r"^[a-z0-9][a-z0-9-]{2,49}$")
KAGGLE_TOKEN = re.compile(r"^KGAT_[A-Za-z0-9]{20,64}$")
FINISHED = {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"}
DATASET_READY_POLLS = 60
RSYNC_MISSING = 23            # the same signal as rsync's "files not there", so a vanished run counts as lost


class KaggleRefused(RuntimeError):
    """Kaggle answered with an error; the message names the call and the status, never the token."""

    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


def _slug(prefix: str, job_id: str) -> str:
    if not isinstance(job_id, str) or not JOB_ID.match(job_id):
        raise ValueError("a job id is <name>-<UTC stamp>")
    slug = f"{prefix}-{job_id.lower()}"
    if not 6 <= len(slug) <= 50:
        raise ValueError("a Kaggle slug is 6 to 50 characters")
    return slug


def kernel_script(job_id: str, *, dataset: str) -> str:
    """The kernel's whole text: find the job, install the worker's requirements, train, zip the run.

    Kaggle keeps an uploaded ``job.zip`` or unpacks it, so the script takes whichever the input holds. Kaggle's
    own torch stays: it is built for the GPU driver there (a pinned PyPI torch may need a newer driver).
    """
    _slug("janus-job", job_id)
    slug = dataset.split("/", 1)[1]
    lines = [
        "import os, re, shutil, subprocess, sys, zipfile",
        "work = '/kaggle/temp/janus'",
        "shutil.rmtree(work, ignore_errors=True)",
        "os.makedirs(work)",
        "for root, dirs, files in os.walk('/kaggle/input'):",
        f"    if {slug!r} not in root:",
        "        continue",
        "    if 'job.zip' in files:",
        "        zipfile.ZipFile(os.path.join(root, 'job.zip')).extractall(work)",
        "    name = os.path.basename(root)",
        "    wanted = (name == 'job' and 'job.json' in files) or (name == 'package' and 'requirements-worker.txt' in files)",
        "    if wanted and not os.path.exists(os.path.join(work, name)):",
        "        shutil.copytree(root, os.path.join(work, name))",
        "assert os.path.isfile(work + '/job/job.json'), 'the job is not in the input'",
        "lines = open(work + '/package/requirements-worker.txt').read().splitlines()",
        "keep = [line for line in lines if line.strip() and not line.startswith('#')",
        "        and re.split(r'[=<>!~; ]', line.strip(), maxsplit=1)[0] != 'torch']",
        "open(work + '/requirements-kaggle.txt', 'w').write('\\n'.join(keep) + '\\n')",
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check',",
        "                '-r', work + '/requirements-kaggle.txt'], check=True)",
        "# Online only for what the job itself names: a base pinned to a public commit (the host admitted it).",
        "environment = dict(os.environ, HF_HUB_OFFLINE='0', TRANSFORMERS_OFFLINE='0', HF_HUB_DISABLE_TELEMETRY='1',",
        "                   PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false', PYTHONPATH=work + '/package')",
        "code = subprocess.run([sys.executable, '-m', 'learning.worker.stages', work + '/job/job.json',",
        "                       work + '/run'], cwd=work + '/package', env=environment).returncode",
        "with zipfile.ZipFile('/kaggle/working/run.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:",
        "    for root, _dirs, files in os.walk(work + '/run'):",
        "        for name in files:",
        "            path = os.path.join(root, name)",
        "            relative = os.path.relpath(path, work + '/run')",
        "            if not relative.startswith(('cache', 'checkpoint-')):",
        "                bundle.write(path, relative)",
        "print('janus worker exit', code)",
    ]
    return "\n".join(lines) + "\n"


def _zip_job(job_folder: Path, package: Path, archive: Path) -> Path:
    """One file on disk (a job holds the base weights: never the whole job in memory)."""
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as bundle:     # weights do not compress
        for prefix, root in (("job", Path(job_folder)), ("package", Path(package))):
            for path in sorted(root.rglob("*")):
                relative = path.relative_to(root)
                if path.is_file() and not any(part.startswith(".") or part == "__pycache__"
                                              for part in relative.parts):
                    bundle.write(path, f"{prefix}/{relative.as_posix()}")
    return archive


def _hours(duration) -> float | None:
    """Kaggle durations are protobuf JSON: '<seconds>s'."""
    if not isinstance(duration, str) or not duration.endswith("s"):
        return None
    try:
        return float(duration[:-1]) / 3600
    except ValueError:
        return None


class KaggleWorker:
    """SshTransport's interface (send, start, running, busy, stop, fetch, clean) on Kaggle."""

    def __init__(self, *, token: str, username: str, package: Path | None = None, http=None,
                 sleep: Callable[[float], None] = time.sleep):
        if not KAGGLE_TOKEN.match(token or "") or not KAGGLE_USER.match(username or ""):
            raise ValueError("a Kaggle worker is an API token and a user name")
        self._token, self.username = token, username
        self.package = Path(package) if package is not None else None
        if http is None:
            from services.lua_adapters.http_clients import client as lua_http_client

            http = lua_http_client(timeout=httpx.Timeout(120.0), follow_redirects=False, trust_env=False)
        self.http, self.sleep = http, sleep
        self.started: set[str] = set()

    def __repr__(self) -> str:
        return f"KaggleWorker(username={self.username!r})"

    # -- the API --------------------------------------------------------------------------------------------

    def _call(self, service: str, method: str, body: dict) -> dict:
        try:
            answer = self.http.post(f"{API}/{service}/{method}", json=body, timeout=120.0,
                                    headers={"Authorization": f"Bearer {self._token}",
                                             "Content-Type": "application/json", "Accept": "application/json"})
        except httpx.TransportError as error:
            raise TransportError(f"Kaggle could not be reached ({type(error).__name__})", returncode=255) from None
        if answer.status_code != 200:
            raise KaggleRefused(f"Kaggle refused {method} with HTTP {answer.status_code}", status=answer.status_code)
        try:
            data = answer.json()
        except ValueError:
            raise KaggleRefused(f"Kaggle answered {method} without JSON") from None
        if isinstance(data, dict) and data.get("error"):
            raise KaggleRefused(f"Kaggle refused {method}: {str(data['error'])[:200]}".replace(self._token, "<token>"))
        return data if isinstance(data, dict) else {}

    def quota(self) -> dict:
        return self._call("kernels.KernelsApiService", "GetAcceleratorQuotaStatistics", {})

    # -- the transport --------------------------------------------------------------------------------------

    def install(self, package: Path) -> None:
        """The worker package travels inside each job's dataset."""
        self.package = Path(package)

    def send(self, job_folder: Path, job_id: str) -> None:
        if self.package is None or not (self.package / "learning" / "worker").is_dir():
            # The trainer installs the package it built for this dispatch first; an uninstalled worker
            # must fail here, not upload a job that cannot run.
            raise TransportError("the worker package was not installed on this Kaggle worker")
        slug = _slug("janus-job", job_id)
        archive = _zip_job(Path(job_folder), self.package, Path(job_folder).parent / f"{job_id}.job.zip")
        try:
            start = self._call("blobs.BlobApiService", "StartBlobUpload",
                               {"type": "DATASET", "name": "job.zip", "contentType": "application/zip",
                                "contentLength": archive.stat().st_size,
                                "lastModifiedEpochSeconds": int(time.time())})
            with archive.open("rb") as stream:      # streamed with its length, as the official client does
                uploaded = self.http.put(str(start["createUrl"]), content=stream, timeout=3600.0,
                                         headers={"Content-Type": "application/zip"})
        except (KeyError, httpx.TransportError) as error:
            raise TransportError(f"uploading the job to Kaggle failed ({type(error).__name__})",
                                 returncode=255) from None
        finally:
            archive.unlink(missing_ok=True)
        if uploaded.status_code not in (200, 201):
            raise KaggleRefused(f"Kaggle refused the job upload with HTTP {uploaded.status_code}")
        created = self._call("datasets.DatasetApiService", "CreateDataset", {
            "ownerSlug": self.username, "slug": slug, "title": slug, "licenseName": "unknown", "isPrivate": True,
            "files": [{"token": start["token"]}], "description": "Janus training job (removed after the run)."})
        if created.get("error"):
            raise KaggleRefused("Kaggle refused the job dataset")
        try:
            self._wait_for_dataset(slug)
        except BaseException:
            self._remove("datasets.DatasetApiService", "DeleteDataset",
                         {"ownerSlug": self.username, "datasetSlug": slug})
            raise

    def _wait_for_dataset(self, slug: str) -> None:
        for _ in range(DATASET_READY_POLLS):
            self.sleep(10.0)
            try:
                status = self._call("datasets.DatasetApiService", "GetDatasetStatus",
                                    {"ownerSlug": self.username, "datasetSlug": slug}).get("status")
            except KaggleRefused:
                continue            # seen live: 403 for a few seconds after the create, then READY
            if status == "READY":
                return
            if status in ("FAILED", "DELETED"):
                raise KaggleRefused(f"Kaggle could not process the job dataset ({status})")
        raise KaggleRefused("the job dataset was not ready on Kaggle in time")

    def _remove(self, service: str, method: str, body: dict) -> None:
        try:
            self._call(service, method, body)
        except (KaggleRefused, TransportError):
            pass                    # already gone, or removed on the next clean

    def start(self, job_id: str) -> None:
        dataset = f"{self.username}/{_slug('janus-job', job_id)}"
        kernel = _slug("janus-train", job_id)
        saved = self._call("kernels.KernelsApiService", "SaveKernel", {
            "slug": f"{self.username}/{kernel}", "newTitle": kernel, "text": kernel_script(job_id, dataset=dataset),
            "language": "python", "kernelType": "script", "isPrivate": True, "enableGpu": True,
            "enableTpu": False, "enableInternet": True, "datasetDataSources": [dataset]})
        if saved.get("error") or saved.get("invalidDatasetSources"):
            raise KaggleRefused("Kaggle refused the training kernel or its job dataset")
        self.started.add(job_id)

    def running(self, job_id: str) -> bool:
        try:
            status = self._call("kernels.KernelsApiService", "GetKernelSessionStatus",
                                {"userName": self.username, "kernelSlug": _slug("janus-train", job_id)}).get("status")
        except KaggleRefused as refused:
            if refused.status == 404:
                return False         # no such kernel: fetching it finds the run lost
            raise
        return status not in FINISHED

    def busy(self) -> bool:
        """Whether a kernel this worker started still runs (Kaggle runs a limited number at once)."""
        return any(self.running(job_id) for job_id in sorted(self.started))

    def stop(self, job_id: str) -> None:
        """Kaggle has no stop file: deleting the kernel ends its session."""
        self._call("kernels.KernelsApiService", "DeleteKernel",
                   {"userName": self.username, "kernelSlug": _slug("janus-train", job_id)})
        self.started.discard(job_id)

    def fetch(self, job_id: str, destination: Path) -> Path:
        try:
            listing = self._call("kernels.KernelsApiService", "ListKernelSessionOutput",
                                 {"userName": self.username, "kernelSlug": _slug("janus-train", job_id)})
        except KaggleRefused:
            raise TransportError("the run is not on Kaggle", returncode=RSYNC_MISSING) from None
        files = {item.get("fileName"): item.get("url") for item in listing.get("files") or []
                 if isinstance(item, dict)}
        destination = Path(destination)
        shutil.rmtree(destination, ignore_errors=True)
        destination.mkdir(parents=True)
        if listing.get("log"):
            (destination / "kaggle.log").write_text(str(listing["log"]), encoding="utf-8")
        url = files.get("run.zip")
        if not url:
            raise TransportError("the Kaggle run left no output", returncode=RSYNC_MISSING)
        try:
            answer = self.http.get(str(url), timeout=3600.0)
        except httpx.TransportError as error:
            raise TransportError(f"downloading the run failed ({type(error).__name__})", returncode=255) from None
        if answer.status_code != 200:
            raise TransportError(f"downloading the run failed with HTTP {answer.status_code}",
                                 returncode=RSYNC_MISSING)
        root = destination.resolve()
        with zipfile.ZipFile(io.BytesIO(answer.content)) as bundle:
            for member in bundle.infolist():
                target = (destination / member.filename).resolve()
                if root not in target.parents:          # the names come from the kernel: not trusted
                    raise TransportError("the Kaggle run holds a path outside its folder")
                bundle.extract(member, destination)
        return destination

    def clean(self, job_id: str, *, keep_result: bool = False) -> None:
        """Remove the kernel and the job dataset; the fetched run is kept on this machine, not on Kaggle."""
        self._remove("kernels.KernelsApiService", "DeleteKernel",
                     {"userName": self.username, "kernelSlug": _slug("janus-train", job_id)})
        self._remove("datasets.DatasetApiService", "DeleteDataset",
                     {"ownerSlug": self.username, "datasetSlug": _slug("janus-job", job_id)})
        self.started.discard(job_id)


def connect(*, token: str, package: Path | None = None, http=None,
            sleep: Callable[[float], None] = time.sleep) -> KaggleWorker:
    """A worker for the account the token belongs to (Kaggle names it; nothing is assumed)."""
    probe = KaggleWorker(token=token, username="janus-probe", package=package, http=http, sleep=sleep)
    answer = probe._call("security.OAuthService", "IntrospectToken", {"token": token})
    username = str(answer.get("username") or "").lower()
    if not answer.get("active") or not KAGGLE_USER.match(username):
        raise KaggleRefused("Kaggle does not accept this token")
    return KaggleWorker(token=token, username=username, package=package, http=probe.http, sleep=sleep)


def weekly_gpu_hours_left(worker: KaggleWorker) -> float | None:
    """This week's GPU hours left, or None when Kaggle does not say."""
    try:
        gpu = worker.quota().get("gpuQuota") or {}
    except (KaggleRefused, TransportError):
        return None
    allowed, used = _hours(gpu.get("totalTimeAllowed")), _hours(gpu.get("timeUsed"))
    if allowed is None or used is None:
        return None
    return round(max(allowed - used, 0.0), 2)
