"""Kaggle as a remote GPU worker (owner decision 8, 18 Nov 2026): the same portable worker, on Kaggle's free GPU.

The job folder (base weights, data, the worker package) goes up as one private dataset; a private script
kernel with a GPU runs the worker on it and leaves the run in its output. Every call is checked here against
a recorded Kaggle API fake; the token is a Bearer header and never appears in an error, a repr or the job.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from learning.kaggle_worker import (KaggleRefused, KaggleWorker, kernel_script, weekly_gpu_hours_left)
from learning.worker_transport import TransportError

TOKEN = "KGAT_" + "f1xture" * 4          # the shape of a Kaggle API token, not a real one


class FakeKaggle:
    """Kaggle's RPC endpoints (POST https://api.kaggle.com/v1/<service>/<method>) and its upload/download URLs."""

    def __init__(self, *, status_sequence=("RUNNING", "COMPLETE"), dataset_ready_after=1, quota=None, refuse=None,
                 status_forbidden_first=0):
        self.calls, self.uploads = [], {}
        self.status_forbidden_first = status_forbidden_first   # Kaggle answers 403 just after a create
        self.status_sequence, self.dataset_polls = list(status_sequence), 0
        self.dataset_ready_after, self.quota, self.refuse = dataset_ready_after, quota, refuse or {}
        self.kernel_text = None

    def post(self, url, *, json=None, headers=None, timeout=None):
        assert headers["Authorization"] == f"Bearer {TOKEN}"
        method = url.rsplit("/", 1)[-1]
        self.calls.append((method, json))
        if method in self.refuse:
            return Response(self.refuse[method], {"message": f"refused, echoing {TOKEN}"})
        if method == "IntrospectToken":
            return Response(200, {"active": True, "username": "xenosians"})
        if method == "GetAcceleratorQuotaStatistics":
            return Response(200, self.quota or {"gpuQuota": {"timeUsed": "3600s", "totalTimeAllowed": "108000s"}})
        if method == "StartBlobUpload":
            name = json["name"]
            return Response(200, {"token": f"blob-{name}", "createUrl": f"https://upload.example/{name}"})
        if method == "CreateDataset":
            return Response(200, {"ref": f"xenosians/{json['slug']}", "status": "ok"})
        if method == "GetDatasetStatus":
            self.dataset_polls += 1
            if self.dataset_polls <= self.status_forbidden_first:
                return Response(403, {"message": "Permission 'datasets.get' was denied"})
            return Response(200, {"status": "READY" if self.dataset_polls >= self.dataset_ready_after
                                  else "BLOBS_RECEIVED"})
        if method == "SaveKernel":
            self.kernel_text = json["text"]
            return Response(200, {"ref": f"/code/xenosians/{json['slug']}", "versionNumber": 1})
        if method == "GetKernelSessionStatus":
            return Response(200, {"status": self.status_sequence.pop(0) if self.status_sequence else "COMPLETE"})
        if method == "ListKernelSessionOutput":
            return Response(200, {"files": [{"fileName": "run.zip", "url": "https://output.example/run.zip"}],
                                  "log": "worker log"})
        if method in ("DeleteDataset", "DeleteKernel"):
            return Response(200, {})
        raise AssertionError(f"unexpected Kaggle call {method}")

    def put(self, url, *, content=None, headers=None, timeout=None):
        assert "Authorization" not in (headers or {})          # signed upload URLs carry no token
        self.uploads[url] = content if isinstance(content, bytes) else b"".join(content)
        return Response(200, {})

    def get(self, url, *, timeout=None, headers=None):
        assert "Authorization" not in (headers or {})
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("result.json", json.dumps({"outcome": "trained", "final_adapter": "final"}))
            bundle.writestr("final/adapter_model.safetensors", b"weights")
        return Response(200, {}, content=archive.getvalue())


class Response:
    def __init__(self, status_code, body, *, content=b""):
        self.status_code, self._body, self.content = status_code, body, content
        self.headers = {"content-type": "application/json"}

    def json(self):
        return self._body


def make_job(tmp_path: Path) -> Path:
    job = tmp_path / "account-20261119T000000Z"
    (job / "base").mkdir(parents=True)
    (job / "base" / "model.safetensors").write_bytes(b"w" * 10)
    (job / "data").mkdir()
    (job / "data" / "train.jsonl").write_text("{}\n", encoding="utf-8")
    (job / "job.json").write_text(json.dumps({"job_id": "account-20261119T000000Z", "device": "cuda"}),
                                  encoding="utf-8")
    return job


def worker(fake, tmp_path, **kwargs) -> KaggleWorker:
    package = tmp_path / "package"
    (package / "learning" / "worker").mkdir(parents=True, exist_ok=True)
    (package / "learning" / "worker" / "stages.py").write_text("# worker\n", encoding="utf-8")
    (package / "requirements-worker.txt").write_text("peft==0.20.0\n", encoding="utf-8")
    return KaggleWorker(token=TOKEN, username="xenosians", http=fake, sleep=lambda seconds: None,
                        package=package, **kwargs)


def test_a_job_goes_up_as_one_private_dataset_and_runs_in_a_private_gpu_kernel(tmp_path):
    fake = FakeKaggle(dataset_ready_after=2)
    kaggle = worker(fake, tmp_path)
    kaggle.send(make_job(tmp_path), "account-20261119T000000Z")
    kaggle.start("account-20261119T000000Z")
    methods = [method for method, _ in fake.calls]
    assert methods == ["StartBlobUpload", "CreateDataset", "GetDatasetStatus", "GetDatasetStatus", "SaveKernel"]
    upload = fake.calls[0][1]
    assert upload["type"] == "DATASET" and upload["name"] == "job.zip" and upload["contentLength"] > 0
    with zipfile.ZipFile(io.BytesIO(fake.uploads["https://upload.example/job.zip"])) as bundle:
        names = set(bundle.namelist())
    assert {"job/job.json", "job/base/model.safetensors", "job/data/train.jsonl",
            "package/learning/worker/stages.py", "package/requirements-worker.txt"} <= names
    dataset = fake.calls[1][1]
    assert dataset["isPrivate"] is True and dataset["slug"] == "janus-job-account-20261119t000000z"
    assert dataset["files"] == [{"token": "blob-job.zip"}]
    kernel = fake.calls[-1][1]
    assert kernel["isPrivate"] is True and kernel["enableGpu"] is True and kernel["kernelType"] == "script"
    assert kernel["datasetDataSources"] == ["xenosians/janus-job-account-20261119t000000z"]
    assert kernel["language"] == "python" and kernel["slug"] == "xenosians/janus-train-account-20261119t000000z"
    assert TOKEN not in json.dumps(fake.calls) and TOKEN not in fake.kernel_text


def test_the_kernel_runs_the_shared_worker_and_zips_the_run_into_its_output():
    script = kernel_script("account-20261119T000000Z", dataset="xenosians/janus-job-account-20261119t000000z")
    assert "learning.worker.stages" in script and "/kaggle/working/run.zip" in script
    assert "'janus-job-account-20261119t000000z'" in script and "job.zip" in script
    assert "requirements-worker.txt" in script and "!= 'torch'" in script   # Kaggle's torch fits its driver
    compile(script, "kernel.py", "exec")                  # a valid Python script


def test_the_kernel_finds_the_job_whether_kaggle_kept_the_zip_or_unpacked_it(tmp_path, monkeypatch):
    """Run the kernel's own text against both input layouts, with pip and the worker stubbed.

    The script never changes the working directory (the worker runs with cwd=); the suite's relative
    paths depend on that, so it is asserted here.
    """
    import os
    import subprocess as real_subprocess

    before = os.getcwd()

    script = kernel_script("account-20261119T000000Z", dataset="xenosians/janus-job-account-20261119t000000z")
    for layout in ("zipped", "unpacked"):
        base = tmp_path / layout
        source = base / "input" / "janus-job-account-20261119t000000z"
        source.mkdir(parents=True)
        files = {"job/job.json": "{}", "package/requirements-worker.txt": "torch==2.13.0\npeft==0.20.0\n",
                 "package/learning/worker/stages.py": "#"}
        if layout == "zipped":
            with zipfile.ZipFile(source / "job.zip", "w") as bundle:
                for name, text in files.items():
                    bundle.writestr(name, text)
        else:
            for name, text in files.items():
                (source / name).parent.mkdir(parents=True, exist_ok=True)
                (source / name).write_text(text)
        commands = []

        def fake_run(argv, **kwargs):
            commands.append(argv)
            if "learning.worker.stages" in argv:
                Path(argv[-1]).mkdir(parents=True)
                (Path(argv[-1]) / "result.json").write_text("{}")
            return real_subprocess.CompletedProcess(argv, 0)

        monkeypatch.setattr(real_subprocess, "run", fake_run)
        text = (script.replace("/kaggle/input", str(base / "input")).replace("/kaggle/temp", str(base / "temp"))
                .replace("/kaggle/working", str(base)))
        exec(compile(text, "kernel.py", "exec"), {"__name__": "__main__"})
        assert (base / "temp/janus/requirements-kaggle.txt").read_text() == "peft==0.20.0\n"
        assert any("learning.worker.stages" in argv for argv in commands), layout
        with zipfile.ZipFile(base / "run.zip") as bundle:
            assert bundle.namelist() == ["result.json"]
        assert os.getcwd() == before


def test_running_fetch_and_clean(tmp_path):
    fake = FakeKaggle(status_sequence=["QUEUED", "RUNNING", "COMPLETE"])
    kaggle = worker(fake, tmp_path)
    assert kaggle.running("account-20261119T000000Z") is True
    assert kaggle.running("account-20261119T000000Z") is True
    assert kaggle.running("account-20261119T000000Z") is False
    run = kaggle.fetch("account-20261119T000000Z", tmp_path / "back")
    assert json.loads((run / "result.json").read_text())["outcome"] == "trained"
    assert (run / "final" / "adapter_model.safetensors").read_bytes() == b"weights"
    assert (run / "kaggle.log").read_text() == "worker log"
    kaggle.clean("account-20261119T000000Z", keep_result=True)
    assert [method for method, _ in fake.calls][-2:] == ["DeleteKernel", "DeleteDataset"]  # nothing stays on Kaggle


def test_a_failed_kernel_still_returns_its_run_and_an_unreadable_one_raises(tmp_path):
    fake = FakeKaggle(status_sequence=["ERROR"])
    kaggle = worker(fake, tmp_path)
    assert kaggle.running("account-20261119T000000Z") is False
    fake.refuse["ListKernelSessionOutput"] = 404
    with pytest.raises(TransportError) as missing:
        kaggle.fetch("account-20261119T000000Z", tmp_path / "back")
    assert missing.value.returncode == 23              # like rsync: the run is not there (counted as lost)


def test_refusals_never_echo_the_token(tmp_path):
    fake = FakeKaggle(refuse={"CreateDataset": 403})
    kaggle = worker(fake, tmp_path)
    with pytest.raises(KaggleRefused) as refused:
        kaggle.send(make_job(tmp_path), "account-20261119T000000Z")
    assert "403" in str(refused.value) and TOKEN not in str(refused.value)
    assert TOKEN not in repr(kaggle)


def test_unreachable_kaggle_is_unknown_not_lost(tmp_path):
    class Offline(FakeKaggle):
        def post(self, url, **kwargs):
            import httpx

            raise httpx.ConnectError("no route")

    kaggle = worker(Offline(), tmp_path)
    with pytest.raises(TransportError) as offline:
        kaggle.running("account-20261119T000000Z")
    assert offline.value.unreachable


def test_busy_is_any_janus_kernel_still_running(tmp_path):
    fake = FakeKaggle(status_sequence=["RUNNING"])
    kaggle = worker(fake, tmp_path)
    assert kaggle.busy() is False                      # nothing started by this worker
    kaggle.started.add("account-20261119T000000Z")
    assert kaggle.busy() is True


def test_weekly_gpu_hours_left_come_from_the_quota(tmp_path):
    fake = FakeKaggle(quota={"gpuQuota": {"timeUsed": "36000s", "totalTimeAllowed": "108000s"}})
    assert weekly_gpu_hours_left(worker(fake, tmp_path)) == 20.0
    fake = FakeKaggle(refuse={"GetAcceleratorQuotaStatistics": 500})
    assert weekly_gpu_hours_left(worker(fake, tmp_path)) is None


@pytest.mark.parametrize("job_id", ["../x-20261119T000000Z", "a b-20261119T000000Z", "account"])
def test_job_ids_are_checked_before_anything_is_named_after_them(tmp_path, job_id):
    with pytest.raises(ValueError):
        worker(FakeKaggle(), tmp_path).running(job_id)


def test_the_worker_takes_its_user_name_from_the_token(tmp_path):
    from learning.kaggle_worker import connect

    fake = FakeKaggle()
    kaggle = connect(token=TOKEN, package=tmp_path, http=fake)
    assert kaggle.username == "xenosians" and fake.calls[0][0] == "IntrospectToken"
    fake = FakeKaggle(refuse={"IntrospectToken": 401})
    with pytest.raises(KaggleRefused):
        connect(token=TOKEN, package=tmp_path, http=fake)


def test_a_status_read_refused_just_after_the_create_is_read_again(tmp_path):
    """Live: GetDatasetStatus answered 403 seconds after CreateDataset succeeded; the dataset did exist."""
    fake = FakeKaggle(status_forbidden_first=2, dataset_ready_after=3)
    worker(fake, tmp_path).send(make_job(tmp_path), "account-20261119T000000Z")
    assert [method for method, _ in fake.calls].count("GetDatasetStatus") == 3


def test_a_send_that_fails_after_the_create_removes_the_dataset(tmp_path):
    fake = FakeKaggle(status_forbidden_first=99)
    kaggle = worker(fake, tmp_path)
    with pytest.raises(KaggleRefused):
        kaggle.send(make_job(tmp_path), "account-20261119T000000Z")
    assert fake.calls[-1][0] == "DeleteDataset"           # nothing is left on the account
    assert not list(tmp_path.glob("*.job.zip"))           # nor the archive here
