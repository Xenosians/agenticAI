"""Running a job folder on a worker and bringing its result back, over SSH or on this machine."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from learning.worker_transport import LocalTransport, SshTransport, TransportError, collect_candidate


def write_job(root: Path) -> Path:
    (root / "data").mkdir(parents=True)
    (root / "job.json").write_text("{}")
    return root


class FakeRunner:
    def __init__(self, results=None):
        self.calls = []
        self.results = results or {}

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        for needle, (code, out) in self.results.items():
            if needle in " ".join(argv):
                return subprocess.CompletedProcess(argv, code, out, "")
        return subprocess.CompletedProcess(argv, 0, "", "")


def test_ssh_sends_the_job_runs_the_worker_and_fetches_only_the_result(tmp_path):
    runner = FakeRunner()
    transport = SshTransport(host="janus-trainer", user="janus", key=tmp_path / "id", known_hosts=tmp_path / "kh",
                             runner=runner)
    transport.send(write_job(tmp_path / "access-20261119T000000Z"), "access-20261119T000000Z")
    transport.start("access-20261119T000000Z")
    transport.fetch("access-20261119T000000Z", tmp_path / "back")
    sync, start, fetch = runner.calls[0], runner.calls[1], runner.calls[2]
    assert sync[0] == "rsync" and sync[-1] == "janus@janus-trainer:jobs/access-20261119T000000Z/"
    assert start[0] == "ssh" and "StrictHostKeyChecking=yes" in " ".join(start)
    assert "systemd-run" in start[-1] and "learning.worker.stages" in start[-1]
    assert fetch[0] == "rsync" and "--exclude=cache/" in fetch
    assert fetch[-2] == "janus@janus-trainer:runs/access-20261119T000000Z/"


@pytest.mark.parametrize("job_id", ["../etc", "a b", "x;rm -rf /", ""])
def test_a_job_id_never_becomes_a_shell_fragment(tmp_path, job_id):
    transport = SshTransport(host="h", user="janus", key=tmp_path / "id", known_hosts=tmp_path / "kh",
                             runner=FakeRunner())
    with pytest.raises(ValueError):
        transport.start(job_id)


def test_ssh_failure_is_reported_with_its_exit_code(tmp_path):
    runner = FakeRunner({"systemd-run": (255, "")})
    transport = SshTransport(host="h", user="janus", key=tmp_path / "id", known_hosts=tmp_path / "kh", runner=runner)
    with pytest.raises(TransportError, match="255"):
        transport.start("access-20261119T000000Z")


def test_status_reads_whether_the_worker_is_still_running(tmp_path):
    runner = FakeRunner({"is-active": (0, "active\n")})
    transport = SshTransport(host="h", user="janus", key=tmp_path / "id", known_hosts=tmp_path / "kh", runner=runner)
    assert transport.running("access-20261119T000000Z") is True
    runner.results = {"is-active": (3, "inactive\n")}
    assert transport.running("access-20261119T000000Z") is False


def test_a_local_worker_runs_the_job_in_its_own_process(tmp_path):
    transport = LocalTransport(root=tmp_path / "local", python=sys.executable, environment={"X": "1"},
                               launcher=lambda argv, **kw: FakeProcess(argv, kw))
    job = write_job(tmp_path / "access-20261119T000000Z")
    transport.send(job, "access-20261119T000000Z")
    process = transport.start("access-20261119T000000Z")
    assert process.argv[1:3] == ["-m", "learning.worker.stages"]
    assert process.kwargs["env"]["X"] == "1"
    # The child writes its log through its own handle; this process keeps none open per run.
    assert process.kwargs["stdout"].closed
    assert (tmp_path / "local/jobs/access-20261119T000000Z/job.json").is_file()


class FakeProcess:
    def __init__(self, argv, kwargs):
        self.argv, self.kwargs = argv, kwargs
        self.pid = 4242

    def poll(self):
        return None


def test_a_fetched_result_becomes_a_candidate_only_with_a_matching_adapter_digest(tmp_path):
    from learning.worker.stages import fingerprint_directory

    run = tmp_path / "run"
    (run / "final").mkdir(parents=True)
    (run / "final/adapter_config.json").write_text("{}")
    (run / "final/adapter_model.safetensors").write_bytes(b"weights")
    digest = fingerprint_directory(run / "final")
    (run / "result.json").write_text(json.dumps({"job_id": "access-20261119T000000Z", "outcome": "trained",
                                                 "final_adapter": "final", "final_adapter_sha256": digest}))
    assert collect_candidate(run)["adapter_directory"] == str((run / "final").resolve())
    (run / "final/adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(TransportError, match="digest"):
        collect_candidate(run)


def test_a_result_without_a_trained_adapter_is_no_candidate(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps({"job_id": "x-20261119T000000Z", "outcome": "no_improvement"}))
    assert collect_candidate(tmp_path) is None
