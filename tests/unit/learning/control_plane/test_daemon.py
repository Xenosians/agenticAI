"""The runnable control plane: explicit settings, token file, clean stop."""
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import pytest

from learning.control_plane import canonical, protocol
from learning.control_plane.daemon import (
    Settings,
    build,
    check_token_file,
    main,
    serve,
)
from learning.control_plane.jobs import JobStore
from learning.control_plane.transport import open_worker, worker_uri

START = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
TOKEN = "a" * 48


def token_file(tmp_path: Path, *, mode: int = 0o600) -> Path:
    path = tmp_path / "worker-tokens.json"
    path.write_text(json.dumps({"worker-a": TOKEN}), encoding="utf-8")
    os.chmod(path, mode)
    return path


async def wait_for(predicate, message: str, *, seconds: float = 5.0) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(message)


def test_settings_require_an_explicit_database_and_token_file():
    with pytest.raises(SystemExit):
        Settings.from_args([])
    with pytest.raises(SystemExit):
        Settings.from_args(["--database", "jobs.sqlite3"])

    settings = Settings.from_args([
        "--database", "jobs.sqlite3", "--tokens", "tokens.json",
        "--host", "100.64.0.7", "--port", "0", "--lease-seconds", "60",
    ])
    assert settings.database == Path("jobs.sqlite3")
    assert settings.tokens == Path("tokens.json")
    assert settings.host == "100.64.0.7"
    assert settings.port == 0
    assert settings.lease_seconds == 60.0
    assert settings.retention_seconds > 0
    assert settings.prune_seconds > 0


@pytest.mark.parametrize(
    "values",
    [
        {"host": ""},
        {"host": " 100.64.0.7"},
        {"port": 65536},
        {"port": -1},
        {"port": True},
        {"lease_seconds": 0},
        {"retention_seconds": -1},
        {"prune_seconds": "60"},
    ],
)
def test_invalid_settings_are_refused(tmp_path: Path, values: dict):
    arguments = {"database": tmp_path / "jobs.sqlite3",
                 "tokens": tmp_path / "tokens.json"}
    arguments.update(values)

    with pytest.raises(canonical.ContractError):
        Settings(**arguments)


def test_main_reports_an_invalid_configuration_without_starting(tmp_path: Path, capsys):
    result = main(["--database", str(tmp_path / "jobs.sqlite3"),
                   "--tokens", str(tmp_path / "tokens.json"), "--port", "70000"])

    assert result == 2
    assert "invalid configuration" in capsys.readouterr().err


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions")
def test_a_shared_token_file_is_refused(tmp_path: Path):
    path = token_file(tmp_path, mode=0o644)
    with pytest.raises(canonical.ContractError):
        check_token_file(path)

    os.chmod(path, 0o600)
    check_token_file(path)

    with pytest.raises(canonical.ContractError):
        check_token_file(tmp_path / "missing.json")


def test_build_opens_the_store_and_the_socket_service(tmp_path: Path):
    database = tmp_path / "jobs.sqlite3"
    settings = Settings(database=database, tokens=token_file(tmp_path), port=0)

    service, server = build(settings)

    assert service.store.path == database
    assert database.exists()
    assert service.lease_seconds == settings.lease_seconds
    assert server.host == "127.0.0.1"
    assert server.requested_port == 0
    assert server.tokens.workers() == ("worker-a",)


def test_the_daemon_serves_a_job_and_stops_cleanly(tmp_path: Path):
    database = tmp_path / "jobs.sqlite3"
    store = JobStore(database)
    store.initialize()
    store.enqueue(job_id="job-1", idempotency_key="key-1", kind="specialist",
                  payload={"dataset": "d1"}, now=START)
    settings = Settings(database=database, tokens=token_file(tmp_path), port=0)

    async def scenario():
        stop = asyncio.Event()
        ports: list[int] = []
        task = asyncio.create_task(
            serve(settings, stop, ready=lambda server: ports.append(server.port)))

        await wait_for(lambda: bool(ports), "the daemon never started listening")
        uri = worker_uri("127.0.0.1", ports[0], "worker-a")
        async with open_worker(uri, TOKEN) as worker:
            lease = await asyncio.wait_for(worker.receive(), 5)
            assert isinstance(lease, protocol.LeaseMessage)
            assert lease.job_id == "job-1"

            await worker.send(protocol.StartedMessage.create(
                sender="worker-a", job_id=lease.job_id,
                lease_token=lease.lease_token, attempt=lease.attempt))
            await worker.send(protocol.SucceededMessage.create(
                sender="worker-a", job_id=lease.job_id,
                lease_token=lease.lease_token, attempt=lease.attempt,
                result={"adapter": "a"}))
            await wait_for(lambda: store.get("job-1").state == "succeeded",
                           "the daemon never recorded the success")

        stop.set()
        await asyncio.wait_for(task, 5)

    asyncio.run(scenario())
