"""The authenticated socket: handshake refusal, typed frames and policy closes."""
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosedError, InvalidStatus

from learning.control_plane import canonical, protocol
from learning.control_plane.jobs import JobStore
from learning.control_plane.service import ControlPlaneService
from learning.control_plane.transport import (
    DEFAULT_PATH,
    ControlPlaneServer,
    WorkerTokens,
    open_worker,
    worker_uri,
)

START = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
TOKEN = "a" * 48
OTHER_TOKEN = "b" * 48


@pytest.fixture()
def store(tmp_path: Path) -> JobStore:
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.initialize()
    return store


def enqueue(store: JobStore, job_id: str = "job-1", key: str = "key-1") -> None:
    store.enqueue(job_id=job_id, idempotency_key=key, kind="specialist",
                  payload={"dataset": job_id}, now=START)


def bearer(token: str = TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def start(store: JobStore, tokens: dict[str, str] | None = None):
    service = ControlPlaneService(store, clock=lambda: START)
    server = ControlPlaneServer(service, WorkerTokens(tokens or {"worker-a": TOKEN}))
    await server.start()
    return service, server


async def wait_for(predicate, message: str, *, seconds: float = 5.0) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(message)


def started_for(lease: protocol.LeaseMessage, **overrides) -> protocol.StartedMessage:
    fields = {"sender": "worker-a", "job_id": lease.job_id,
              "lease_token": lease.lease_token, "attempt": lease.attempt}
    fields.update(overrides)
    return protocol.StartedMessage.create(**fields)


def test_a_missing_token_is_refused_with_401(store: JobStore):
    async def scenario():
        _, server = await start(store)
        try:
            uri = worker_uri("127.0.0.1", server.port, "worker-a")
            with pytest.raises(InvalidStatus) as caught:
                async with connect(uri, proxy=None):
                    pass
            assert caught.value.response.status_code == 401
        finally:
            await server.stop()

    asyncio.run(scenario())


def test_a_wrong_token_or_another_worker_is_refused_with_401(store: JobStore):
    async def scenario():
        _, server = await start(store)
        try:
            wrong_token = worker_uri("127.0.0.1", server.port, "worker-a")
            with pytest.raises(InvalidStatus) as caught:
                async with connect(wrong_token, additional_headers=bearer(OTHER_TOKEN),
                                   proxy=None):
                    pass
            assert caught.value.response.status_code == 401

            wrong_worker = (f"ws://127.0.0.1:{server.port}{DEFAULT_PATH}"
                            f"?worker=worker-b")
            with pytest.raises(InvalidStatus) as caught:
                async with connect(wrong_worker, additional_headers=bearer(), proxy=None):
                    pass
            assert caught.value.response.status_code == 401
        finally:
            await server.stop()

    asyncio.run(scenario())


def test_another_path_is_404(store: JobStore):
    async def scenario():
        _, server = await start(store)
        try:
            uri = worker_uri("127.0.0.1", server.port, "worker-a",
                             path="/v1/not-the-worker-path")
            with pytest.raises(InvalidStatus) as caught:
                async with connect(uri, additional_headers=bearer(), proxy=None):
                    pass
            assert caught.value.response.status_code == 404
        finally:
            await server.stop()

    asyncio.run(scenario())


def test_a_valid_worker_receives_a_lease_and_finishes_it(store: JobStore):
    enqueue(store, "job-1", "key-1")
    enqueue(store, "job-2", "key-2")

    async def scenario():
        _, server = await start(store)
        try:
            uri = worker_uri("127.0.0.1", server.port, "worker-a")
            async with open_worker(uri, TOKEN) as worker:
                lease = await asyncio.wait_for(worker.receive(), 5)
                assert isinstance(lease, protocol.LeaseMessage)
                assert lease.job_id == "job-1"
                assert lease.worker_id == "worker-a"
                assert lease.payload == {"dataset": "job-1"}
                assert store.get("job-1").state == "leased"

                await worker.send(started_for(lease))
                await wait_for(lambda: store.get("job-1").state == "running",
                               "the worker's start never reached the store")

                progress = protocol.ProgressMessage.create(
                    sender="worker-a", job_id=lease.job_id,
                    lease_token=lease.lease_token, attempt=lease.attempt,
                    stage="sft", fraction=0.5)
                await worker.send(progress)
                await wait_for(
                    lambda: store.get("job-1").progress is not None,
                    "the worker's progress never reached the store")

                succeeded = protocol.SucceededMessage.create(
                    sender="worker-a", job_id=lease.job_id,
                    lease_token=lease.lease_token, attempt=lease.attempt,
                    result={"adapter": "a"})
                await worker.send(succeeded)
                await wait_for(lambda: store.get("job-1").state == "succeeded",
                               "the worker's success never reached the store")
                assert store.get("job-1").result == {"adapter": "a"}

                next_lease = await asyncio.wait_for(worker.receive(), 5)
                assert isinstance(next_lease, protocol.LeaseMessage)
                assert next_lease.job_id == "job-2"
        finally:
            await server.stop()

    asyncio.run(scenario())


def test_a_malformed_frame_closes_with_policy_code(store: JobStore):
    enqueue(store)

    async def scenario():
        _, server = await start(store)
        try:
            uri = worker_uri("127.0.0.1", server.port, "worker-a")
            async with connect(uri, additional_headers=bearer(), proxy=None) as connection:
                lease = protocol.parse_outbound(await asyncio.wait_for(connection.recv(), 5))
                assert isinstance(lease, protocol.LeaseMessage)

                await connection.send("not a control-plane message")
                with pytest.raises(ConnectionClosedError) as caught:
                    await asyncio.wait_for(connection.recv(), 5)
                assert caught.value.rcvd.code == 1008

            assert store.get("job-1").state == "leased"
        finally:
            await server.stop()

    asyncio.run(scenario())


def test_an_oversized_frame_closes_at_the_transport_limit(store: JobStore):
    enqueue(store)

    async def scenario():
        _, server = await start(store)
        try:
            uri = worker_uri("127.0.0.1", server.port, "worker-a")
            async with connect(uri, additional_headers=bearer(), proxy=None) as connection:
                await asyncio.wait_for(connection.recv(), 5)
                await connection.send(b" " * (protocol.MAX_MESSAGE_BYTES + 1))
                with pytest.raises(ConnectionClosedError) as caught:
                    await asyncio.wait_for(connection.recv(), 5)
                assert caught.value.rcvd.code == 1009
        finally:
            await server.stop()

    asyncio.run(scenario())


def test_a_sender_that_differs_from_the_connection_is_refused(store: JobStore):
    enqueue(store)

    async def scenario():
        _, server = await start(store)
        try:
            uri = worker_uri("127.0.0.1", server.port, "worker-a")
            async with connect(uri, additional_headers=bearer(), proxy=None) as connection:
                lease = protocol.parse_outbound(await asyncio.wait_for(connection.recv(), 5))
                assert isinstance(lease, protocol.LeaseMessage)

                await connection.send(protocol.message_json(started_for(lease, sender="worker-b")))
                with pytest.raises(ConnectionClosedError) as caught:
                    await asyncio.wait_for(connection.recv(), 5)
                assert caught.value.rcvd.code == 1008

            assert store.get("job-1").state == "leased"
        finally:
            await server.stop()

    asyncio.run(scenario())


def test_a_second_connection_for_one_worker_is_refused(store: JobStore):
    enqueue(store)

    async def scenario():
        _, server = await start(store)
        try:
            uri = worker_uri("127.0.0.1", server.port, "worker-a")
            async with open_worker(uri, TOKEN) as first:
                lease = await asyncio.wait_for(first.receive(), 5)
                assert isinstance(lease, protocol.LeaseMessage)

                async with connect(uri, additional_headers=bearer(), proxy=None) as second:
                    with pytest.raises(ConnectionClosedError) as caught:
                        await asyncio.wait_for(second.recv(), 5)
                    assert caught.value.rcvd.code == 1008

                assert store.get("job-1").state == "leased"
        finally:
            await server.stop()

    asyncio.run(scenario())


def test_a_reconnect_adopts_the_held_lease_and_gets_no_second_job(store: JobStore):
    enqueue(store, "job-1", "key-1")

    async def scenario():
        service, server = await start(store)
        try:
            uri = worker_uri("127.0.0.1", server.port, "worker-a")
            async with open_worker(uri, TOKEN) as first:
                lease = await asyncio.wait_for(first.receive(), 5)
                assert isinstance(lease, protocol.LeaseMessage)

            await wait_for(lambda: service.sessions() == (),
                           "the first session never detached")
            enqueue(store, "job-2", "key-2")

            async with open_worker(uri, TOKEN) as second:
                await second.send(started_for(lease))
                await wait_for(lambda: store.get("job-1").state == "running",
                               "the adopted lease never started")
                assert store.get("job-2").state == "queued"

                await second.send(protocol.SucceededMessage.create(
                    sender="worker-a", job_id=lease.job_id,
                    lease_token=lease.lease_token, attempt=lease.attempt,
                    result={"adapter": "a"}))
                await wait_for(lambda: store.get("job-1").state == "succeeded",
                               "the adopted lease never finished")
                next_lease = await asyncio.wait_for(second.receive(), 5)
                assert isinstance(next_lease, protocol.LeaseMessage)
                assert next_lease.job_id == "job-2"
        finally:
            await server.stop()

    asyncio.run(scenario())


def test_progress_before_start_returns_lease_lost(store: JobStore):
    enqueue(store)

    async def scenario():
        _, server = await start(store)
        try:
            uri = worker_uri("127.0.0.1", server.port, "worker-a")
            async with open_worker(uri, TOKEN) as worker:
                lease = await asyncio.wait_for(worker.receive(), 5)
                assert isinstance(lease, protocol.LeaseMessage)

                await worker.send(protocol.ProgressMessage.create(
                    sender="worker-a", job_id=lease.job_id,
                    lease_token=lease.lease_token, attempt=lease.attempt,
                    stage="sft", fraction=0.5))
                lost = await asyncio.wait_for(worker.receive(), 5)

                assert isinstance(lost, protocol.LeaseLostMessage)
                assert lost.reason_code == "job_conflict"
                assert store.get("job-1").state == "leased"
        finally:
            await server.stop()

    asyncio.run(scenario())


def test_tokens_are_validated_and_loaded_from_a_file(tmp_path: Path):
    tokens = WorkerTokens({"worker-a": TOKEN})
    assert tokens.verify("worker-a", TOKEN) is True
    assert tokens.verify("worker-a", OTHER_TOKEN) is False
    assert tokens.verify("worker-b", TOKEN) is False
    assert tokens.verify("worker-a", "not ascii \u00e9") is False
    assert tokens.workers() == ("worker-a",)

    path = tmp_path / "worker-tokens.json"
    path.write_text(json.dumps({"worker-a": TOKEN}), encoding="utf-8")
    assert WorkerTokens.from_file(path).verify("worker-a", TOKEN) is True

    path.write_text("[]", encoding="utf-8")
    with pytest.raises(canonical.ContractError):
        WorkerTokens.from_file(path)
    path.write_text("{", encoding="utf-8")
    with pytest.raises(canonical.ContractError):
        WorkerTokens.from_file(path)

    for invalid in ({}, {"worker-a": "short"}, {"Worker A": TOKEN}):
        with pytest.raises(canonical.ContractError):
            WorkerTokens(invalid)


def test_the_worker_uri_is_the_one_shape_the_server_accepts():
    assert worker_uri("100.64.0.1", 8765, "worker-a") == (
        "ws://100.64.0.1:8765/v1/learning/worker?worker=worker-a")
    assert worker_uri("fd7a::1", 8765, "worker-a", secure=True) == (
        "wss://[fd7a::1]:8765/v1/learning/worker?worker=worker-a")

    for port in (0, 65536, True, "8765"):
        with pytest.raises(canonical.ContractError):
            worker_uri("127.0.0.1", port, "worker-a")
    with pytest.raises(canonical.ContractError):
        worker_uri("127.0.0.1", 8765, "Worker A")
