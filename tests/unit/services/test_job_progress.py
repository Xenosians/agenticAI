import asyncio
import builtins
import json
from types import SimpleNamespace
from functools import wraps

import pytest

from services.job_progress import JobProgress, job_progress, progress_url, renew_lease, report_progress


def run_async(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))
    return run


@pytest.mark.parametrize("url", ["https://user:secret@example.com", "ftp://example.com", "https://example.com?token=secret"])
def test_rejects_credentials_and_query_in_origin(url):
    with pytest.raises(ValueError):
        progress_url(url)


def test_tls_origin_and_bounded_content_free_queue():
    publisher = JobProgress("https://example.com/prefix", "private-token", "a" * 32, 2)
    assert publisher.url == "wss://example.com/prefix/ai_socket/websocket?vsn=2.0.0"
    for stage in ["routing", "planning", "synthesizing"]:
        publisher.publish(stage)
    assert publisher.queue.qsize() == 1
    assert publisher.queue.get_nowait() == "synthesizing"
    with pytest.raises(ValueError):
        publisher.publish("arbitrary model text or a password")


@run_async
async def test_real_websocket_protocol_authentication_and_progress():
    from websockets.asyncio.server import serve
    observed = asyncio.Queue()
    released = asyncio.Event()

    async def handler(socket):
        await observed.put({"path": socket.request.path,
                            "token": socket.request.headers.get("X-Internal-Token")})
        async for raw in socket:
            frame = json.loads(raw)
            await observed.put(frame)
            await socket.send(json.dumps([frame[0], frame[1], frame[2], "phx_reply", {"status": "ok", "response": {}}]))
        released.set()

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        publisher = JobProgress(f"http://127.0.0.1:{port}", "test-private", "a" * 32, 2)
        task = asyncio.create_task(publisher.run())
        try:
            async with asyncio.timeout(5):
                headers = await observed.get()
                assert headers == {"path": "/ai_socket/websocket?vsn=2.0.0", "token": "test-private"}
                join = await observed.get()
                assert join == ["1", "1", "ai_jobs:" + "a" * 32, "phx_join", {"attempt": 2}]
                publisher.publish("planning")
                frame = await observed.get()
                assert frame[2:] == ["ai_jobs:" + "a" * 32, "progress", {"stage": "planning"}]
                assert "test-private" not in json.dumps(frame)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        async with asyncio.timeout(5):
            await released.wait()


@run_async
async def test_progress_scope_isolated_and_unavailable_server_does_not_fail_job(monkeypatch):
    calls = []

    async def unavailable(self):
        calls.append(self)
        raise ConnectionError("offline")

    monkeypatch.setattr(JobProgress, "run", unavailable)
    settings = SimpleNamespace(require_phoenix_base_url=lambda: "http://127.0.0.1:4000",
                               require_internal_job_token=lambda: "test-private")
    async with job_progress(settings, "b" * 32, 1) as outer:
        report_progress("planning")
        async with job_progress(settings, "c" * 32, 1) as inner:
            report_progress("executing")
            assert inner.latest == "executing" and outer.latest == "planning"
        report_progress("synthesizing")
        assert outer.latest == "synthesizing"
        # Yield by awaiting a task, without timing assumptions.
        await asyncio.create_task(asyncio.sleep(0))
    report_progress("routing")
    assert outer.latest == "synthesizing"


@run_async
async def test_scope_preserves_caller_cancellation():
    settings = SimpleNamespace(job_websocket_enabled=False)
    with pytest.raises(asyncio.CancelledError):
        async with job_progress(settings, "b" * 32, 1):
            raise asyncio.CancelledError()


@run_async
async def test_real_idle_heartbeats_are_drained_and_reconnect_rejoins_before_progress():
    from websockets.asyncio.server import serve
    connections = 0
    heartbeats = []
    joined_again = asyncio.Event()
    progress_received = asyncio.Event()
    disconnected = asyncio.Event()

    async def handler(socket):
        nonlocal connections
        connections += 1
        connection = connections
        try:
            async for raw in socket:
                frame = json.loads(raw)
                if frame[3] == "phx_join":
                    assert frame[2] == "ai_jobs:" + "d" * 32
                    assert frame[4] == {"attempt": 3}
                    if connection == 2:
                        joined_again.set()
                elif frame[3] == "heartbeat":
                    assert frame[0] is None and frame[2] == "phoenix" and frame[4] == {}
                    heartbeats.append((connection, frame[1]))
                elif frame[3] == "progress":
                    assert connection == 2
                    assert frame[4] == {"stage": "executing"}
                    progress_received.set()
                else:
                    pytest.fail("The progress observer sent an execution message")
                reject_heartbeat = connection == 1 and frame[3] == "heartbeat" and len(heartbeats) == 6
                await socket.send(json.dumps([frame[0], frame[1], frame[2], "phx_reply", {
                    "status": "error" if reject_heartbeat else "ok", "response": {}}]))
                # Keep transport open: only reading the sixth heartbeat's
                # rejection can trigger timely recovery. An observer which
                # merely sends heartbeats would miss it and fill max_queue.
        finally:
            if connection == 2:
                disconnected.set()

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        publisher = JobProgress(f"http://127.0.0.1:{port}", "fixture-private", "d" * 32, 3,
                                heartbeat_interval=0.02)
        task = asyncio.create_task(publisher.run())
        try:
            async with asyncio.timeout(5):
                await joined_again.wait()
                publisher.publish("executing")
                await progress_received.wait()
            assert connections == 2
            assert len([item for item in heartbeats if item[0] == 1]) == 6
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        async with asyncio.timeout(5):
            await disconnected.wait()


@run_async
async def test_missing_optional_websocket_dependency_finishes_observer_without_task_error(monkeypatch):
    original_import = builtins.__import__

    def without_websockets(name, *args, **kwargs):
        if name == "websockets.asyncio.client":
            raise ImportError("fixture missing optional dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_websockets)
    publisher = JobProgress("http://127.0.0.1:4000", "fixture-private", "e" * 32, 1)
    task = asyncio.create_task(publisher.run())
    await task
    assert task.exception() is None


@run_async
async def test_lease_renewal_is_a_request_and_reply_on_the_attempts_joined_socket():
    from websockets.asyncio.server import serve
    job_id = "f" * 32
    lease = {"job_id": job_id, "attempt": 4, "status": "processing", "lease_expires_at": "2026-10-17T12:00:00Z"}
    events = []

    async def handler(socket):
        async for raw in socket:
            frame = json.loads(raw)
            events.append((frame[3], frame[4]))
            response = lease if frame[3] == "renew_lease" else {}
            await socket.send(json.dumps([frame[0], frame[1], frame[2], "phx_reply",
                                          {"status": "ok", "response": response}]))

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        settings = SimpleNamespace(require_phoenix_base_url=lambda: f"http://127.0.0.1:{port}",
                                   require_internal_job_token=lambda: "fixture-private")
        assert await renew_lease(job_id, 4) is None  # no socket for this attempt yet
        async with job_progress(settings, job_id, 4) as publisher:
            async with asyncio.timeout(5):
                while not publisher.joined:
                    await asyncio.sleep(0.01)
                assert await renew_lease(job_id, 4) == {"status": "ok", "response": lease}
            assert await renew_lease(job_id, 5) is None  # another attempt never borrows it
        assert await renew_lease(job_id, 4) is None  # the scope ended
    assert ("renew_lease", {}) in events
    assert all("fixture-private" not in json.dumps(payload) for _, payload in events)
