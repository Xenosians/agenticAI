"""The AI stops a cancelled attempt when Phoenix asks.

Phoenix records the cancel first, then asks the AI to stop the attempt's
task. Only an attempt that is still running for that exact attempt number
is stopped; a cancelled attempt sends no completion.
"""
import asyncio
from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import app as api
from subagents.core.definitions.types import HubResult

TOKEN = "isolated-internal-token"


def client_for(runtime, monkeypatch):
    from api import extensions
    monkeypatch.setattr(extensions, "get_settings", lambda: SimpleNamespace(
        require_internal_job_token=lambda: TOKEN,
    ))
    monkeypatch.setattr(api, "require_ready_runtime", lambda request: runtime)
    application = FastAPI()
    application.add_api_route("/v1/jobs/{job_id}/cancel", api.cancel_durable_job, methods=["POST"])
    return TestClient(application)


def test_cancel_requires_the_internal_credential(monkeypatch):
    runtime = SimpleNamespace(active_jobs={}, active_attempts={})
    client = client_for(runtime, monkeypatch)
    assert client.post(f"/v1/jobs/{'a' * 32}/cancel", json={"attempt": 1}).status_code == 401


def test_cancel_stops_the_running_attempt_and_sends_no_completion(monkeypatch):
    completions, started = [], []

    class Hub:
        async def run(self, message, **kwargs):
            started.append(message)
            await asyncio.Event().wait()  # a long model or tool step
            return HubResult(status="success", user_request=message, answer="never")

    @contextmanager
    def owner(*args, **kwargs):
        yield None, False

    @asynccontextmanager
    async def progress(*args, **kwargs):
        yield

    async def heartbeat(*args):
        await asyncio.Event().wait()

    monkeypatch.setattr("services.extensions.context.owner_execution", owner)
    monkeypatch.setattr("services.job_progress.job_progress", progress)
    monkeypatch.setattr(api, "job_heartbeat_worker", heartbeat)

    async def scenario():
        request = api.JobExecuteRequest.model_validate({
            "job_id": "d" * 32, "user_id": "alice", "message": "router uptime", "attempt": 2, "context": []})
        runtime = SimpleNamespace(
            settings=SimpleNamespace(), hub=Hub(), active_jobs={}, active_attempts={},
            trajectory_recorder=SimpleNamespace(record=lambda **kwargs: None),
            completion_outbox=SimpleNamespace(put=lambda *args: completions.append(args)),
            outbox_wakeup=asyncio.Event(),
        )
        task = api.schedule_job(runtime, request)
        while not started:
            await asyncio.sleep(0.01)
        # A different attempt number never stops this task.
        assert api.stop_job_attempt(runtime, request.job_id, 1) == "not_running"
        assert not task.done()
        assert api.stop_job_attempt(runtime, request.job_id, 2) == "stopping"
        await asyncio.gather(task, return_exceptions=True)
        return task, runtime

    task, runtime = asyncio.run(scenario())
    assert task.cancelled()
    assert completions == []
    assert runtime.active_jobs == {}


def test_cancel_of_an_attempt_that_is_not_running_is_reported(monkeypatch):
    runtime = SimpleNamespace(active_jobs={}, active_attempts={})
    client = client_for(runtime, monkeypatch)
    response = client.post(f"/v1/jobs/{'e' * 32}/cancel", json={"attempt": 1},
                           headers={"x-internal-token": TOKEN})
    assert response.status_code == 200
    assert response.json() == {"job_id": "e" * 32, "attempt": 1, "status": "not_running"}


def test_cancel_rejects_an_invalid_attempt(monkeypatch):
    runtime = SimpleNamespace(active_jobs={}, active_attempts={})
    client = client_for(runtime, monkeypatch)
    for body in ({}, {"attempt": 0}, {"attempt": "1"}, {"attempt": 1, "extra": True}):
        response = client.post(f"/v1/jobs/{'f' * 32}/cancel", json=body, headers={"x-internal-token": TOKEN})
        assert response.status_code == 422


def test_an_attempt_stops_when_phoenix_ends_its_lease(monkeypatch):
    # Phoenix's stop request may never arrive. The attempt's own heartbeat then
    # learns the lease ended (cancelled, failed closed or superseded) and the
    # attempt stops there, sending no completion.
    completions, started = [], []

    class Hub:
        async def run(self, message, **kwargs):
            started.append(message)
            await asyncio.Event().wait()
            return HubResult(status="success", user_request=message, answer="never")

    @contextmanager
    def owner(*args, **kwargs):
        yield None, False

    @asynccontextmanager
    async def progress(*args, **kwargs):
        yield

    async def heartbeat_until_lease_ends(*args):
        while not started:
            await asyncio.sleep(0.01)
        # The renewal answered job_not_processing: the worker returns.

    monkeypatch.setattr("services.extensions.context.owner_execution", owner)
    monkeypatch.setattr("services.job_progress.job_progress", progress)
    monkeypatch.setattr(api, "job_heartbeat_worker", heartbeat_until_lease_ends)

    async def scenario():
        request = api.JobExecuteRequest.model_validate({
            "job_id": "c" * 32, "user_id": "alice", "message": "router uptime", "attempt": 1, "context": []})
        runtime = SimpleNamespace(
            settings=SimpleNamespace(), hub=Hub(), active_jobs={}, active_attempts={},
            trajectory_recorder=SimpleNamespace(record=lambda **kwargs: None),
            completion_outbox=SimpleNamespace(put=lambda *args: completions.append(args)),
            outbox_wakeup=asyncio.Event(),
        )
        task = api.schedule_job(runtime, request)
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=5)
        return task, runtime

    task, runtime = asyncio.run(scenario())
    assert task.cancelled()
    assert completions == []
    assert runtime.active_jobs == {}
