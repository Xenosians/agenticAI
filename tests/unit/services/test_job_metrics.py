import asyncio
from dataclasses import replace
import json
from types import SimpleNamespace

import httpx
import pytest

from services import job_metrics as module
from services.job_metrics import JobMetricPublisher, job_metrics, metric_payload, report_inference_metric
from subagents.llm.runtime.observability import InferenceMetric


def observation(**overrides):
    value = InferenceMetric(
        operation="generate", model_key="PRIVATE-MODEL", priority=1,
        loaded_before=True, loaded_after=True, queue_wait_seconds=0.01,
        execution_seconds=0.2, total_seconds=0.22, generated_tokens=7,
        tokens_per_second=35, first_token_seconds=0.03, memory_before=None, memory_after=None,
        stage="SPECIALIST_REASONING", repair_of=None, adapter="local", outcome="success",
        error_type="PRIVATE-EXCEPTION", input_message_count=3, estimated_input_tokens=None,
        request_token_budget=None, output_token_cap=None, dropped_messages=0, role="jira-specialist",
    )
    return replace(value, **overrides)


def test_export_is_content_free_and_preserves_observed_and_unknown_values():
    data = metric_payload(observation(), attempt=2, event_id="a" * 32)
    assert data["duration_ms"] == 220 and data["queue_ms"] == 10 and data["generated_tokens"] == 7
    assert data["estimated_input_tokens"] is None and data["request_token_budget"] is None
    assert "PRIVATE" not in json.dumps(data)
    assert not {"model_key", "error_type", "messages", "memory_before", "user_id"} & data.keys()
    remote = metric_payload(observation(adapter="http", queue_wait_seconds=None,
        execution_seconds=None, first_token_seconds=None, generated_tokens=None), attempt=1, event_id="b" * 32)
    assert all(remote[key] is None for key in ("queue_ms", "execution_ms", "first_token_ms", "generated_tokens"))
    assert metric_payload(observation(adapter="hf-causal"), attempt=1, event_id="c" * 32)["adapter"] == "hf-causal"


@pytest.mark.parametrize("change", [
    {"total_seconds": float("nan")}, {"execution_seconds": -1}, {"total_seconds": True},
    {"generated_tokens": True}, {"generated_tokens": -1}, {"generated_tokens": 1_000_000_001},
    {"stage": "PRIVATE-PROMPT"}, {"repair_of": "PRIVATE-PROMPT"},
    {"role": "SECRET credential"}, {"adapter": "invented"}, {"outcome": "invented"},
])
def test_invalid_observations_are_not_exported(change):
    with pytest.raises(ValueError):
        metric_payload(observation(**change), attempt=1, event_id="a" * 32)


def test_scopes_bind_events_to_each_job_and_attempt_and_restore_parent(monkeypatch):
    async def scenario():
        received = []
        async def capture(self, payload):
            received.append((self.url, dict(payload)))
        monkeypatch.setattr(JobMetricPublisher, "_send", capture)
        settings = SimpleNamespace(require_phoenix_base_url=lambda: "https://fixture.example/prefix",
            require_internal_job_token=lambda: "PRIVATE-TOKEN")
        assert report_inference_metric(observation()) is False
        async with job_metrics(settings, "a" * 32, 2):
            assert report_inference_metric(observation())
            async with job_metrics(settings, "b" * 32, 3):
                assert report_inference_metric(observation())
            assert report_inference_metric(observation())
        assert report_inference_metric(observation()) is False
        assert sorted(payload["attempt"] for _, payload in received) == [2, 2, 3]
        assert all(f"/jobs/{'b' * 32}/metrics" in url for url, payload in received if payload["attempt"] == 3)
        assert len({payload["event_id"] for _, payload in received}) == 3
        assert "PRIVATE" not in json.dumps(received)
    asyncio.run(scenario())


def test_bounded_queue_and_observer_failure_never_change_caller_result():
    async def scenario():
        entered = asyncio.Event()
        async def unavailable(_payload):
            entered.set()
            await asyncio.Event().wait()
        publisher = JobMetricPublisher("https://fixture.example", "key", "a" * 32, 1,
            send=unavailable, queue_size=1, flush_timeout=0.01)
        assert publisher.publish(observation()) is True
        assert publisher.publish(observation()) is False
        await entered.wait()
        await publisher.close()
        assert publisher.worker.done()
    asyncio.run(scenario())


def test_scope_keeps_caller_cancellation_and_handles_unconfigured_metrics():
    async def scenario():
        with pytest.raises(asyncio.CancelledError):
            async with job_metrics(SimpleNamespace(), "a" * 32, 1):
                assert report_inference_metric(observation()) is False
                raise asyncio.CancelledError()
    asyncio.run(scenario())


def test_transport_retries_only_same_observation_and_does_not_follow_redirects(monkeypatch):
    async def scenario():
        calls, options = [], []
        class Client:
            def __init__(self, **kwargs): options.append(kwargs)
            async def __aenter__(self): return self
            async def __aexit__(self, *_): pass
            async def post(self, url, **kwargs):
                calls.append((url, kwargs))
                if len(calls) == 1: raise httpx.ConnectError("PRIVATE-EXCEPTION")
                return SimpleNamespace(status_code=201)
        monkeypatch.setattr(httpx, "AsyncClient", Client)
        publisher = JobMetricPublisher("https://fixture.example", "PRIVATE-TOKEN", "a" * 32, 1)
        payload = metric_payload(observation(), attempt=1, event_id="a" * 32)
        await publisher._send(payload)
        assert len(calls) == 2 and calls[0][1]["json"] == calls[1][1]["json"] == payload
        assert len(options) == 1
        # Same transport options as before, plus the Lua stage every outbound request crosses.
        hooks = options[0].pop("event_hooks")
        assert options == [{"timeout": 0.75, "follow_redirects": False, "trust_env": False}]
        assert [hook.__name__ for hook in hooks["request"]] == ["_async_request"]
        calls.clear()
        async def conflict(_self, url, **kwargs):
            calls.append((url, kwargs))
            return SimpleNamespace(status_code=409)
        monkeypatch.setattr(Client, "post", conflict)
        await publisher._send(payload)
        assert len(calls) == 1
    asyncio.run(scenario())


def test_actual_inference_boundary_publishes_a_job_observation(monkeypatch):
    async def scenario():
        from subagents.llm.runtime.base import GenerationOutput
        from subagents.llm.runtime.inference import InferenceCoordinator
        from subagents.llm.runtime.scheduler import GpuScheduler
        received = []
        async def capture(self, payload): received.append(payload)
        monkeypatch.setattr(JobMetricPublisher, "_send", capture)
        class Manager:
            def exists(self, _key): return True
            def is_loaded(self, _key): return True
            def generate_observed(self, **_kwargs): return GenerationOutput(text="Fixture response", generated_tokens=3, first_token_seconds=0.01)
        settings = SimpleNamespace(require_phoenix_base_url=lambda: "https://fixture.example",
            require_internal_job_token=lambda: "PRIVATE-TOKEN")
        coordinator = InferenceCoordinator(Manager(), GpuScheduler(), memory_sampler=lambda: None, metric_sink=lambda _: None)
        async with job_metrics(settings, "c" * 32, 4):
            text = await coordinator.generate(model_key="PRIVATE-MODEL",
                messages=[{"role": "user", "content": "PRIVATE-PROMPT"}], max_new_tokens=10,
                priority=1, stage="SPECIALIST_REASONING")
        assert text == "Fixture response" and len(received) == 1
        assert received[0]["attempt"] == 4 and received[0]["generated_tokens"] == 3
        assert received[0]["duration_ms"] >= 0 and received[0]["execution_ms"] >= 0
        assert "PRIVATE" not in json.dumps(received)
    asyncio.run(scenario())


def test_metric_transport_failure_preserves_the_original_operation_error(monkeypatch):
    async def scenario():
        async def unavailable(self, _payload): raise ValueError("PRIVATE-TOKEN")
        monkeypatch.setattr(JobMetricPublisher, "_send", unavailable)
        settings = SimpleNamespace(require_phoenix_base_url=lambda: "https://fixture.example",
            require_internal_job_token=lambda: "PRIVATE-TOKEN")
        with pytest.raises(RuntimeError, match="original operation failure"):
            async with job_metrics(settings, "d" * 32, 1):
                assert report_inference_metric(observation())
                raise RuntimeError("original operation failure")
    asyncio.run(scenario())


def test_metrics_work_when_optional_progress_websocket_is_disabled(monkeypatch):
    async def scenario():
        from services.job_progress import job_progress
        received = []
        async def capture(self, payload): received.append(payload)
        monkeypatch.setattr(JobMetricPublisher, "_send", capture)
        settings = SimpleNamespace(job_websocket_enabled=False,
            require_phoenix_base_url=lambda: "https://fixture.example",
            require_internal_job_token=lambda: "PRIVATE-TOKEN")
        async with job_progress(settings, "e" * 32, 3) as progress:
            assert progress is None and report_inference_metric(observation())
        assert len(received) == 1 and received[0]["attempt"] == 3
    asyncio.run(scenario())


def test_disabled_or_unconfigured_child_cannot_publish_into_parent_job(monkeypatch):
    async def scenario():
        received = []
        async def capture(self, payload): received.append((self.url, payload))
        monkeypatch.setattr(JobMetricPublisher, "_send", capture)
        settings = SimpleNamespace(require_phoenix_base_url=lambda: "https://fixture.example",
            require_internal_job_token=lambda: "PRIVATE-TOKEN")
        async with job_metrics(settings, "a" * 32, 1):
            for child in (SimpleNamespace(), SimpleNamespace(job_metrics_enabled=False)):
                async with job_metrics(child, "b" * 32, 2):
                    assert report_inference_metric(observation()) is False
            assert report_inference_metric(observation()) is True
        assert len(received) == 1 and "/jobs/" + "a" * 32 + "/metrics" in received[0][0]
    asyncio.run(scenario())
