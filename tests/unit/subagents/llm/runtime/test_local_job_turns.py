"""Jobs take turns on the local model.

The GPU scheduler admits one generation at a time. Without job turns, two jobs
on the local model interleave their steps, and the second job's routing
overtakes the first job's answer. A job keeps the local model from its first
local step until it ends. Jobs on an outbound connection never wait for it.
"""
import asyncio

import pytest

from services.extensions.context import execution_profile
from subagents.llm.runtime.base import GenerationOutput
from subagents.llm.runtime.inference import InferenceCoordinator
from subagents.llm.runtime.observability import CudaMemorySnapshot
from subagents.llm.runtime.scheduler import GpuScheduler, job_turn
from subagents.llm.runtime.stages import InferenceStage


class LocalModel:
    """Records generations in the order they run on the local GPU."""

    def __init__(self):
        self.ran = []

    def exists(self, model_key):
        return True

    def is_loaded(self, model_key):
        return True

    def generate_observed(self, *, model_key, messages, max_new_tokens):
        self.ran.append(messages[-1]["content"])
        return GenerationOutput(text="{}", generated_tokens=1)


def memory():
    return CudaMemorySnapshot(
        available=False, device_index=None, device_name=None, free_bytes=None,
        total_bytes=None, allocated_bytes=None, reserved_bytes=None,
    )


def local_runtime():
    model = LocalModel()
    return InferenceCoordinator(model, GpuScheduler(), memory, lambda metric: None), model


async def route(coordinator, text, stage=InferenceStage.CONTROL_ROUTING):
    return await coordinator.generate(
        model_key="hub-main", messages=[{"role": "user", "content": text}],
        max_new_tokens=8, priority=20, stage=stage,
    )


def test_a_second_local_job_waits_until_the_first_job_ends():
    async def scenario():
        coordinator, model = local_runtime()
        first_routed, tool_done = asyncio.Event(), asyncio.Event()

        async def first_job():
            with job_turn():
                await route(coordinator, "first routing")
                first_routed.set()
                await tool_done.wait()  # a tool call between two model steps
                await route(coordinator, "first answer")

        async def second_job():
            await first_routed.wait()
            with job_turn():
                await route(coordinator, "second routing")

        jobs = [asyncio.create_task(first_job()), asyncio.create_task(second_job())]
        await first_routed.wait()
        await asyncio.sleep(0.05)
        tool_done.set()
        await asyncio.wait_for(asyncio.gather(*jobs), 5)
        return model.ran

    assert asyncio.run(scenario()) == ["first routing", "first answer", "second routing"]


def test_a_job_on_an_outbound_connection_does_not_wait_for_the_local_model(monkeypatch):
    for key in ("EXTERNAL_INFERENCE_REQUEST_TOKEN_BUDGET", "EXTERNAL_INFERENCE_MAX_OUTPUT_TOKENS",
                "EXTERNAL_HTTP_REQUEST_TOKEN_BUDGET", "EXTERNAL_GROQ_HTTP_REQUEST_TOKEN_BUDGET"):
        monkeypatch.delenv(key, raising=False)

    async def remote(profile, messages, max_new_tokens, **options):
        return "outbound answer"

    monkeypatch.setattr("services.extensions.transport.remote_generate", remote)

    async def scenario():
        coordinator, model = local_runtime()
        holding, finish = asyncio.Event(), asyncio.Event()

        async def local_job():
            with job_turn():
                await route(coordinator, "local routing")
                holding.set()
                await finish.wait()

        async def outbound_job():
            token = execution_profile.set({"backend": "http", "url": "https://api.groq.com/openai/v1",
                                           "model": "selected"})
            try:
                with job_turn():
                    return await route(coordinator, "outbound answer", InferenceStage.PRIMARY_RESPONSE)
            finally:
                execution_profile.reset(token)

        task = asyncio.create_task(local_job())
        await holding.wait()
        answer = await asyncio.wait_for(outbound_job(), 2)
        finish.set()
        await task
        return answer, model.ran

    assert asyncio.run(scenario()) == ("outbound answer", ["local routing"])


def test_parallel_local_steps_of_one_job_share_its_turn():
    async def scenario():
        coordinator, model = local_runtime()
        with job_turn():
            await asyncio.wait_for(asyncio.gather(
                route(coordinator, "read one"), route(coordinator, "read two"),
            ), 2)
        return sorted(model.ran)

    assert asyncio.run(scenario()) == ["read one", "read two"]


def test_a_job_that_fails_or_is_cancelled_gives_up_its_turn():
    async def scenario():
        coordinator, model = local_runtime()

        async def failing_job():
            with job_turn():
                await route(coordinator, "failing routing")
                raise RuntimeError("tool failed")

        with pytest.raises(RuntimeError):
            await failing_job()

        started = asyncio.Event()

        async def cancelled_job():
            with job_turn():
                await route(coordinator, "cancelled routing")
                started.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(cancelled_job())
        await started.wait()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        with job_turn():
            await asyncio.wait_for(route(coordinator, "next routing"), 2)
        return model.ran

    assert asyncio.run(scenario()) == ["failing routing", "cancelled routing", "next routing"]


def test_a_waiting_job_says_it_is_waiting_for_the_model(monkeypatch):
    stages = []
    monkeypatch.setattr("services.job_progress.report_progress", stages.append)

    async def scenario():
        coordinator, _ = local_runtime()
        holding, finish = asyncio.Event(), asyncio.Event()

        async def first_job():
            with job_turn():
                await route(coordinator, "first routing")
                holding.set()
                await finish.wait()

        async def second_job():
            with job_turn():
                await route(coordinator, "second routing")

        first = asyncio.create_task(first_job())
        await holding.wait()
        second = asyncio.create_task(second_job())
        await asyncio.sleep(0.05)
        waiting = list(stages)
        finish.set()
        await asyncio.wait_for(asyncio.gather(first, second), 5)
        return waiting

    waiting = asyncio.run(scenario())
    assert waiting[-1] == "waiting_provider"
    assert stages[-1] == "routing"


def test_steps_outside_a_job_never_wait_for_a_turn():
    async def scenario():
        coordinator, model = local_runtime()
        holding, finish = asyncio.Event(), asyncio.Event()

        async def job():
            with job_turn():
                await route(coordinator, "job routing")
                holding.set()
                await finish.wait()

        task = asyncio.create_task(job())
        await holding.wait()
        # Startup warm-up and approved actions run outside any job.
        await asyncio.wait_for(route(coordinator, "approved action summary"), 2)
        finish.set()
        await task
        return model.ran

    assert asyncio.run(scenario()) == ["job routing", "approved action summary"]


def test_a_durable_job_runs_its_steps_inside_one_turn(monkeypatch):
    import api.app as api
    from contextlib import asynccontextmanager, contextmanager
    from types import SimpleNamespace

    from subagents.core.definitions.types import HubResult
    from subagents.llm.runtime import scheduler as scheduler_module

    seen = []

    class Hub:
        async def run(self, message, **kwargs):
            seen.append(scheduler_module._current_job_turn.get() is not None)
            return HubResult(status="success", user_request=message, answer="done")

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
    request = api.JobExecuteRequest.model_validate({
        "job_id": "c" * 32, "user_id": "alice", "message": "router uptime", "attempt": 1, "context": []})
    completions = []
    runtime = SimpleNamespace(
        settings=SimpleNamespace(), hub=Hub(), active_jobs={request.job_id: object()},
        trajectory_recorder=SimpleNamespace(record=lambda **kwargs: None),
        completion_outbox=SimpleNamespace(put=lambda *args: completions.append(args)),
        outbox_wakeup=asyncio.Event(),
    )
    asyncio.run(api.execute_job(runtime, request))
    assert seen == [True]
    assert scheduler_module._current_job_turn.get() is None
    assert completions[0][2]["status"] == "completed"
