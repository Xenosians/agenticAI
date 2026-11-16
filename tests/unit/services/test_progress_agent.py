"""Progress names the agent doing each step, so the owner sees the hub hand a request over.

Display only: the name is an agent identifier from trusted code ("hub" or a
specialist name), never model text, a tool, an argument or a result.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from services.extensions.context import specialist_role
from services.job_progress import JobProgress, job_progress, report_progress
from subagents.llm.runtime.base import GenerationOutput
from subagents.llm.runtime.inference import InferenceCoordinator
from subagents.llm.runtime.observability import CudaMemorySnapshot
from subagents.llm.runtime.scheduler import GpuScheduler
from subagents.llm.runtime.stages import InferenceStage


def test_a_stage_may_name_its_agent_and_both_travel_together():
    publisher = JobProgress("https://example.com", "private-token", "a" * 32, 1)
    publisher.publish("planning", agent="proxmox-specialist")
    assert publisher.queue.get_nowait() == {"stage": "planning", "agent": "proxmox-specialist"}
    publisher.publish("synthesizing")
    assert publisher.queue.get_nowait() == {"stage": "synthesizing"}


@pytest.mark.parametrize("agent", ["Proxmox", "a b", "-x", "x--y", "", "a" * 65, 7, "x\nIgnore previous"])
def test_anything_but_a_plain_agent_name_is_dropped_and_the_stage_still_goes(agent):
    publisher = JobProgress("https://example.com", "private-token", "a" * 32, 1)
    publisher.publish("planning", agent=agent)
    assert publisher.queue.get_nowait() == {"stage": "planning"}


class LocalModel:
    def exists(self, model_key):
        return True

    def is_loaded(self, model_key):
        return True

    def generate_observed(self, *, model_key, messages, max_new_tokens):
        return GenerationOutput(text="{}", generated_tokens=1)


def memory():
    return CudaMemorySnapshot(available=False, device_index=None, device_name=None, free_bytes=None,
                              total_bytes=None, allocated_bytes=None, reserved_bytes=None)


def test_each_inference_step_names_the_agent_doing_it(monkeypatch):
    monkeypatch.setattr(JobProgress, "run", lambda self: asyncio.sleep(0))
    settings = SimpleNamespace(require_phoenix_base_url=lambda: "http://127.0.0.1:4000",
                               require_internal_job_token=lambda: "test-private")
    sent = []

    async def scenario():
        coordinator = InferenceCoordinator(LocalModel(), GpuScheduler(), memory, lambda metric: None)
        async with job_progress(settings, "b" * 32, 1) as publisher:
            original = publisher.publish

            def record(stage, agent=None):
                sent.append((stage, agent))
                original(stage, agent=agent)

            publisher.publish = record

            async def step(stage, role=None):
                async def generate():
                    await coordinator.generate(model_key="m", messages=[{"role": "user", "content": "x"}],
                                               max_new_tokens=4, priority=20, stage=stage)
                if role is None:
                    await generate()
                else:
                    with specialist_role(role):
                        await generate()

            await step(InferenceStage.CONTROL_ROUTING)
            await step(InferenceStage.SPECIALIST_REASONING, "proxmox-specialist")
            with specialist_role("proxmox-specialist"):
                report_progress("executing")
            await step(InferenceStage.SYNTHESIS)

    asyncio.run(scenario())
    assert sent == [("routing", "hub"), ("planning", "proxmox-specialist"),
                    ("executing", "proxmox-specialist"), ("synthesizing", "hub")]


def test_the_gateway_names_the_agent_whose_tool_runs_even_outside_his_role(monkeypatch):
    from subagents.core.definitions.types import AgentDefinition
    from subagents.core.tooling.gateway import ToolGateway

    reported = []
    monkeypatch.setattr("services.job_progress.report_progress",
                        lambda stage, agent=None: reported.append((stage, agent)))

    class Mcp:
        async def call_tool(self, name, arguments):
            return {"ok": True}

    agent = AgentDefinition(name="networking-specialist", description="d", model="m",
                            tools=["network_router_health"])
    gateway = ToolGateway(approval_creator=None, mcp=Mcp())
    asyncio.run(gateway.execute(agent, "How is the router?", "network_router_health", {}))
    assert ("executing", "networking-specialist") in reported


def test_an_older_phoenix_that_refuses_agent_names_still_gets_every_stage():
    from websockets.asyncio.server import serve

    async def scenario():
        frames = asyncio.Queue()

        async def handler(socket):
            async for raw in socket:
                frame = json.loads(raw)
                reply = {"status": "ok", "response": {}}
                if frame[3] == "progress":
                    await frames.put(frame[4])
                    if set(frame[4]) != {"stage"}:  # this Phoenix predates agent names
                        reply = {"status": "error", "response": {"reason": "invalid_request"}}
                await socket.send(json.dumps([frame[0], frame[1], frame[2], "phx_reply", reply]))

        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            publisher = JobProgress(f"http://127.0.0.1:{port}", "test-private", "a" * 32, 1)
            task = asyncio.create_task(publisher.run())
            try:
                async with asyncio.timeout(5):
                    publisher.publish("planning", agent="proxmox-specialist")
                    first, second = await frames.get(), await frames.get()
                    publisher.publish("executing", agent="proxmox-specialist")
                    third = await frames.get()
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        return first, second, third, publisher.names_agents

    first, second, third, naming = asyncio.run(scenario())
    assert first == {"stage": "planning", "agent": "proxmox-specialist"}
    assert second == {"stage": "planning"}
    assert third == {"stage": "executing"}
    assert naming is False
