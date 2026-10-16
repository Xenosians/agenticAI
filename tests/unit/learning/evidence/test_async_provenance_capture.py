import asyncio
import threading
from types import SimpleNamespace

import pytest

import learning.evidence.hub_routing as hub_evidence
import subagents.core.orchestration.runtime as runtime_module
from services.extensions.context import execution_profile
from subagents.core.definitions.types import AgentDefinition, AgentTask
from subagents.core.orchestration.candidate_selector import Candidate, CandidateSet
from subagents.core.orchestration.router import LLMRouter, RoutingContractError
from subagents.core.orchestration.runtime import AgentRuntime
from subagents.llm.runtime.stages import InferenceStage


async def _release_started_worker(started, release):
    # Keep the event loop running while the simulated checkpoint read blocks.
    async with asyncio.timeout(2):
        while not started.is_set():
            await asyncio.sleep(0)
        release.set()


def _capture_arguments(request="Check the account state"):
    return {
        "model_key": "hub-main",
        "model_profile_resolver": lambda _: None,
        "specialists": [],
        "messages": [
            {"role": "system", "content": "Routing contract"},
            {"role": "user", "content": request},
        ],
        "context_messages": [],
        "user_request": request,
        "repair_mode": False,
        "include_workflow_protocol": False,
        "repair_error": None,
        "max_new_tokens": 128,
        "response": '{"delegations":[]}',
    }


def test_hub_identity_read_yields_loop_and_activates_evidence_in_caller(monkeypatch):
    started, release = threading.Event(), threading.Event()
    loop_thread = threading.get_ident()

    def blocked_identity(**kwargs):
        assert threading.get_ident() != loop_thread
        started.set()
        assert release.wait(3), "The event loop could not release the checkpoint reader"
        return hub_evidence.HubModelIdentity(model_key=kwargs["model_key"])

    monkeypatch.setattr(hub_evidence, "build_hub_model_identity", blocked_identity)

    async def scenario():
        hub_evidence.reset_hub_routing_trace()
        heartbeat = asyncio.create_task(_release_started_worker(started, release))
        try:
            attempt = await hub_evidence.capture_hub_routing_generation_async(**_capture_arguments())
            hub_evidence.set_current_hub_routing_parsed_output({"delegations": []})
            hub_evidence.accept_current_hub_routing_attempt(validated_delegation_count=0)
            await heartbeat
            trace = hub_evidence.consume_hub_routing_trace()
            assert len(trace) == 1
            assert trace[0].attempt_id == attempt.attempt_id
            assert trace[0].parsed_output == {"delegations": []}
            assert trace[0].validation_status == "accepted"
        finally:
            release.set()
            await heartbeat

    asyncio.run(scenario())


def test_concurrent_hub_capture_keeps_each_tasks_trace_separate():
    async def request_capture(request):
        hub_evidence.reset_hub_routing_trace()
        arguments = _capture_arguments(request)
        arguments["model_profile_resolver"] = None
        await hub_evidence.capture_hub_routing_generation_async(**arguments)
        await asyncio.sleep(0)
        hub_evidence.accept_current_hub_routing_attempt(validated_delegation_count=0)
        trace = hub_evidence.consume_hub_routing_trace()
        assert len(trace) == 1
        return trace[0]

    async def scenario():
        hub_evidence.reset_hub_routing_trace()
        first, second = await asyncio.gather(request_capture("first"), request_capture("second"))
        assert first.user_request_sha256 != second.user_request_sha256
        assert first.attempt_id != second.attempt_id
        assert hub_evidence.consume_hub_routing_trace() == []

    asyncio.run(scenario())


def test_cancelled_hub_fingerprint_does_not_activate_partial_evidence(monkeypatch):
    started, release = threading.Event(), threading.Event()

    def blocked_identity(**kwargs):
        started.set()
        assert release.wait(3)
        return hub_evidence.HubModelIdentity(model_key=kwargs["model_key"])

    monkeypatch.setattr(hub_evidence, "build_hub_model_identity", blocked_identity)

    async def request_capture():
        hub_evidence.reset_hub_routing_trace()
        try:
            await hub_evidence.capture_hub_routing_generation_async(**_capture_arguments())
        except asyncio.CancelledError:
            assert hub_evidence.accept_current_hub_routing_attempt(validated_delegation_count=0) is None
            assert hub_evidence.consume_hub_routing_trace() == []
            raise

    async def scenario():
        task = asyncio.create_task(request_capture())
        try:
            async with asyncio.timeout(2):
                while not started.is_set():
                    await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()

    asyncio.run(scenario())


class _Inference:
    def __init__(self, response="not-json"):
        self.response = response
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_compact_remote_routing_does_not_resolve_or_fingerprint_local_model(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Remote routing must not inspect a local checkpoint")

    inference = _Inference('{"delegations":[]}')
    router = LLMRouter(
        registry=SimpleNamespace(), inference=inference, model_key="hub-main",
        model_profile_resolver=forbidden,
    )
    # Isolate the evidence boundary after candidate discovery has succeeded.
    monkeypatch.setattr(router, "_routing_input", lambda *args, **kwargs: (
        [], "Compact routing contract", InferenceStage.SEMANTIC_ROUTING,
        CandidateSet((Candidate("account-specialist", ("account_status",)),), confidence=1),
    ))
    monkeypatch.setattr(hub_evidence, "fingerprint_runtime_model_artifact", forbidden)

    async def scenario():
        hub_evidence.reset_hub_routing_trace()
        with pytest.raises(RoutingContractError, match="No supported action"):
            await router.route("Check the account state")
        trace = hub_evidence.consume_hub_routing_trace()
        assert len(trace) == 1
        assert trace[0].model_identity.model_weights_sha256 is None
        assert trace[0].model_identity.incomplete_reason == "model-profile-resolver-unavailable"

    asyncio.run(scenario())


def _specialist_runtime(monkeypatch, agent_name="account-specialist"):
    agent = AgentDefinition(
        name=agent_name, description="Test specialist", model="test-local-model",
        tools=["account_status"], max_steps=1, system_prompt="Test prompt",
    )
    monkeypatch.setattr(runtime_module, "build_agent_capability_catalog", lambda *args, **kwargs: [])
    monkeypatch.setattr(runtime_module, "build_worker_system_prompt", lambda *args, **kwargs: "Contract")
    inference = _Inference()
    runtime = AgentRuntime(
        agent_registry=SimpleNamespace(get=lambda _: agent), inference=inference,
        tool_gateway=None,
        model_profile_resolver=lambda _: SimpleNamespace(worker_prompt_profile="standard"),
    )
    task = AgentTask(task_id="async-evidence", agent_name=agent_name, user_request="Check state")
    return runtime, inference, task


def test_specialist_fingerprint_yields_event_loop_before_generation(monkeypatch):
    started, release = threading.Event(), threading.Event()
    loop_thread = threading.get_ident()

    def blocked_provenance(**kwargs):
        assert threading.get_ident() != loop_thread
        started.set()
        assert release.wait(3), "The event loop could not release the checkpoint reader"
        return SimpleNamespace(model_dump=lambda **_: {"captured": True})

    monkeypatch.setattr(runtime_module, "build_specialist_execution_provenance", blocked_provenance)
    runtime, inference, task = _specialist_runtime(monkeypatch)

    async def scenario():
        heartbeat = asyncio.create_task(_release_started_worker(started, release))
        try:
            result = await runtime.run(task)
            await heartbeat
            assert result.outcome_code == "tool_parse_error"
            assert task.execution_provenance == {"captured": True}
            assert len(inference.calls) == 1
        finally:
            release.set()
            await heartbeat

    asyncio.run(scenario())


@pytest.mark.parametrize("agent_name, expected_captures", [
    ("account-specialist", 0), ("networking-specialist", 1),
])
def test_remote_selection_skips_local_provenance_except_protected_networking(
    monkeypatch, agent_name, expected_captures,
):
    captures = []

    def capture(**kwargs):
        captures.append(kwargs)
        return SimpleNamespace(model_dump=lambda **_: {"captured": True})

    monkeypatch.setattr(runtime_module, "build_specialist_execution_provenance", capture)
    runtime, inference, task = _specialist_runtime(monkeypatch, agent_name)
    token = execution_profile.set({"backend": "http", "model": "remote-model"})
    try:
        result = asyncio.run(runtime.run(task))
    finally:
        execution_profile.reset(token)
    assert result.outcome_code == "tool_parse_error"
    assert len(captures) == expected_captures
    assert len(inference.calls) == 1
    assert task.execution_provenance == ({"captured": True} if expected_captures else None)
