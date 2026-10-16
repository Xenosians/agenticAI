import asyncio

import pytest

from services.extensions.budget import ExternalInferenceBudgetError
from services.extensions.context import execution_profile, retrieval_evidence
from subagents.llm.runtime.base import GenerationOutput
from subagents.llm.runtime.inference import InferenceCoordinator
from subagents.llm.runtime.observability import CudaMemorySnapshot, emit_inference_metric
from subagents.llm.runtime.scheduler import GpuScheduler
from subagents.llm.runtime.stages import InferenceStage


class RecordingManager:
    def __init__(self):
        self.calls = []
        self.loaded = False

    def exists(self, model_key):
        return model_key in {"renamed-model", "networking-user-model", "uploaded-model"}

    def is_loaded(self, model_key):
        return self.loaded

    def load(self, model_key):
        self.calls.append((model_key, "warm"))
        self.loaded = True

    def generate_observed(self, **kwargs):
        self.calls.append(kwargs)
        self.loaded = True
        return GenerationOutput(text="local result", generated_tokens=4)

    def register_user_model(self, profile):
        return "uploaded-model"


def memory():
    return CudaMemorySnapshot(
        available=False, device_index=None, device_name=None, free_bytes=None,
        total_bytes=None, allocated_bytes=None, reserved_bytes=None,
    )


@pytest.fixture
def runtime(monkeypatch):
    for key in (
        "EXTERNAL_INFERENCE_REQUEST_TOKEN_BUDGET", "EXTERNAL_INFERENCE_MAX_OUTPUT_TOKENS",
        "EXTERNAL_HTTP_REQUEST_TOKEN_BUDGET", "EXTERNAL_HARNESS_REQUEST_TOKEN_BUDGET",
        "EXTERNAL_GROQ_HTTP_REQUEST_TOKEN_BUDGET", "EXTERNAL_GROQ_HARNESS_REQUEST_TOKEN_BUDGET",
    ):
        monkeypatch.delenv(key, raising=False)
    manager, metrics, external_calls = RecordingManager(), [], []
    coordinator = InferenceCoordinator(manager, GpuScheduler(), memory, metrics.append)

    async def external(profile, messages, max_new_tokens):
        external_calls.append((profile, messages, max_new_tokens))
        return "remote result"

    monkeypatch.setattr("services.extensions.transport.remote_generate", external)
    monkeypatch.setattr("services.extensions.harness.harness_generate", external)
    profile_token = execution_profile.set({
        "backend": "http", "url": "https://api.groq.com/openai/v1", "model": "selected",
    })
    evidence_token = retrieval_evidence.set([{"content": "private reference", "trust": "untrusted_retrieval"}])
    try:
        yield coordinator, manager, metrics, external_calls
    finally:
        retrieval_evidence.reset(evidence_token)
        execution_profile.reset(profile_token)


def generate(coordinator, stage, **overrides):
    args = dict(
        model_key="renamed-model", messages=[{"role": "user", "content": "Read status"}],
        max_new_tokens=2000, priority=20, stage=stage,
    )
    args.update(overrides)
    return asyncio.run(coordinator.generate(**args))


@pytest.mark.parametrize("stage", [
    InferenceStage.CONTROL_ROUTING, InferenceStage.CANDIDATE_NARROWING, InferenceStage.NETWORKING,
])
def test_local_stages_keep_registered_model_and_exclude_private_retrieval(runtime, stage):
    coordinator, manager, metrics, external = runtime
    assert generate(coordinator, stage) == "local result"
    assert not external
    assert manager.calls[0]["model_key"] == "renamed-model"
    assert manager.calls[0]["messages"] == [{"role": "user", "content": "Read status"}]
    # Preserve local generation limits; external limits do not clip local work.
    assert manager.calls[0]["max_new_tokens"] == 2000
    assert metrics[0].stage == stage
    assert metrics[0].adapter == "local"
    assert metrics[0].generated_tokens == 4
    assert metrics[0].memory_before == memory()


def test_model_name_cannot_override_trusted_stage(runtime):
    coordinator, manager, metrics, external = runtime
    assert generate(coordinator, InferenceStage.SEMANTIC_ROUTING, model_key="networking-user-model") == "remote result"
    assert not manager.calls
    assert len(external[0][1]) == 1  # Routing cannot consume private RAG evidence.
    assert metrics[0].stage == "SEMANTIC_ROUTING"


@pytest.mark.parametrize("stage,repair_of,object_mode", [
    (InferenceStage.SEMANTIC_ROUTING, None, True),
    (InferenceStage.REPAIR, InferenceStage.SEMANTIC_ROUTING, True),
    (InferenceStage.SPECIALIST_REASONING, None, False),
    (InferenceStage.REPAIR, InferenceStage.SPECIALIST_REASONING, False),
    (InferenceStage.PRIMARY_RESPONSE, None, False),
    (InferenceStage.SYNTHESIS, None, False),
])
def test_verified_http_format_follows_stage_and_preserves_existing_dispatch_budget(
    runtime, monkeypatch, stage, repair_of, object_mode,
):
    from services.extensions.response_format import HttpResponseFormat
    coordinator, manager, metrics, _ = runtime
    calls = []
    async def http(profile, messages, max_new_tokens, *, response_format=None):
        calls.append((profile, messages, max_new_tokens, response_format))
        return "provider output"
    monkeypatch.setattr("services.extensions.transport.remote_generate", http)
    execution_profile.set({"backend": "http", "url": "https://api.groq.com/openai/v1",
                           "model": "openai/gpt-oss-20b"})
    assert generate(coordinator, stage, repair_of=repair_of, model_key="networking-user-model",
                    max_new_tokens=512) == "provider output"
    assert not manager.calls and len(calls) == 1
    selected, messages, output_cap, response_format = calls[0]
    assert response_format is (HttpResponseFormat.JSON_OBJECT if object_mode else None)
    assert selected["request_token_budget"] == (5000 if object_mode else 6000)
    assert output_cap == 512  # Formatting must not raise a generation budget.
    assert messages[-1] == {"role": "user", "content": "Read status"}
    assert "response_format" not in selected  # It is not an owner/profile option.
    assert metrics[0].outcome == "success" and metrics[0].adapter == "http"


@pytest.mark.parametrize("backend", ["pi", "opencode"])
def test_harness_semantic_stage_does_not_receive_http_object_protocol(runtime, monkeypatch, backend):
    coordinator, manager, _, _ = runtime
    calls = []
    async def harness(profile, messages, max_new_tokens, **options):
        calls.append(options)
        return "harness output"
    monkeypatch.setattr("services.extensions.harness.harness_generate", harness)
    execution_profile.set({"backend": backend, "model": "groq/openai/gpt-oss-20b"})
    assert generate(coordinator, InferenceStage.SEMANTIC_ROUTING) == "harness output"
    assert calls == [{}] and not manager.calls


@pytest.mark.parametrize("origin, expected_adapter", [
    (InferenceStage.CONTROL_ROUTING, "local"),
    (InferenceStage.NETWORKING, "local"),
    (InferenceStage.SEMANTIC_ROUTING, "http"),
    (InferenceStage.SPECIALIST_REASONING, "http"),
])
def test_repair_inherits_explicit_origin_disclosure_policy(runtime, origin, expected_adapter):
    coordinator, manager, metrics, external = runtime
    generate(coordinator, InferenceStage.REPAIR, repair_of=origin)
    assert metrics[0].stage == "REPAIR"
    assert metrics[0].repair_of == origin
    assert metrics[0].adapter == expected_adapter
    messages = external[0][1] if external else manager.calls[0]["messages"]
    assert any("UNTRUSTED REFERENCE" in message["content"] for message in messages) is (
        origin == InferenceStage.SPECIALIST_REASONING
    )


@pytest.mark.parametrize("stage, repair_of", [
    (InferenceStage.REPAIR, None), (InferenceStage.REPAIR, InferenceStage.REPAIR),
    ("user-supplied-stage", None), (InferenceStage.PRIMARY_RESPONSE, InferenceStage.SYNTHESIS),
])
def test_invalid_stage_never_dispatches_and_records_safe_failure(runtime, stage, repair_of):
    coordinator, manager, metrics, external = runtime
    with pytest.raises(ValueError):
        generate(coordinator, stage, repair_of=repair_of)
    assert not manager.calls and not external
    assert len(metrics) == 1
    assert metrics[0].stage == "INVALID"
    assert metrics[0].outcome == "error"


@pytest.mark.parametrize("backend, stage, config, request_cap, output_cap", [
    ("http", InferenceStage.SEMANTIC_ROUTING, {"request_token_budget": 20000, "max_output_tokens": 5000}, 5000, 768),
    ("pi", InferenceStage.PRIMARY_RESPONSE, {"request_token_budget": 20000, "max_output_tokens": 5000}, 5000, 1024),
    ("opencode", InferenceStage.SYNTHESIS, {"request_token_budget": 1500, "max_output_tokens": 64}, 1500, 64),
])
def test_stage_budgets_only_lower_provider_and_profile_limits(runtime, backend, stage, config, request_cap, output_cap):
    coordinator, _, metrics, external = runtime
    profile = {"backend": backend, "url": "https://api.groq.com/openai/v1", "model": "custom/model", **config}
    execution_profile.set(profile)
    original = dict(profile)
    assert generate(coordinator, stage, max_new_tokens=20000) == "remote result"
    actual_profile, _, actual_output = external[0]
    assert actual_profile["request_token_budget"] == request_cap
    assert actual_output == output_cap
    assert profile == original
    metric = metrics[0]
    assert metric.request_token_budget == request_cap
    assert metric.output_token_cap == output_cap
    assert metric.estimated_input_tokens > 0
    assert metric.total_seconds >= 0
    assert metric.memory_before is None and metric.generated_tokens is None


def test_lower_environment_limits_are_preserved(runtime, monkeypatch):
    coordinator, _, metrics, external = runtime
    monkeypatch.setenv("EXTERNAL_INFERENCE_REQUEST_TOKEN_BUDGET", "1200")
    monkeypatch.setenv("EXTERNAL_INFERENCE_MAX_OUTPUT_TOKENS", "96")
    generate(coordinator, InferenceStage.SEMANTIC_ROUTING)
    assert external[0][0]["request_token_budget"] == 1200
    assert external[0][2] == 96


def test_oversized_mandatory_prompt_fails_before_external_dispatch(runtime):
    coordinator, manager, metrics, external = runtime
    with pytest.raises(ExternalInferenceBudgetError):
        generate(coordinator, InferenceStage.SEMANTIC_ROUTING, messages=[
            {"role": "system", "content": "mandatory instruction " * 2000},
            {"role": "user", "content": "Read status"},
        ])
    assert not external and not manager.calls
    assert len(metrics) == 1
    metric = metrics[0]
    assert metric.outcome == "budget_rejected"
    assert metric.request_token_budget == 5000
    assert metric.estimated_input_tokens > metric.request_token_budget


def test_external_failure_emits_counts_without_prompt_credentials_or_exception_text(runtime, monkeypatch, capsys):
    coordinator, _, metrics, external = runtime

    async def failure(*args):
        raise RuntimeError("credential-value-and-private-prompt")

    monkeypatch.setattr("services.extensions.transport.remote_generate", failure)
    with pytest.raises(RuntimeError):
        generate(coordinator, InferenceStage.PRIMARY_RESPONSE)
    assert len(metrics) == 1
    metric = metrics[0]
    assert metric.outcome == "error" and metric.error_type == "RuntimeError"
    assert metric.adapter == "http" and metric.memory_before is None
    emit_inference_metric(metric)
    output = capsys.readouterr().out
    assert "stage=PRIMARY_RESPONSE" in output and "outcome=error" in output
    assert "credential-value" not in output and "private reference" not in output
    assert "GPU_METRIC" not in output


def test_unregistered_control_model_fails_without_remote_fallback(runtime):
    coordinator, manager, metrics, external = runtime
    with pytest.raises(KeyError):
        generate(coordinator, InferenceStage.CONTROL_ROUTING, model_key="missing")
    assert not external and not manager.calls
    assert metrics[0].outcome == "error" and metrics[0].adapter == "local"


def test_warm_uses_same_explicit_stage_policy_and_reports_skipped_remote(runtime):
    coordinator, manager, metrics, external = runtime
    asyncio.run(coordinator.warm("renamed-model", stage=InferenceStage.CONTROL_ROUTING))
    asyncio.run(coordinator.warm("renamed-model", stage=InferenceStage.PRIMARY_RESPONSE))
    assert manager.calls == [("renamed-model", "warm")]
    assert not external
    assert [metric.outcome for metric in metrics] == ["success", "skipped"]
    assert [metric.stage for metric in metrics] == ["CONTROL_ROUTING", "PRIMARY_RESPONSE"]
    assert metrics[1].memory_before is None


@pytest.mark.parametrize("backend, expected", [("local", "networking-user-model"), ("uploaded", "uploaded-model")])
def test_selected_local_models_use_registered_path_with_stage_retrieval(runtime, backend, expected):
    coordinator, manager, metrics, external = runtime
    execution_profile.set({"backend": backend, "model": "networking-user-model"})
    assert generate(coordinator, InferenceStage.SPECIALIST_REASONING) == "local result"
    assert not external
    assert manager.calls[0]["model_key"] == expected
    assert len(manager.calls[0]["messages"]) == 2
    assert metrics[0].adapter == backend
    assert metrics[0].memory_before is not None


def test_local_generation_failure_preserves_error_and_actual_memory_observation(runtime, monkeypatch):
    coordinator, manager, metrics, external = runtime

    def failure(**kwargs):
        raise RuntimeError("GPU backend failed")

    monkeypatch.setattr(manager, "generate_observed", failure)
    with pytest.raises(RuntimeError, match="GPU backend failed"):
        generate(coordinator, InferenceStage.NETWORKING)
    assert not external
    assert len(metrics) == 1
    assert metrics[0].outcome == "error"
    assert metrics[0].memory_before == metrics[0].memory_after == memory()
    assert metrics[0].total_seconds >= 0
    assert metrics[0].execution_seconds is None  # Scheduler did not return timings.


def test_cancelled_external_request_is_observed_without_retry(runtime, monkeypatch):
    coordinator, _, metrics, external = runtime
    attempts = []

    async def cancelled(*args):
        attempts.append(1)
        raise asyncio.CancelledError()

    monkeypatch.setattr("services.extensions.transport.remote_generate", cancelled)
    with pytest.raises(asyncio.CancelledError):
        generate(coordinator, InferenceStage.SYNTHESIS)
    assert attempts == [1]
    assert len(metrics) == 1
    assert metrics[0].outcome == "cancelled"
    assert metrics[0].adapter == "http"


@pytest.mark.parametrize("failure_at", [1, 2])
def test_memory_observation_failure_does_not_block_local_generation(runtime, failure_at, caplog):
    coordinator, manager, metrics, external = runtime
    observations = []

    def sample():
        observations.append(1)
        if len(observations) == failure_at:
            raise RuntimeError("private observer diagnostic")
        return memory()

    coordinator.memory_sampler = sample
    assert generate(coordinator, InferenceStage.NETWORKING) == "local result"
    assert len(manager.calls) == 1 and not external
    assert len(observations) == 2
    assert len(metrics) == 1 and metrics[0].outcome == "success"
    assert metrics[0].loaded_before is False and metrics[0].loaded_after is True
    assert (metrics[0].memory_before is None) is (failure_at == 1)
    assert (metrics[0].memory_after is None) is (failure_at == 2)
    assert "error_type=RuntimeError" in caplog.text
    assert "private observer diagnostic" not in caplog.text


def test_failed_loaded_state_observation_does_not_block_local_generation(runtime, monkeypatch):
    coordinator, manager, metrics, external = runtime

    def failed_observation(model_key):
        raise RuntimeError("loaded state observation failed")

    monkeypatch.setattr(manager, "is_loaded", failed_observation)
    assert generate(coordinator, InferenceStage.CONTROL_ROUTING) == "local result"
    assert len(manager.calls) == 1 and not external
    assert metrics[0].loaded_before is None and metrics[0].loaded_after is None
    assert metrics[0].outcome == "success"


@pytest.mark.parametrize("original", [RuntimeError("provider failure"), asyncio.CancelledError()])
@pytest.mark.parametrize("observer_error", [RuntimeError, asyncio.CancelledError])
def test_metric_sink_failure_preserves_original_error_or_cancellation(runtime, monkeypatch, original, observer_error):
    coordinator, _, _, _ = runtime

    async def failure(*args):
        raise original

    def failed_sink(metric):
        raise observer_error("sink failed")

    coordinator.metric_sink = failed_sink
    monkeypatch.setattr("services.extensions.transport.remote_generate", failure)
    with pytest.raises(type(original)) as caught:
        generate(coordinator, InferenceStage.SYNTHESIS)
    assert caught.value is original


def test_metric_sink_failure_preserves_successful_result(runtime):
    coordinator, _, _, external = runtime

    def failed_sink(metric):
        raise RuntimeError("sink failed")

    coordinator.metric_sink = failed_sink
    assert generate(coordinator, InferenceStage.PRIMARY_RESPONSE) == "remote result"
    assert len(external) == 1


def test_failed_final_memory_observation_preserves_generation_error(runtime, monkeypatch):
    coordinator, manager, metrics, _ = runtime
    observations = []
    original = RuntimeError("generation failed")

    def sample():
        observations.append(1)
        if len(observations) == 2:
            raise ValueError("memory observer failed")
        return memory()

    def failure(**kwargs):
        raise original

    coordinator.memory_sampler = sample
    monkeypatch.setattr(manager, "generate_observed", failure)
    with pytest.raises(RuntimeError) as caught:
        generate(coordinator, InferenceStage.CONTROL_ROUTING)
    assert caught.value is original
    assert len(metrics) == 1
    assert metrics[0].outcome == "error" and metrics[0].error_type == "RuntimeError"
    assert metrics[0].memory_before == memory() and metrics[0].memory_after is None
