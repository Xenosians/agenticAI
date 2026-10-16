import asyncio

import pytest

from services.extensions.diagnostics import ExternalProviderError
from services.extensions.response_format import (
    HttpResponseFormat, http_response_format_for_stage, supported_http_response_formats,
)
from services.extensions.transport import remote_generate
from subagents.llm.runtime.stages import InferenceStage


def profile(**overrides):
    return {"backend": "http", "url": "https://api.groq.com/openai/v1",
            "model": "openai/gpt-oss-20b", **overrides}


@pytest.mark.parametrize("model", ["openai/gpt-oss-20b", "openai/gpt-oss-120b"])
@pytest.mark.parametrize("repair", [False, True])
def test_verified_routing_and_its_explicit_repair_request_object_syntax(model, repair):
    result = http_response_format_for_stage(
        profile(model=model), stage=InferenceStage.REPAIR if repair else InferenceStage.SEMANTIC_ROUTING,
        repair_of=InferenceStage.SEMANTIC_ROUTING if repair else None,
    )
    assert result is HttpResponseFormat.JSON_OBJECT


@pytest.mark.parametrize("stage", [
    InferenceStage.CONTROL_ROUTING, InferenceStage.CANDIDATE_NARROWING, InferenceStage.NETWORKING,
    InferenceStage.SPECIALIST_REASONING, InferenceStage.PRIMARY_RESPONSE, InferenceStage.SYNTHESIS,
])
def test_other_stages_and_their_repairs_keep_original_output_protocol(stage):
    selected = profile(stage="SEMANTIC_ROUTING", response_format="json_object")
    assert http_response_format_for_stage(selected, stage=stage) is None
    assert http_response_format_for_stage(selected, stage=InferenceStage.REPAIR, repair_of=stage) is None


@pytest.mark.parametrize("overrides", [
    {"url": "https://proxy.example/v1"},
    {"url": "https://api.groq.com.evil.test/openai/v1"},
    {"url": "http://api.groq.com/openai/v1"},
    {"url": "https://api.groq.com/openai/v1/custom"},
    {"url": "https://api.groq.com/openai/v1?stage=SEMANTIC_ROUTING"},
    {"model": "unverified-model"},
    {"backend": "pi", "model": "groq/openai/gpt-oss-20b"},
    {"backend": "opencode", "model": "groq/openai/gpt-oss-20b"},
    {"backend": "local"}, {"backend": "uploaded"},
    {"url": None}, {"model": {"untrusted": "value"}},
])
def test_unsupported_endpoint_model_or_adapter_gets_no_inferred_capability(overrides):
    candidate = profile(**overrides)
    assert supported_http_response_formats(candidate) == frozenset()
    assert http_response_format_for_stage(candidate, stage=InferenceStage.SEMANTIC_ROUTING) is None


@pytest.mark.parametrize("stage,repair_of", [("invented", None), (InferenceStage.REPAIR, None),
                                           (InferenceStage.REPAIR, InferenceStage.REPAIR)])
def test_invalid_intent_fails_before_provider_format_selection(stage, repair_of):
    with pytest.raises(ValueError):
        http_response_format_for_stage(profile(), stage=stage, repair_of=repair_of)


def test_http_payload_adds_only_typed_syntax_option_and_preserves_prompt_budget_and_authority(monkeypatch):
    calls = []
    messages = [{"role": "system", "content": 'Return JSON {"delegations":[]}.'},
                {"role": "user", "content": "Create the explicitly supplied account Alice Smith."}]
    async def request(url, **kwargs):
        calls.append((url, kwargs))
        return {"choices": [{"message": {"content": '{"delegations":[]}'}, "finish_reason": "stop"}]}
    monkeypatch.setattr("services.extensions.transport.request_json", request)
    result = asyncio.run(remote_generate(profile(), messages, 64, response_format=HttpResponseFormat.JSON_OBJECT))
    assert result == '{"delegations":[]}'
    assert len(calls) == 1
    body = calls[0][1]["payload"]
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"] == messages and body["max_tokens"] == 64
    assert set(body) == {"model", "messages", "max_tokens", "stream", "response_format"}
    assert "tools" not in body and "tool_choice" not in body and body["stream"] is False


def test_profile_cannot_sneak_in_output_format_when_trusted_call_does_not_request_it(monkeypatch):
    captured = []
    async def request(url, **kwargs):
        captured.append(kwargs["payload"])
        return {"choices": [{"message": {"content": "[ ]"}, "finish_reason": "stop"}]}
    monkeypatch.setattr("services.extensions.transport.request_json", request)
    assert asyncio.run(remote_generate(profile(response_format={"type": "json_object"}), [], 64)) == "[ ]"
    assert "response_format" not in captured[0]


@pytest.mark.parametrize("selected,mode", [
    (profile(model="unverified-model"), HttpResponseFormat.JSON_OBJECT),
    (profile(url="https://proxy.example/v1"), HttpResponseFormat.JSON_OBJECT),
    (profile(), "json_object"), (profile(), {"type": "json_object"}),
])
def test_unsupported_or_untyped_format_cannot_dispatch(selected, mode, monkeypatch):
    async def unexpected(*args, **kwargs):
        pytest.fail("Unsupported output format reached provider")
    monkeypatch.setattr("services.extensions.transport.request_json", unexpected)
    with pytest.raises(ValueError, match="Unsupported HTTP response format"):
        asyncio.run(remote_generate(selected, [], 64, response_format=mode))


def test_truncated_json_mode_output_is_rejected_without_retry_even_when_syntax_looks_valid(monkeypatch):
    calls = []
    async def request(url, **kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": '{"delegations":[]}'}, "finish_reason": "length"}]}
    monkeypatch.setattr("services.extensions.transport.request_json", request)
    with pytest.raises(ExternalProviderError, match="provider_output_truncated"):
        asyncio.run(remote_generate(profile(), [], 64, response_format=HttpResponseFormat.JSON_OBJECT))
    assert len(calls) == 1


def test_transport_never_patches_malformed_provider_output(monkeypatch):
    async def request(url, **kwargs):
        return {"choices": [{"message": {"content": '{"delegations":['}, "finish_reason": "stop"}]}
    monkeypatch.setattr("services.extensions.transport.request_json", request)
    assert asyncio.run(remote_generate(profile(), [], 64, response_format=HttpResponseFormat.JSON_OBJECT)) == '{"delegations":['
