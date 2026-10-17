"""Anthropic Messages API adapter: request shape, authority boundary, failures."""

import asyncio
import json

import httpx
import pytest

from services.extensions import anthropic
from services.extensions.diagnostics import ExternalProviderError
from services.extensions.transport import remote_generate


URL = "https://api.anthropic.com"
_REAL_ASYNC_CLIENT = httpx.AsyncClient


def profile(**extra):
    return {"backend": "http", "url": URL, "model": "claude-sonnet-5-5",
            "api_key": "unit-test-anthropic-key", **extra}


@pytest.fixture(autouse=True)
def clear_capability_cache():
    anthropic._capability_cache.clear()
    yield
    anthropic._capability_cache.clear()


def capability_record(*, disabled=True, thinking=True, max_tokens=128_000):
    return {
        "id": "claude-sonnet-5-5", "display_name": "Claude Sonnet 5.5", "max_tokens": max_tokens,
        "capabilities": {
            "effort": {"supported": True, "low": {"supported": True},
                       "medium": {"supported": True}, "high": {"supported": True}},
            "thinking": {"supported": thinking, "types": {
                "adaptive": {"supported": True}, "disabled": {"supported": disabled},
                "enabled": {"supported": False}}},
        },
    }


class FakeTransport:
    """Real httpx client over a mock transport; records each request."""

    def __init__(self, handler):
        self.requests = []
        self.handler = handler

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)


@pytest.fixture
def http(monkeypatch):
    def install(handler):
        fake = FakeTransport(handler)

        def client(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(fake)
            return _REAL_ASYNC_CLIENT(*args, **kwargs)

        monkeypatch.setattr(anthropic.httpx, "AsyncClient", client)
        return fake
    return install


def ok_message(text="safe response", stop_reason="end_turn", extra_blocks=()):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
            "content": [*extra_blocks, {"type": "text", "text": text}],
            "stop_reason": stop_reason, "usage": {"input_tokens": 40, "output_tokens": 9}}


def router(message_body, *, status=200, model_record=None, model_status=200):
    def handle(request):
        if request.url.path.startswith("/v1/models/"):
            return httpx.Response(model_status, json=model_record or capability_record())
        if request.url.path == "/v1/messages":
            return httpx.Response(status, json=message_body)
        return httpx.Response(404, json={"type": "error", "error": {"type": "not_found_error"}})
    return handle


def test_messages_request_is_inference_only(http):
    fake = http(router(ok_message()))
    result = asyncio.run(remote_generate(profile(), [
        {"role": "system", "content": "You are the account specialist."},
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
        {"role": "system", "content": "VERIFIED WORKFLOW CONTENT: account jdoe"},
        {"role": "user", "content": "UNTRUSTED REFERENCE DATA."},
        {"role": "user", "content": "check account jdoe"},
    ], 256))
    assert result == "safe response"
    message = [r for r in fake.requests if r.url.path == "/v1/messages"][0]
    assert message.headers["x-api-key"] == "unit-test-anthropic-key"
    assert message.headers["anthropic-version"] == anthropic.API_VERSION
    assert "authorization" not in message.headers
    body = json.loads(message.content)
    assert body["system"] == ("You are the account specialist.\n\n"
                              "VERIFIED WORKFLOW CONTENT: account jdoe")
    assert body["messages"] == [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
        {"role": "user", "content": "UNTRUSTED REFERENCE DATA.\n\ncheck account jdoe"},
    ]
    # Thinking disabled when the model allows it: the whole budget is the answer.
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_tokens"] == 256
    for forbidden in ("tools", "tool_choice", "mcp_servers", "container"):
        assert forbidden not in body


def test_adaptive_only_models_reserve_thinking_allowance(http):
    fake = http(router(ok_message(extra_blocks=[{"type": "thinking", "thinking": "", "signature": "s"}]),
                       model_record=capability_record(disabled=False)))
    assert asyncio.run(remote_generate(profile(), [{"role": "user", "content": "hi"}], 256)) == "safe response"
    body = json.loads([r for r in fake.requests if r.url.path == "/v1/messages"][0].content)
    assert "thinking" not in body
    assert body["max_tokens"] == 256 + anthropic.DEFAULT_THINKING_ALLOWANCE


def test_effort_and_thinking_options_are_sent(http):
    fake = http(router(ok_message()))
    asyncio.run(remote_generate(profile(reasoning_effort="low", thinking_mode="adaptive"),
                                [{"role": "user", "content": "hi"}], 128))
    body = json.loads([r for r in fake.requests if r.url.path == "/v1/messages"][0].content)
    assert body["output_config"] == {"effort": "low"}
    assert body["thinking"] == {"type": "adaptive"}


def test_unsupported_effort_level_fails_before_inference(http):
    record = capability_record()
    record["capabilities"]["effort"]["low"] = {"supported": False}
    fake = http(router(ok_message(), model_record=record))
    with pytest.raises(ValueError, match="effort"):
        asyncio.run(remote_generate(profile(reasoning_effort="low"), [{"role": "user", "content": "hi"}], 128))
    assert not [r for r in fake.requests if r.url.path == "/v1/messages"]


@pytest.mark.parametrize("stop_reason, category", [
    ("max_tokens", "provider_output_truncated"),
    ("model_context_window_exceeded", "provider_output_truncated"),
    ("refusal", "provider_refused"),
    ("pause_turn", "provider_incomplete_turn"),
    ("tool_use", "provider_unexpected_tool_use"),
    ("something_new", "provider_incomplete_turn"),
])
def test_incomplete_stop_reasons_fail_closed(http, stop_reason, category):
    http(router(ok_message('{"tool":', stop_reason=stop_reason)))
    with pytest.raises(ExternalProviderError) as raised:
        asyncio.run(remote_generate(profile(), [{"role": "user", "content": "hi"}], 128))
    assert raised.value.category == category


def test_unrequested_tool_blocks_and_empty_text_are_rejected(http):
    http(router(ok_message(extra_blocks=[{"type": "tool_use", "id": "t", "name": "x", "input": {}}])))
    with pytest.raises(ExternalProviderError, match="provider_unexpected_tool_use"):
        asyncio.run(remote_generate(profile(), [{"role": "user", "content": "hi"}], 128))
    http(router(ok_message("   ")))
    with pytest.raises(ExternalProviderError, match="provider_empty_response"):
        asyncio.run(remote_generate(profile(), [{"role": "user", "content": "hi"}], 128))


@pytest.mark.parametrize("status, error_type, category", [
    (401, "authentication_error", "provider_authentication_failed"),
    (402, "billing_error", "provider_billing_required"),
    (403, "permission_error", "provider_permission_denied"),
    (413, "request_too_large", "provider_request_too_large"),
    (429, "rate_limit_error", "provider_rate_limited"),
    (500, "api_error", "provider_unavailable"),
    (504, "timeout_error", "provider_timeout"),
    (529, "overloaded_error", "provider_overloaded"),
])
def test_provider_errors_are_classified_without_leaking_the_key(http, capsys, status, error_type, category):
    body = {"type": "error", "error": {"type": error_type,
                                        "message": "bad key unit-test-anthropic-key"}}
    http(router(body, status=status))
    with pytest.raises(ExternalProviderError) as raised:
        asyncio.run(remote_generate(profile(), [{"role": "user", "content": "hi"}], 128))
    assert raised.value.category == category
    assert raised.value.status_code == status
    assert "unit-test-anthropic-key" not in str(raised.value)
    assert "unit-test-anthropic-key" not in capsys.readouterr().out


def test_missing_model_fails_before_inference(http):
    fake = http(router(ok_message(), model_status=404,
                       model_record={"type": "error", "error": {"type": "not_found_error"}}))
    with pytest.raises(ExternalProviderError, match="provider_model_unavailable"):
        asyncio.run(remote_generate(profile(), [{"role": "user", "content": "hi"}], 128))
    assert not [r for r in fake.requests if r.url.path == "/v1/messages"]


def test_capability_lookup_outage_does_not_block_inference(http):
    fake = http(router(ok_message(), model_status=503,
                       model_record={"type": "error", "error": {"type": "api_error"}}))
    assert asyncio.run(remote_generate(profile(), [{"role": "user", "content": "hi"}], 128)) == "safe response"
    body = json.loads([r for r in fake.requests if r.url.path == "/v1/messages"][0].content)
    assert "thinking" not in body
    assert body["max_tokens"] == 128 + anthropic.DEFAULT_THINKING_ALLOWANCE


def test_max_tokens_never_exceeds_model_limit(http):
    fake = http(router(ok_message(), model_record=capability_record(disabled=False, max_tokens=1024)))
    asyncio.run(remote_generate(profile(), [{"role": "user", "content": "hi"}], 512))
    body = json.loads([r for r in fake.requests if r.url.path == "/v1/messages"][0].content)
    assert body["max_tokens"] == 1024


def test_endpoint_accepts_root_with_or_without_v1():
    assert anthropic.endpoint("https://api.anthropic.com", "messages") == "https://api.anthropic.com/v1/messages"
    assert anthropic.endpoint("https://api.anthropic.com/v1/", "models") == "https://api.anthropic.com/v1/models"


def test_protocol_selection_uses_trusted_metadata_only():
    assert anthropic.uses_messages_api({"backend": "http", "url": URL})
    assert not anthropic.uses_messages_api({"backend": "http", "url": "https://openrouter.ai/api/v1"})
    assert not anthropic.uses_messages_api({"backend": "pi", "url": URL})
    assert not anthropic.uses_messages_api({"backend": "http", "url": "https://api.anthropic.com.evil.test"})
    with pytest.raises(ValueError):
        anthropic.uses_messages_api({"backend": "http", "url": URL, "api_format": "made-up"})


def test_role_mapping_never_ends_with_assistant_prefill():
    with pytest.raises(ValueError):
        anthropic.split_system([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])
    system, turns = anthropic.split_system([{"role": "system", "content": "only instructions"}])
    assert system == "only instructions"
    assert turns[0]["role"] == "user"


def test_probe_reports_reachable_model(http):
    from services.extensions.probe import remote_probe
    http(router(ok_message("PROVIDER_OK")))
    result = asyncio.run(remote_probe(profile(), max_tokens=64))
    assert result["state"] == "reachable"
    assert result["probe_complete"] is True
    assert result["display_name"] == "Claude Sonnet 5.5"
    assert result["response"] == "PROVIDER_OK"
    assert "api_key" not in json.dumps(result)


def test_probe_maps_authentication_failure(http):
    from services.extensions.probe import ProviderProbeError, remote_probe
    http(router({}, model_status=401,
                model_record={"type": "error", "error": {"type": "authentication_error"}}))
    with pytest.raises(ProviderProbeError) as raised:
        asyncio.run(remote_probe(profile(), max_tokens=64))
    assert raised.value.code == "provider_authentication_failed"
    assert raised.value.provider_status == 401


def test_list_models_returns_public_identifiers_only(http):
    def handle(request):
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [
            {"id": "claude-sonnet-5-5", "display_name": "Claude Sonnet 5.5", "lifecycle": "active"},
            {"id": "claude-old", "display_name": "Old", "lifecycle": "retired"},
        ], "has_more": False})
    http(handle)
    assert asyncio.run(anthropic.list_models(URL, "unit-test-anthropic-key")) == [
        {"id": "claude-sonnet-5-5", "name": "Claude Sonnet 5.5"}]


def test_connection_preset_and_options_are_exposed():
    from services.extensions.options import inference_option_support, validate_model_options
    from services.extensions.providers import connection_presets, get_provider_preset

    preset = get_provider_preset("anthropic")
    assert preset["url"] == URL and preset["requires_api_key"] is True
    assert any(row["id"] == "anthropic-http" for row in connection_presets())
    assert any(row.get("url") == URL and "thinking_modes" in row for row in inference_option_support())
    assert validate_model_options(profile(reasoning_effort="high", thinking_mode="disabled")) == {
        "reasoning_effort": "high", "thinking_mode": "disabled"}
    with pytest.raises(ValueError):
        validate_model_options({"backend": "http", "url": "https://openrouter.ai/api/v1",
                                "model": "x", "thinking_mode": "adaptive"})


def test_custom_gateway_uses_messages_api_only_when_declared(http, monkeypatch):
    monkeypatch.setenv("EXTENSION_ALLOWED_ORIGINS", "https://claude-gateway.example.test")
    gateway = {**profile(), "url": "https://claude-gateway.example.test/v1", "api_format": "anthropic"}
    assert anthropic.uses_messages_api(gateway)
    assert not anthropic.uses_messages_api({**gateway, "api_format": "openai"})
    fake = http(router(ok_message()))
    assert asyncio.run(remote_generate(gateway, [{"role": "user", "content": "hi"}], 64)) == "safe response"
    paths = [r.url.host + r.url.path for r in fake.requests]
    assert "claude-gateway.example.test/v1/messages" in paths


def test_saved_connection_records_protocol_and_hides_credential(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from services.extensions import service
    from services.extensions.store import ExtensionStore

    store = ExtensionStore(tmp_path)
    monkeypatch.setattr(service, "extension_store", lambda: store)
    saved = asyncio.run(service.command("alice", "model", {
        "name": "Claude", "backend": "http", "model": "claude-sonnet-5-5",
        "url": URL, "api_format": "anthropic", "api_key": "unit-test-anthropic-key",
        "reasoning_effort": "medium", "thinking_mode": "disabled",
    }, SimpleNamespace(model_profiles={})))
    assert saved["api_format"] == "anthropic"
    assert saved["reasoning_effort"] == "medium" and saved["thinking_mode"] == "disabled"
    assert "api_key" not in saved and saved["credential_configured"] is True
    cleared = asyncio.run(service.command("alice", "model_options", {
        "id": saved["id"], "thinking_mode": "", "reasoning_effort": None,
    }, SimpleNamespace(model_profiles={})))
    assert "thinking_mode" not in cleared and "reasoning_effort" not in cleared
    with pytest.raises(ValueError):
        asyncio.run(service.command("alice", "model", {
            "name": "Bad", "backend": "http", "model": "m", "url": URL,
            "api_format": "graphql", "api_key": "k"}, SimpleNamespace(model_profiles={})))
