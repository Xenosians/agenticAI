"""Anthropic-format gateways (for example 9router): model records, plain-HTTP opt-in, labels."""

import asyncio

import httpx
import pytest

from services.extensions import anthropic
from services.extensions.budget import external_provider
from services.extensions.transport import insecure_http_origins, is_insecure_transport, remote_generate, validate_url

GATEWAY = "http://198.51.100.7:20128"
_REAL_ASYNC_CLIENT = httpx.AsyncClient

# Shape returned by a live 9router /v1/models/{id} (values abbreviated).
SONNET_RECORD = {
    "id": "cc/claude-sonnet-5-5", "object": "model", "owned_by": "cc",
    "capabilities": {"tools": True, "reasoning": True, "thinkingFormat": "claude-adaptive",
                     "thinkingCanDisable": True, "thinkingEffortSupported": False, "maxOutput": 128000},
    "context_length": 1000000, "max_completion_tokens": 128000,
}
OPUS_RECORD = {**SONNET_RECORD, "id": "cc/claude-opus-5-5",
               "capabilities": {**SONNET_RECORD["capabilities"], "thinkingCanDisable": False}}


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    anthropic._capability_cache.clear()
    monkeypatch.delenv("EXTENSION_INSECURE_HTTP_ORIGINS", raising=False)
    monkeypatch.delenv("EXTENSION_ALLOWED_ORIGINS", raising=False)
    yield
    anthropic._capability_cache.clear()


def allow_gateway(monkeypatch):
    monkeypatch.setenv("EXTENSION_ALLOWED_ORIGINS", GATEWAY)
    monkeypatch.setenv("EXTENSION_INSECURE_HTTP_ORIGINS", GATEWAY)


def gateway_profile(model="cc/claude-sonnet-5-5", **extra):
    return {"backend": "http", "url": GATEWAY + "/v1", "api_format": "anthropic",
            "provider_id": "9router", "model": model, "api_key": "unit-test-gateway-key", **extra}


def test_gateway_record_maps_to_anthropic_capabilities():
    record = anthropic.normalize_model_record(SONNET_RECORD)
    assert record["max_tokens"] == 128000
    assert record["capabilities"]["thinking"]["types"]["disabled"]["supported"] is True
    assert record["capabilities"]["effort"]["supported"] is False
    assert anthropic.normalize_model_record(OPUS_RECORD)["capabilities"]["thinking"]["types"]["disabled"] == {
        "supported": False}


def test_anthropic_records_and_unknown_facts_are_left_alone():
    official = {"id": "claude-sonnet-5-5", "max_tokens": 64000, "capabilities": {"thinking": {"supported": True}}}
    assert anthropic.normalize_model_record(official) is official
    bare = anthropic.normalize_model_record({"id": "x", "capabilities": {"vision": True}})
    assert "thinking" not in bare["capabilities"] and "max_tokens" not in bare


def test_request_plan_disables_thinking_only_when_gateway_allows_it(monkeypatch):
    allow_gateway(monkeypatch)
    for record, expected in ((SONNET_RECORD, "disabled"), (OPUS_RECORD, None)):
        anthropic._capability_cache.clear()

        async def fake_get_model(url, model, api_key, record=record):
            return record

        monkeypatch.setattr(anthropic, "get_model", fake_get_model)
        plan = asyncio.run(anthropic.request_plan(gateway_profile(), None))
        assert plan.thinking == expected
        assert plan.max_tokens_limit == 128000
    with pytest.raises(ValueError):
        asyncio.run(anthropic.request_plan(gateway_profile(), "low"))


def test_plain_http_requires_explicit_exact_origin(monkeypatch):
    monkeypatch.setenv("EXTENSION_ALLOWED_ORIGINS", GATEWAY)
    with pytest.raises(ValueError, match="HTTPS"):
        validate_url(GATEWAY + "/v1")
    monkeypatch.setenv("EXTENSION_INSECURE_HTTP_ORIGINS", GATEWAY)
    assert validate_url(GATEWAY + "/v1/") == GATEWAY + "/v1"
    assert is_insecure_transport(GATEWAY + "/v1")
    assert not is_insecure_transport("https://api.anthropic.com")
    assert not is_insecure_transport("http://127.0.0.1:11434/v1")
    with pytest.raises(ValueError):
        validate_url("http://198.51.100.7:20129/v1")   # other port
    with pytest.raises(ValueError):
        validate_url("http://198.51.100.8:20128/v1")   # other host


def test_insecure_origin_must_also_be_allowed(monkeypatch):
    monkeypatch.setenv("EXTENSION_INSECURE_HTTP_ORIGINS", GATEWAY)
    with pytest.raises(ValueError, match="EXTENSION_ALLOWED_ORIGINS"):
        validate_url(GATEWAY + "/v1")


@pytest.mark.parametrize("entry", ["https://198.51.100.7", "http://198.51.100.7/v1", "198.51.100.7:20128",
                                   "http://user:pw@198.51.100.7"])
def test_insecure_origin_entries_must_be_exact_http_origins(monkeypatch, entry):
    monkeypatch.setenv("EXTENSION_INSECURE_HTTP_ORIGINS", entry)
    with pytest.raises(ValueError):
        insecure_http_origins()


def test_provider_id_labels_budget_and_logs():
    assert external_provider(gateway_profile(), backend="http") == "9router"
    assert external_provider({**gateway_profile(), "provider_id": "bad id!"}, backend="http") == "custom"
    assert external_provider({"backend": "http", "url": "https://api.anthropic.com"}, backend="http") == "anthropic"


def test_gateway_inference_requests_json_and_uses_gateway_records(monkeypatch):
    allow_gateway(monkeypatch)
    seen = []

    def handler(request):
        seen.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=SONNET_RECORD)
        return httpx.Response(200, json={
            "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
            "content": [{"type": "text", "text": "safe response"}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 2025, "output_tokens": 3}})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _REAL_ASYNC_CLIENT(
        *a, **{**k, "transport": transport}))
    text = asyncio.run(remote_generate(gateway_profile(), [{"role": "user", "content": "hi"}], 64))
    assert text == "safe response"
    post = next(r for r in seen if r.method == "POST")
    assert post.url.path == "/v1/messages"
    assert post.headers["accept"] == "application/json"
    assert b'"thinking":{"type":"disabled"}' in post.content.replace(b" ", b"")


def test_saved_gateway_connection_is_flagged_insecure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from services.extensions import service
    from services.extensions.store import ExtensionStore

    allow_gateway(monkeypatch)
    store = ExtensionStore(tmp_path)
    monkeypatch.setattr(service, "extension_store", lambda: store)
    settings = SimpleNamespace(model_profiles={})
    saved = asyncio.run(service.command("alice", "model", {
        "name": "Claude via 9router", "backend": "http", "model": "cc/claude-sonnet-5-5",
        "url": GATEWAY + "/v1", "api_format": "anthropic", "provider_id": "9Router",
        "api_key": "unit-test-gateway-key"}, settings))
    assert saved["provider_id"] == "9router" and saved["insecure_transport"] is True
    assert "api_key" not in saved
    listed = asyncio.run(service.command("alice", "list", {}, settings))
    row = next(r for r in listed["resources"] if r["id"] == saved["id"])
    assert row["insecure_transport"] is True and "api_key" not in row
    with pytest.raises(ValueError, match="provider_id"):
        asyncio.run(service.command("alice", "model", {
            "name": "x", "backend": "http", "model": "m", "url": GATEWAY + "/v1",
            "provider_id": "../etc", "api_key": "k"}, settings))


def test_anthropic_format_gateways_use_the_anthropic_envelope():
    from services.extensions.budget import default_external_request_budget

    official = default_external_request_budget({"backend": "http", "url": "https://api.anthropic.com"}, backend="http")
    assert default_external_request_budget(gateway_profile(), backend="http") == official
    assert default_external_request_budget({**gateway_profile(), "api_format": "openai"}, backend="http") < official


def test_long_system_prompts_are_cacheable_short_ones_are_plain():
    long_body = anthropic.build_request(gateway_profile(), [
        {"role": "system", "content": "catalog " * 1000}, {"role": "user", "content": "hi"}], 64)
    assert long_body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert long_body["system"][0]["text"].startswith("catalog")
    short_body = anthropic.build_request(gateway_profile(), [
        {"role": "system", "content": "short"}, {"role": "user", "content": "hi"}], 64)
    assert short_body["system"] == "short"
