"""Streamed answer text (SRS 1.9 FR-080): previews only, the durable answer is unchanged.

The Messages API path streams the answer-writing steps when a job's progress
channel is open. Every check a complete response gets still applies to the
streamed one, the final text still crosses the Lua response phase, and a
preview can never become, extend or replace the answer.
"""

import asyncio
import json

import httpx
import pytest

from services.extensions import anthropic
from services.extensions.diagnostics import ExternalProviderError
from services.extensions.transport import remote_generate
from services.job_progress import JobProgress, answer_stream
from services.lua_adapters.plugins import anthropic_request
from services.lua_adapters.runtime import LuaAdapterError


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


def record():
    return {"id": "claude-sonnet-5-5", "max_tokens": 128_000, "capabilities": {
        "thinking": {"supported": True, "types": {"disabled": {"supported": True}}}}}


def sse(*events):
    lines = []
    for event in events:
        lines.append(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n")
    return "".join(lines).encode()


def text_stream(*parts, stop_reason="end_turn", extra=()):
    return sse(
        {"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant",
                                               "content": [], "stop_reason": None,
                                               "usage": {"input_tokens": 40, "output_tokens": 1}}},
        *extra,
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        *({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": part}}
          for part in parts),
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": stop_reason}, "usage": {"output_tokens": 9}},
        {"type": "message_stop"},
    )


@pytest.fixture
def http(monkeypatch):
    def install(stream_body, *, status=200):
        requests = []

        def handle(request):
            requests.append(request)
            if request.url.path.startswith("/v1/models/"):
                return httpx.Response(200, json=record())
            body = json.loads(request.content)
            if body.get("stream") is True:
                return httpx.Response(status, content=stream_body,
                                      headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json={
                "type": "message", "role": "assistant", "content": [{"type": "text", "text": "complete"}],
                "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}})

        def client(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handle)
            return _REAL_ASYNC_CLIENT(*args, **kwargs)

        monkeypatch.setattr(anthropic.httpx, "AsyncClient", client)
        return requests
    return install


class Collector:
    def __init__(self):
        self.parts = []

    def __call__(self, text):
        self.parts.append(text)


def messages_posts(requests):
    return [json.loads(r.content) for r in requests if r.url.path == "/v1/messages"]


def test_a_stream_returns_the_same_answer_and_shows_each_part_first(http):
    requests = http(text_stream("The VPN ", "is up."))
    preview = Collector()
    with answer_stream(preview):
        answer = asyncio.run(remote_generate(profile(), [{"role": "user", "content": "Is the VPN up?"}], 128))
    assert answer == "The VPN is up."
    assert preview.parts == ["The VPN ", "is up."]
    (body,) = messages_posts(requests)
    assert body["stream"] is True
    for forbidden in ("tools", "tool_choice", "mcp_servers", "container"):
        assert forbidden not in body


def test_without_a_listener_the_request_is_not_streamed(http):
    requests = http(text_stream("unused"))
    assert asyncio.run(remote_generate(profile(), [{"role": "user", "content": "hi"}], 128)) == "complete"
    (body,) = messages_posts(requests)
    assert body["stream"] is False


@pytest.mark.parametrize("stop_reason, category", [
    ("max_tokens", "provider_output_truncated"),
    ("refusal", "provider_refused"),
    ("tool_use", "provider_unexpected_tool_use"),
    ("something_new", "provider_incomplete_turn"),
])
def test_a_streamed_answer_keeps_every_stop_reason_check(http, stop_reason, category):
    http(text_stream("partial", stop_reason=stop_reason))
    with answer_stream(Collector()), pytest.raises(ExternalProviderError) as raised:
        asyncio.run(remote_generate(profile(), [{"role": "user", "content": "hi"}], 128))
    assert raised.value.category == category


def test_a_stream_that_ends_early_or_carries_a_tool_block_fails_closed(http):
    cut = text_stream("never finished")
    cut = cut[:cut.index(b"event: message_delta")]
    http(cut)
    with answer_stream(Collector()), pytest.raises(ExternalProviderError, match="provider_incomplete_turn"):
        asyncio.run(remote_generate(profile(), [{"role": "user", "content": "hi"}], 128))
    tool = {"type": "content_block_start", "index": 1,
            "content_block": {"type": "tool_use", "id": "t", "name": "x", "input": {}}}
    http(text_stream("text", extra=[tool]))
    with answer_stream(Collector()), pytest.raises(ExternalProviderError, match="provider_unexpected_tool_use"):
        asyncio.run(remote_generate(profile(), [{"role": "user", "content": "hi"}], 128))


def test_hidden_reasoning_never_reaches_the_preview(http):
    thinking = [{"type": "content_block_start", "index": 1, "content_block": {"type": "thinking", "thinking": ""}},
                {"type": "content_block_delta", "index": 1,
                 "delta": {"type": "thinking_delta", "thinking": "secret plan"}}]
    http(text_stream("Visible.", extra=thinking))
    preview = Collector()
    with answer_stream(preview):
        assert asyncio.run(remote_generate(profile(), [{"role": "user", "content": "hi"}], 128)) == "Visible."
    assert preview.parts == ["Visible."]


def test_a_provider_error_event_and_an_empty_stream_fail(http):
    http(sse({"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
             {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}))
    with answer_stream(Collector()), pytest.raises(ExternalProviderError, match="provider_overloaded"):
        asyncio.run(remote_generate(profile(), [{"role": "user", "content": "hi"}], 128))
    http(text_stream("   "))
    with answer_stream(Collector()), pytest.raises(ExternalProviderError, match="provider_empty_response"):
        asyncio.run(remote_generate(profile(), [{"role": "user", "content": "hi"}], 128))


def test_a_failing_listener_never_changes_the_answer(http):
    http(text_stream("Still ", "answered."))

    def broken(_text):
        raise RuntimeError("socket gone")

    with answer_stream(broken):
        answer = asyncio.run(remote_generate(profile(), [{"role": "user", "content": "hi"}], 128))
    assert answer == "Still answered."


def test_the_lua_codec_must_keep_the_requested_stream_flag():
    messages = [{"role": "user", "content": "hi"}]
    assert anthropic_request(profile(), messages, 64, stream=True)["stream"] is True
    assert anthropic_request(profile(), messages, 64)["stream"] is False
    with pytest.raises(LuaAdapterError):
        anthropic_request(profile(), messages, 64, stream="yes")


def test_answer_text_frames_are_bounded_ordered_and_only_for_the_current_attempt():
    publisher = JobProgress("https://example.com", "private-token", "a" * 32, 2)
    publisher.publish_answer("Hello ")
    publisher.publish_answer("world")
    frames = [publisher.answers.get_nowait() for _ in range(publisher.answers.qsize())]
    assert [frame["sequence"] for frame in frames] == [1, 2]
    assert "".join(frame["text"] for frame in frames) == "Hello world"
    publisher.publish_answer("x" * 10_000)
    sizes = []
    while not publisher.answers.empty():
        sizes.append(len(publisher.answers.get_nowait()["text"].encode()))
    assert sizes and max(sizes) <= JobProgress.MAX_ANSWER_FRAME_BYTES
    with pytest.raises(ValueError):
        publisher.publish_answer(123)
