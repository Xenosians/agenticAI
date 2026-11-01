"""Per-connection settings: answer length, request size, timeout, retries and instructions."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest

from services.extensions.diagnostics import ExternalProviderError, HarnessFailure
from services.extensions.options import request_timeout_seconds, validate_model_options
from services.extensions.store import ExtensionStore


def connection(backend="http", **options):
    return {"backend": backend, "url": "https://api.groq.com/openai/v1",
            "model": "openai/gpt-oss-120b" if backend == "http" else "groq/openai/gpt-oss-120b", **options}


@pytest.mark.parametrize("options", [
    {"answer_max_tokens": 32}, {"answer_max_tokens": 1024}, {"request_timeout_seconds": 5},
    {"request_timeout_seconds": 600}, {"request_timeout_seconds": 42.5}, {"retry_attempts": 0}, {"retry_attempts": 3},
    {"retry_wait_seconds": 0}, {"retry_wait_seconds": 10}, {"retry_wait_seconds": 1.5},
])
def test_limits_within_range_are_kept_as_given(options):
    assert validate_model_options(connection(**options)) == options


@pytest.mark.parametrize("options", [
    {"answer_max_tokens": 31}, {"answer_max_tokens": 1025}, {"answer_max_tokens": True},
    {"answer_max_tokens": "512"}, {"answer_max_tokens": 512.0},
    {"request_timeout_seconds": 4.9}, {"request_timeout_seconds": 601}, {"request_timeout_seconds": float("nan")},
    {"request_timeout_seconds": False}, {"retry_attempts": 4}, {"retry_attempts": -1}, {"retry_attempts": 1.0},
    {"retry_wait_seconds": 10.5}, {"retry_wait_seconds": -0.1}, {"retry_wait_seconds": float("inf")},
])
def test_limits_out_of_range_or_of_the_wrong_type_are_refused(options):
    with pytest.raises(ValueError):
        validate_model_options(connection(**options))


@pytest.mark.parametrize("name, value", [
    ("request_timeout_seconds", 60), ("retry_attempts", 3), ("retry_wait_seconds", 3)])
def test_provider_limits_need_an_external_connection(name, value):
    with pytest.raises(ValueError, match="external model connection"):
        validate_model_options({"backend": "local", "model": "hub-main", name: value})


def test_the_answer_length_applies_to_local_models_too():
    local = {"backend": "local", "model": "hub-main", "answer_max_tokens": 256}
    assert validate_model_options(local) == {"answer_max_tokens": 256}


def test_instructions_are_kept_trimmed_for_every_kind_of_connection():
    for profile in (connection(), {"backend": "local", "model": "hub-main"}):
        result = validate_model_options({**profile, "custom_instructions": "  Answer in Bahasa Indonesia.\n"})
        assert result == {"custom_instructions": "Answer in Bahasa Indonesia."}


@pytest.mark.parametrize("text", [
    "x" * 2001, 42, "Reply with \x00 inside", "Use api_key=abcdef123456 for lookups",
    # Assembled here so the repository never holds a key header verbatim.
    "-----BEGIN RSA " + "PRIVATE KEY-----",
])
def test_instructions_that_are_too_long_malformed_or_hold_secrets_are_refused(text):
    with pytest.raises(ValueError):
        validate_model_options({**connection(), "custom_instructions": text})


def test_saved_settings_update_in_place_and_clear_with_empty_values(tmp_path):
    store = ExtensionStore(tmp_path)
    saved = store.put("alice", "model", {**connection(), "state": "ready"}, "encrypted-test-key")
    updated = store.update_model_options("alice", saved["id"], {
        "answer_max_tokens": 512, "request_timeout_seconds": 30, "retry_attempts": 2,
        "retry_wait_seconds": 1.5, "custom_instructions": "  Keep answers short.\n"})
    assert {key: updated[key] for key in ("answer_max_tokens", "request_timeout_seconds", "retry_attempts",
                                          "retry_wait_seconds", "custom_instructions")} == {
        "answer_max_tokens": 512, "request_timeout_seconds": 30, "retry_attempts": 2,
        "retry_wait_seconds": 1.5, "custom_instructions": "Keep answers short."}
    assert store.get("alice", saved["id"], secret=True)["api_key"] == "encrypted-test-key"
    cleared = store.update_model_options("alice", saved["id"], {
        "answer_max_tokens": None, "retry_attempts": None, "custom_instructions": ""})
    assert not {"answer_max_tokens", "retry_attempts", "custom_instructions"} & set(cleared)
    assert cleared["request_timeout_seconds"] == 30
    blank = store.update_model_options("alice", saved["id"], {"custom_instructions": "   "})
    assert "custom_instructions" not in blank
    with pytest.raises(ValueError):
        store.update_model_options("alice", saved["id"], {"retry_attempts": 9})


def test_a_new_connection_saves_its_settings_and_skips_empty_instructions(tmp_path, monkeypatch):
    from services.extensions import service
    store = ExtensionStore(tmp_path)
    monkeypatch.setattr(service, "extension_store", lambda: store)
    payload = {"name": "Groq answers", **connection(), "api_key": "test-credential",
               "answer_max_tokens": 600, "retry_attempts": 1}

    async def run():
        first = await service.command("alice", "model", {**payload, "custom_instructions": " Be brief. "},
                                      SimpleNamespace(model_profiles={}))
        second = await service.command("alice", "model", {**payload, "name": "Groq plain", "custom_instructions": ""},
                                       SimpleNamespace(model_profiles={}))
        return first, second

    first, second = asyncio.run(run())
    assert first["custom_instructions"] == "Be brief." and first["answer_max_tokens"] == 600
    assert "custom_instructions" not in second and second["retry_attempts"] == 1
    assert "api_key" not in first


def test_the_connection_timeout_replaces_only_the_default():
    assert request_timeout_seconds(connection(), 90.0) == 90.0
    assert request_timeout_seconds(connection(request_timeout_seconds=30), 90.0) == 30.0


class FakeResponse:
    status_code = 200

    def __init__(self, body: bytes):
        self.body = body

    async def aiter_bytes(self):
        yield self.body


class FakeClient:
    def __init__(self, outcome):
        self.outcome, self.timeouts = outcome, []

    @asynccontextmanager
    async def stream(self, method, url, **kwargs):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        yield FakeResponse(self.outcome)


def use_client(monkeypatch, fake):
    from services.extensions import transport

    @asynccontextmanager
    async def shared(**options):
        fake.timeouts.append(options["timeout"])
        yield fake

    monkeypatch.setattr(transport, "shared_async_client", shared)


def test_requests_wait_for_the_connection_timeout(monkeypatch):
    from services.extensions import transport
    body = b'{"choices":[{"message":{"content":"Router is up."},"finish_reason":"stop"}],"usage":{"total_tokens":20}}'
    fake = FakeClient(body)
    use_client(monkeypatch, fake)
    messages = [{"role": "user", "content": "Is the router up?"}]
    assert asyncio.run(transport.remote_generate(connection(request_timeout_seconds=30), messages, 64)) == "Router is up."
    assert asyncio.run(transport.remote_generate(connection(), messages, 64)) == "Router is up."
    assert fake.timeouts == [30.0, 90.0]


def test_a_request_that_times_out_is_reported_as_a_timeout(monkeypatch):
    from services.extensions import transport
    use_client(monkeypatch, FakeClient(httpx.ReadTimeout("slow")))
    with pytest.raises(ExternalProviderError) as raised:
        asyncio.run(transport.request_json("https://api.groq.com/openai/v1/chat/completions", payload={}))
    assert raised.value.category == "provider_timeout"


def run_with_retries(profile, failures):
    from services.extensions import retry
    attempts, waits = [], []

    async def send():
        attempts.append(len(attempts))
        if failures:
            raise failures.pop(0)
        return "answer"

    async def sleep(seconds):
        waits.append(seconds)

    original, retry._sleep = retry._sleep, sleep
    try:
        return asyncio.run(retry.with_retries(profile, send)), attempts, waits
    finally:
        retry._sleep = original


@pytest.mark.parametrize("category", [
    "provider_rate_limited", "provider_overloaded", "provider_unavailable", "provider_connection_failed"])
def test_safe_failures_are_retried_with_doubling_waits(category):
    failures = [ExternalProviderError(category), ExternalProviderError(category)]
    result, attempts, waits = run_with_retries(connection(retry_attempts=2, retry_wait_seconds=1.5), failures)
    assert result == "answer" and len(attempts) == 3 and waits == [1.5, 3.0]


def test_harness_failures_follow_the_same_rule():
    failures = [HarnessFailure("pi", "groq", "provider_rate_limited")]
    result, attempts, _ = run_with_retries(connection("pi", retry_attempts=1), failures)
    assert result == "answer" and len(attempts) == 2


@pytest.mark.parametrize("category", [
    "provider_timeout", "provider_authentication_failed", "provider_billing_required",
    "provider_output_truncated", "provider_malformed_response", "provider_model_unavailable"])
def test_failures_that_could_repeat_or_cost_twice_are_not_retried(category):
    with pytest.raises(ExternalProviderError):
        run_with_retries(connection(retry_attempts=3), [ExternalProviderError(category)])


def test_without_retry_settings_a_failure_is_final_and_attempts_stop_at_the_limit():
    with pytest.raises(ExternalProviderError):
        run_with_retries(connection(), [ExternalProviderError("provider_rate_limited")])
    failures = [ExternalProviderError("provider_overloaded") for _ in range(3)]
    with pytest.raises(ExternalProviderError):
        run_with_retries(connection(retry_attempts=2, retry_wait_seconds=0), failures)
    assert len(failures) == 0
