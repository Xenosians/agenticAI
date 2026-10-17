import asyncio
import json
from types import SimpleNamespace

import pytest

from services.extensions.admission import ProviderAdmission, ProviderAdmissionError, validate_admission_settings
from services.extensions.options import reasoning_effort
from services.extensions.store import ExtensionStore


def profile(backend="http", **options):
    return {
        "backend": backend,
        "url": "https://api.groq.com/openai/v1",
        "model": "openai/gpt-oss-120b" if backend == "http" else "groq/openai/gpt-oss-120b",
        "api_key": "private-test-credential",
        **options,
    }


def test_http_pi_and_opencode_share_quota_and_unconfigured_cannot_bypass():
    limiter = ProviderAdmission()

    async def run():
        await limiter.admit(profile(provider_tokens_per_minute=100, admission_max_wait_seconds=0), 60)
        for backend in ("pi", "opencode"):
            with pytest.raises(ProviderAdmissionError, match="wait_exceeded"):
                await limiter.admit(profile(backend, admission_max_wait_seconds=0), 50)
        # Different credentials and models are separate accounting identities.
        await limiter.admit(profile(api_key="another-test-credential", provider_tokens_per_minute=100), 60)
        await limiter.admit(profile(model="another-model", provider_tokens_per_minute=100), 60)

    asyncio.run(run())
    assert len(limiter._buckets) == 3
    assert "private-test-credential" not in repr(limiter._buckets)
    assert all(len(key) == 64 for key in limiter._buckets)


def test_conflicting_profile_uses_lower_limit_and_does_not_reset_reservations():
    limiter = ProviderAdmission()

    async def run():
        await limiter.admit(profile(provider_tokens_per_minute=100), 60)
        with pytest.raises(ProviderAdmissionError, match="wait_exceeded"):
            await limiter.admit(profile("pi", provider_tokens_per_minute=70, admission_max_wait_seconds=0), 20)
        with pytest.raises(ProviderAdmissionError, match="wait_exceeded"):
            await limiter.admit(profile("opencode", provider_tokens_per_minute=1000, admission_max_wait_seconds=0), 20)
        with pytest.raises(ProviderAdmissionError, match="too_large"):
            await limiter.admit(profile(), 71)

    asyncio.run(run())


def test_atomic_concurrent_reservations_cannot_oversubscribe():
    limiter = ProviderAdmission()

    async def run():
        options = profile(provider_tokens_per_minute=100, admission_max_wait_seconds=0)
        results = await asyncio.gather(*(limiter.admit(options, 40) for _ in range(10)), return_exceptions=True)
        assert sum(not isinstance(item, Exception) for item in results) == 2
        assert all(isinstance(item, ProviderAdmissionError) for item in results[2:])

    asyncio.run(run())


def test_reported_usage_only_increases_reservation_and_blocks_other_adapters():
    limiter = ProviderAdmission()

    async def run():
        receipt = await limiter.admit(profile(provider_tokens_per_minute=100), 40)
        limiter.charge_usage(receipt, 80)
        for value in (20, None, "90", True, -1):
            limiter.charge_usage(receipt, value)
        with pytest.raises(ProviderAdmissionError, match="wait_exceeded"):
            await limiter.admit(profile("pi", admission_max_wait_seconds=0), 21)
        await limiter.admit(profile("opencode", admission_max_wait_seconds=0), 20)
        assert next(iter(limiter._buckets.values())).token_total == 100

    asyncio.run(run())


def test_fifo_waits_for_window_and_records_wait_time():
    limiter = ProviderAdmission(window_seconds=0.02)

    async def run():
        options = profile(provider_requests_per_minute=1, admission_max_wait_seconds=1)
        await limiter.admit(options, 1)
        order = []

        async def request(number):
            receipt = await limiter.admit(options, 1)
            order.append(number)
            assert receipt.wait_seconds > 0

        await asyncio.gather(request(1), request(2), request(3))
        assert order == [1, 2, 3]

    asyncio.run(run())


def test_pending_cancellation_releases_queue_slot_without_refunding_dispatched_request():
    now = [100.0]
    limiter = ProviderAdmission(max_waiters=1, clock=lambda: now[0])

    async def run():
        options = profile(provider_requests_per_minute=1)
        await limiter.admit(options, 1)
        pending = asyncio.create_task(limiter.admit(options, 1))
        await asyncio.sleep(0)
        with pytest.raises(ProviderAdmissionError, match="queue_full"):
            await limiter.admit(options, 1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert not next(iter(limiter._buckets.values())).waiters
        with pytest.raises(ProviderAdmissionError, match="wait_exceeded"):
            await limiter.admit(profile(admission_max_wait_seconds=0), 1)
        now[0] += 60
        await limiter.admit(options, 1)

    asyncio.run(run())


def test_delayed_event_loop_cannot_dispatch_after_wait_deadline(monkeypatch):
    now = [100.0]
    limiter = ProviderAdmission(clock=lambda: now[0])

    async def delayed_sleep(_):
        # The provider window has cleared, but this caller's shorter deadline
        # also expired while the event loop was unavailable.
        now[0] += 60

    monkeypatch.setattr(asyncio, "sleep", delayed_sleep)

    async def run():
        await limiter.admit(profile(provider_requests_per_minute=1), 1)
        with pytest.raises(ProviderAdmissionError, match="wait_exceeded"):
            await limiter.admit(profile(admission_max_wait_seconds=5), 1)
        assert not next(iter(limiter._buckets.values())).reservations

    asyncio.run(run())


def test_limiter_state_is_bounded_and_expired_identities_can_be_reclaimed():
    now = [100.0]
    limiter = ProviderAdmission(max_identities=1, clock=lambda: now[0])

    async def run():
        await limiter.admit(profile(), 10)
        with pytest.raises(ProviderAdmissionError, match="capacity_exceeded"):
            await limiter.admit(profile(api_key="different"), 10)
        now[0] += 60
        await limiter.admit(profile(api_key="different"), 10)

    asyncio.run(run())


@pytest.mark.parametrize("options", [
    {"provider_tokens_per_minute": True}, {"provider_tokens_per_minute": "8000"},
    {"provider_tokens_per_minute": 0}, {"provider_requests_per_minute": 1.5},
    {"provider_requests_per_minute": 1_000_001}, {"admission_max_wait_seconds": True},
    {"admission_max_wait_seconds": float("nan")}, {"admission_max_wait_seconds": float("inf")},
    {"admission_max_wait_seconds": -1}, {"admission_max_wait_seconds": 121},
])
def test_invalid_settings_are_rejected(options):
    with pytest.raises(ValueError):
        validate_admission_settings(options)


@pytest.mark.parametrize("backend", ["http", "pi"])
@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_explicit_supported_groq_reasoning_effort(backend, effort):
    assert reasoning_effort(profile(backend, reasoning_effort=effort)) == effort


@pytest.mark.parametrize("options", [
    profile(reasoning_effort="none"), profile(reasoning_effort=True),
    profile(model="unsupported-model", reasoning_effort="low"),
    profile("opencode", reasoning_effort="low"),
    profile(url="https://openrouter.ai/api/v1", reasoning_effort="low"),
])
def test_unverified_reasoning_mappings_fail_closed(options):
    with pytest.raises(ValueError):
        reasoning_effort(options)


def test_owner_model_options_patch_preserves_credential_identity_and_selection(tmp_path):
    store = ExtensionStore(tmp_path)
    data = {key: value for key, value in profile().items() if key != "api_key"}
    saved = store.put("alice", "model", {**data, "state": "ready"}, "encrypted-test-key")
    store.select("alice", saved["id"], None)
    with store.connect() as db:
        before = bytes(db.execute("SELECT secret FROM resource WHERE owner=? AND id=?", ("alice", saved["id"])).fetchone()[0])
    updated = store.update_model_options("alice", saved["id"], {"provider_tokens_per_minute": 8000, "reasoning_effort": "low"})
    assert updated["id"] == saved["id"]
    assert updated["provider_tokens_per_minute"] == 8000
    assert "api_key" not in updated
    assert store.selected("alice")["model_id"] == saved["id"]
    assert store.get("alice", saved["id"], secret=True)["api_key"] == "encrypted-test-key"
    with store.connect() as db:
        after = bytes(db.execute("SELECT secret FROM resource WHERE owner=? AND id=?", ("alice", saved["id"])).fetchone()[0])
    assert before == after
    updated = store.update_model_options("alice", saved["id"], {"provider_tokens_per_minute": None, "reasoning_effort": ""})
    assert "provider_tokens_per_minute" not in updated and "reasoning_effort" not in updated
    with pytest.raises(ValueError):
        store.update_model_options("bob", saved["id"], {"provider_tokens_per_minute": 8000})
    for changes in ({"api_key": "replace"}, {"backend": "local"}, {"reasoning_effort": "none"}):
        with pytest.raises(ValueError):
            store.update_model_options("alice", saved["id"], changes)


def test_http_failure_retains_reservation_and_other_harness_cannot_bypass(monkeypatch, tmp_path):
    from services.extensions import admission, harness, transport
    limiter = ProviderAdmission()
    monkeypatch.setattr(admission, "provider_admission", limiter)
    monkeypatch.setattr(harness, "harness_binary", lambda _: tmp_path / "harness")
    monkeypatch.setattr(harness, "_resolve_node_runtime", lambda: tmp_path / "node")
    calls = []

    async def failed_request(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["payload"]["reasoning_effort"] == "low"
        assert kwargs["payload"]["max_tokens"] == 128
        raise ValueError("provider_connection_failed")

    async def no_child(*args, **kwargs):
        raise AssertionError("Admission must reject before launching harness")

    monkeypatch.setattr(transport, "request_json", failed_request)
    monkeypatch.setattr(harness.asyncio, "create_subprocess_exec", no_child)

    async def run():
        with pytest.raises(ValueError, match="connection_failed"):
            await transport.remote_generate(profile(provider_requests_per_minute=1, reasoning_effort="low"), [], 128)
        for backend in ("pi", "opencode"):
            with pytest.raises(ProviderAdmissionError, match="wait_exceeded"):
                await harness.harness_generate(profile(backend, admission_max_wait_seconds=0), [], 128)

    asyncio.run(run())
    assert len(calls) == 1


def test_model_options_action_is_owner_scoped_and_rejects_other_fields(tmp_path, monkeypatch):
    from services.extensions import service
    store = ExtensionStore(tmp_path)
    monkeypatch.setattr(service, "extension_store", lambda: store)
    data = {key: value for key, value in profile().items() if key != "api_key"}
    saved = store.put("alice", "model", {**data, "state": "ready"}, "test-key")

    async def run():
        result = await service.command("alice", "model_options", {"id": saved["id"], "admission_max_wait_seconds": 0}, SimpleNamespace())
        assert result["admission_max_wait_seconds"] == 0
        assert "api_key" not in result
        with pytest.raises(ValueError):
            await service.command("bob", "model_options", {"id": saved["id"], "reasoning_effort": "low"}, SimpleNamespace())
        with pytest.raises(ValueError):
            await service.command("alice", "model_options", {"id": saved["id"], "api_key": "replacement"}, SimpleNamespace())

    asyncio.run(run())


def test_pi_explicit_effort_reaches_isolated_cli_without_enabling_tools(monkeypatch, tmp_path):
    from services.extensions import admission, harness
    monkeypatch.setattr(admission, "provider_admission", ProviderAdmission())
    monkeypatch.setattr(harness, "harness_binary", lambda _: tmp_path / "harness")
    monkeypatch.setattr(harness, "_resolve_node_runtime", lambda: tmp_path / "node")

    async def checked_child(*args, **kwargs):
        assert args[args.index("--thinking") + 1] == "low"
        assert all(flag in args for flag in ("--no-tools", "--no-mcp", "--no-extensions", "--no-context-files"))
        assert "private-test-credential" not in args
        assert kwargs["env"]["GROQ_API_KEY"] == "private-test-credential"
        from pathlib import Path
        catalog = json.loads((Path(kwargs["env"]["PI_CODING_AGENT_DIR"]) / "models.json").read_text())
        assert catalog["providers"]["groq"]["models"][0]["reasoning"] is True
        settings = json.loads((Path(kwargs["env"]["PI_CODING_AGENT_DIR"]) / "settings.json").read_text())
        assert settings == {"retry": {"enabled": False, "maxRetries": 0, "provider": {"maxRetries": 0}}}
        raise RuntimeError("checked isolated launch arguments")

    monkeypatch.setattr(harness.asyncio, "create_subprocess_exec", checked_child)
    with pytest.raises(RuntimeError, match="checked isolated"):
        asyncio.run(harness.harness_generate(profile("pi", reasoning_effort="low"), [], 128))


@pytest.mark.parametrize("adapter", ["http", "pi", "probe"])
def test_reported_usage_is_charged_before_next_request(monkeypatch, tmp_path, adapter):
    from services.extensions import admission, harness, probe, transport
    limiter = ProviderAdmission()
    monkeypatch.setattr(admission, "provider_admission", limiter)
    monkeypatch.setattr(harness, "harness_binary", lambda _: tmp_path / "harness")
    monkeypatch.setattr(harness, "_resolve_node_runtime", lambda: tmp_path / "node")

    async def response(*args, **kwargs):
        return {"choices": [{"message": {"content": "Status evidence"}, "finish_reason": "stop"}], "usage": {"total_tokens": 900}}

    async def child(*args, **kwargs):
        event = {"type": "message_end", "message": {
            "role": "assistant", "stopReason": "stop", "usage": {"totalTokens": 900},
            "content": [{"type": "text", "text": "Status evidence"}],
        }}
        reader = asyncio.StreamReader()
        reader.feed_data(json.dumps(event).encode() + b"\n")
        reader.feed_eof()

        async def nothing():
            pass

        async def exited():
            return 0

        return SimpleNamespace(stdin=SimpleNamespace(write=lambda _: None, drain=nothing, close=lambda: None),
                               stdout=reader, returncode=0, wait=exited)

    monkeypatch.setattr(transport, "request_json", response)
    monkeypatch.setattr(probe, "request_json", response)
    monkeypatch.setattr(harness.asyncio, "create_subprocess_exec", child)

    async def run():
        if adapter == "pi":
            assert await harness.harness_generate(profile("pi", provider_tokens_per_minute=1000), [], 128) == "Status evidence"
        elif adapter == "http":
            assert await transport.remote_generate(profile(provider_tokens_per_minute=1000), [], 128) == "Status evidence"
        else:
            await probe.remote_probe(profile(provider_tokens_per_minute=1000), max_tokens=128)
        assert next(iter(limiter._buckets.values())).token_total == 900
        with pytest.raises(ProviderAdmissionError, match="wait_exceeded"):
            await limiter.admit(profile("opencode", admission_max_wait_seconds=0), 101)

    asyncio.run(run())


def test_inference_option_catalog_matches_admitted_profiles_and_is_detached(tmp_path, monkeypatch):
    from services.extensions import service
    from services.extensions.options import inference_option_support
    store = ExtensionStore(tmp_path)
    monkeypatch.setattr(service, "extension_store", lambda: store)
    monkeypatch.setattr(service, "harness_binary", lambda _: tmp_path / "harness")
    settings = SimpleNamespace(model_profiles={})

    async def run():
        catalog = await service.command("alice", "list", {}, settings)
        rows = catalog["inference_option_support"]
        endpoint_wide = [row for row in rows if "model" not in row]
        assert [row["url"] for row in endpoint_wide] == ["https://api.anthropic.com"]
        assert endpoint_wide[0]["thinking_modes"] == ["adaptive", "disabled"]
        support = [row for row in rows if "model" in row]
        assert len(support) == 4
        assert {row["backend"] for row in support} == {"http", "pi"}
        for index, row in enumerate(support):
            assert row["reasoning_efforts"] == ["low", "medium", "high"]
            saved = await service.command("alice", "model", {
                "name": f"Supported reasoning connection {index}",
                "backend": row["backend"], "model": row["model"], "url": row["url"],
                "api_key": "catalog-roundtrip-test-credential", "reasoning_effort": "low",
            }, settings)
            assert saved["reasoning_effort"] == "low"
            assert "api_key" not in saved
            assert row["url"] == ("https://api.groq.com/openai/v1" if row["backend"] == "http" else "")
        listing = await service.command("alice", "list", {}, settings)
        assert len(listing["resources"]) == 4
        assert listing["selected"] == {"model_id": None, "dataset_id": None}
        support[0]["reasoning_efforts"].append("unsupported")
        support[0]["model"] = "another-model"
        assert "unsupported" not in inference_option_support()[0]["reasoning_efforts"]
        assert inference_option_support()[0]["model"] != "another-model"

    asyncio.run(run())
