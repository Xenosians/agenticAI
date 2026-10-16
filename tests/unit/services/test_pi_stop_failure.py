import asyncio
import json
from types import SimpleNamespace

import pytest

from services.extensions import harness
from services.extensions.diagnostics import HarnessFailure


def install_pi_response(monkeypatch, tmp_path, message):
    monkeypatch.setattr(harness, "harness_binary", lambda _: tmp_path / "pi")
    monkeypatch.setattr(harness, "_resolve_node_runtime", lambda: tmp_path / "node")

    async def spawn(*args, **options):
        reader = asyncio.StreamReader()
        reader.feed_data(json.dumps({"type": "message_end", "message": message}).encode() + b"\n")
        reader.feed_eof()

        async def drain():
            pass

        async def wait():
            return 0

        return SimpleNamespace(
            stdin=SimpleNamespace(write=lambda _: None, drain=drain, close=lambda: None),
            stdout=reader,
            returncode=0,
            wait=wait,
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)


def test_pi_provider_error_is_classified_and_secret_safe(monkeypatch, tmp_path, capsys):
    api_key = "private-pi-test-credential"
    install_pi_response(monkeypatch, tmp_path, {
        "role": "assistant",
        "stopReason": "error",
        "errorMessage": f"Request too large on tokens per minute. Requested 8100. Key {api_key}",
        "content": [{"type": "text", "text": "private partial model content"}],
    })

    with pytest.raises(HarnessFailure) as caught:
        asyncio.run(harness.harness_generate(
            {"backend": "pi", "model": "groq/example", "api_key": api_key},
            [{"role": "user", "content": "Read status"}],
            128,
        ))

    assert caught.value.category == "provider_token_budget_exceeded"
    assert caught.value.exit_code == 0
    logs = capsys.readouterr().out
    assert "Requested 8100" in logs
    assert "[REDACTED]" in logs
    assert api_key not in logs + str(caught.value)
    assert "private partial model content" not in logs + str(caught.value)
    assert "Requested 8100" not in str(caught.value)


def test_pi_length_stop_rejects_partial_tool_proposal(monkeypatch, tmp_path, capsys):
    partial = '{"tool": "process_exec", "arguments":'
    install_pi_response(monkeypatch, tmp_path, {
        "role": "assistant",
        "stopReason": "length",
        "content": [{"type": "text", "text": partial}],
    })

    with pytest.raises(HarnessFailure) as caught:
        asyncio.run(harness.harness_generate(
            {"backend": "pi", "model": "groq/example", "api_key": "test-key"},
            [{"role": "user", "content": "Read status"}],
            128,
        ))

    assert caught.value.category == "provider_output_truncated"
    logs = capsys.readouterr().out
    assert "stop_reason=length" in logs
    assert partial not in logs + str(caught.value)
