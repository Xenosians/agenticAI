"""Real Lua adapter protocol, provenance, isolation and failure contracts."""
import asyncio
import hashlib
import json
import math
import subprocess
import sys

import pytest

from services.lua_adapters import AdapterLimits, LuaAdapterError, run_adapter, run_adapter_async
from services.lua_adapters import runtime as adapter_runtime

BOUND = {"owner_id": "fixture-owner", "resource_id": "fixture-model", "stage": "CONTROL_ROUTING"}


def invoke(payload=None, *, source=None, kind="model", phase="request", binding=None, **kwargs):
    return run_adapter(kind=kind, phase=phase, payload=payload,
                       binding=BOUND if binding is None else binding, source=source, **kwargs)


def refused(code, *, source=None, payload=None, **kwargs):
    with pytest.raises(LuaAdapterError) as caught:
        invoke(payload, source=source, **kwargs)
    assert caught.value.code == code, caught.value.message
    return caught.value


@pytest.mark.parametrize("kind", ["model", "tool", "connection", "source"])
@pytest.mark.parametrize("phase", ["request", "response", "admission", "warm"])
def test_default_actually_runs_lua_for_each_host_boundary(kind, phase):
    payload = {"messages": [], "nested": [None, {}, [], [False, 0, 1.5, "é"]], "empty": {}, "optional": None}
    before = json.dumps(payload, sort_keys=True)
    result = invoke(payload, kind=kind, phase=phase)
    assert json.dumps(result.payload, sort_keys=True) == before
    assert json.dumps(payload, sort_keys=True) == before
    receipt = result.receipt
    assert receipt["schema"] == "lua-adapter-receipt.v1"
    assert (receipt["kind"], receipt["phase"]) == (kind, phase)
    assert receipt["runtime"] == "lua54-isolated.v1"
    assert receipt["input_sha256"] == receipt["output_sha256"]
    assert receipt["script_sha256"] == hashlib.sha256(adapter_runtime.DEFAULT_SOURCE.read_bytes()).hexdigest()
    assert type(receipt["elapsed_ms"]) is int and receipt["elapsed_ms"] >= 0
    assert {"process", "no_new_privileges", "resource_limits", "no_new_descriptors"} <= set(receipt["isolation"])


def test_custom_lua_translates_a_protocol_and_preserves_explicit_json_types():
    source = """return function(event)
      local texts = adapter.array()
      for _, message in ipairs(event.payload.messages) do
        table.insert(texts, message.content:upper())
      end
      event.payload = adapter.object({prompts = texts, optional = adapter.null,
        empty_array = adapter.array(), empty_object = adapter.object()})
      return event
    end"""
    result = invoke({"messages": [{"role": "user", "content": "vpn drops"}]}, source=source)
    assert result.payload == {"prompts": ["VPN DROPS"], "optional": None, "empty_array": [], "empty_object": {}}
    assert result.receipt["input_sha256"] != result.receipt["output_sha256"]
    assert result.receipt["script_sha256"] == hashlib.sha256(source.encode()).hexdigest()


@pytest.mark.parametrize("change", [
    'event.schema = "other"', 'event.kind = "tool"', 'event.phase = "response"',
    'event.binding.owner_id = "another-owner"', 'event.binding.stage = "SYNTHESIS"',
    'event.binding.resource_id = "another-model"',
])
def test_a_script_cannot_change_the_host_binding(change):
    refused("lua_adapter_binding_changed", source=f"return function(event) {change}; return event end")


def test_binding_boolean_cannot_be_changed_to_numerically_equal_integer():
    refused("lua_adapter_binding_changed", binding={"pinned": True},
            source="return function(event) event.binding.pinned=1; return event end")


@pytest.mark.parametrize("change", [{"binding": {**BOUND, "owner_id": "another-owner"}}, {"phase": "response"},
                                    {"kind": "tool"}, {"schema": "lua-adapter-event.v0"}])
def test_the_host_refuses_a_worker_answer_whose_binding_changed(monkeypatch, change):
    """The host checks the binding itself, independently of the worker's own check."""
    event = {"schema": "lua-adapter-event.v1", "kind": "model", "phase": "request", "binding": BOUND,
             "payload": None, **change}
    answer = {"schema": "lua-adapter-worker.v1", "event": event, "elapsed_ms": 0,
              "isolation": sorted(adapter_runtime.REQUIRED_ISOLATION)}
    monkeypatch.setattr(adapter_runtime.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, json.dumps(answer).encode(), b""))
    refused("lua_adapter_binding_changed")


def test_input_and_output_json_numbers_keep_integer_float_and_negative_zero_types():
    result = invoke({"integer": 1, "float": 1.0, "negative_zero": -0.0, "large": 2**63 - 1})
    assert type(result.payload["integer"]) is int and type(result.payload["float"]) is float
    assert math.copysign(1, result.payload["negative_zero"]) == -1
    assert result.payload["large"] == 2**63 - 1


def test_script_has_only_pure_lua_functions_and_no_host_credentials(monkeypatch):
    monkeypatch.setenv("AI_INTERNAL_JOB_TOKEN", "must-never-enter-worker")
    names = ["io", "os", "require", "dofile", "loadfile", "load", "debug", "package", "python",
             "coroutine", "pcall", "xpcall", "setmetatable", "getmetatable", "print", "_G", "string.dump", "math.random"]
    fields = ", ".join(f'[{json.dumps(name)}] = ({name} ~= nil)' for name in names)
    source = "return function(event) event.payload = {" + fields + "}; return event end"
    result = invoke(source=source)
    assert result.payload == {name: False for name in names}
    assert "must-never-enter-worker" not in json.dumps(result.receipt)
    assert set(adapter_runtime.worker_environment()) == {"PATH", "LANG", "PYTHONDONTWRITEBYTECODE", "PYTHONNOUSERSITE"}


@pytest.mark.parametrize("secret", ["api_key", "API-Key", "authorization", "password", "access_token", "client_secret", "private_key"])
def test_credentials_are_refused_before_starting_a_worker(monkeypatch, secret):
    def dispatch(*_, **__):
        pytest.fail("Credential-bearing input reached a worker")
    monkeypatch.setattr(adapter_runtime.subprocess, "run", dispatch)
    refused("lua_adapter_secret_input", payload={"nested": [{secret: "fixture-secret"}]})


def test_error_messages_never_echo_lua_or_payload_text():
    error = refused("lua_adapter_failed", payload={"text": "private-input"},
                    source='return function(event) error("private-script:" .. event.payload.text) end')
    assert "private" not in error.message and "private" not in str(error)


@pytest.mark.parametrize("source,code", [
    ("return 42", "lua_adapter_contract_invalid"),
    ("return function(event) return {a = } end", "lua_adapter_syntax_error"),
    ("return function(event) return nil end", "lua_adapter_output_invalid"),
    ("return function(event) return 'ok' end", "lua_adapter_output_invalid"),
    ("return function(event) event.payload=function() end; return event end", "lua_adapter_output_invalid"),
    ("return function(event) event.payload={n=0/0}; return event end", "lua_adapter_output_invalid"),
    ("return function(event) event.payload={n=math.huge}; return event end", "lua_adapter_output_invalid"),
    ("return function(event) event.payload={[1]='a',[3]='b'}; return event end", "lua_adapter_output_invalid"),
    ("return function(event) event.payload=adapter.array({a=1}); return event end", "lua_adapter_output_invalid"),
    ("return function(event) event.payload={}; event.payload.loop=event.payload; return event end", "lua_adapter_output_invalid"),
    ("return function(event) adapter.null.hidden='x'; event.payload=adapter.null; return event end", "lua_adapter_output_invalid"),
    ("return function(event) event.payload={authorization='fake'}; return event end", "lua_adapter_output_invalid"),
    ("return function(event) while true do end end", "lua_adapter_instruction_limit"),
])
def test_bad_scripts_do_not_fall_back_to_the_original_request(source, code):
    refused(code, source=source)


def test_output_and_memory_are_bounded_by_the_actual_worker():
    refused("lua_adapter_output_too_large", source="return function(event) event.payload=('x'):rep(10000); return event end",
            limits=AdapterLimits(output_bytes=1000))
    refused("lua_adapter_memory_limit", source="return function(event) event.payload=('x'):rep(10000000); return event end",
            limits=AdapterLimits(memory_bytes=1_000_000))


def test_wall_clock_timeout_cannot_be_reported_as_adapter_success():
    refused("lua_adapter_time_limit", source="return function(event) return event end",
            limits=AdapterLimits(wall_seconds=0.000001))


@pytest.mark.parametrize("payload", [float("nan"), float("inf"), 2**64, {1: "non-string"}, (1, 2), {"x": object()}])
def test_non_json_inputs_are_refused(payload):
    refused("lua_adapter_invalid_input", payload=payload)


@pytest.mark.parametrize("source", ["", "  ", "return function(event) return event end\x00", "\x1bLua", "x" * 20_001])
def test_sources_must_be_bounded_plain_text(source):
    refused("lua_adapter_invalid_source", source=source)


@pytest.mark.parametrize("kind,phase", [(None, "request"), ([], "request"), ("model", {}), ("unknown", "request")])
def test_unknown_dispatch_boundaries_are_refused(kind, phase):
    refused("lua_adapter_invalid_input", kind=kind, phase=phase)


@pytest.mark.parametrize("binding", [{}, {"oversized": "x" * 16_000}])
def test_binding_is_required_and_bounded(binding):
    refused("lua_adapter_binding_invalid", binding=binding)


def test_input_size_limit_is_applied_before_worker_dispatch():
    refused("lua_adapter_input_too_large", payload="x" * adapter_runtime.MAX_INPUT_BYTES)


def test_nesting_limit_counts_payload_independently_of_the_event_envelope():
    value = 0
    for _ in range(16):
        value = {"nested": value}
    assert invoke(value).payload == value
    refused("lua_adapter_invalid_input", payload={"extra": value})


def test_unavailable_worker_never_returns_a_synthetic_receipt(monkeypatch):
    monkeypatch.setattr(adapter_runtime.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 1, b"", b"private-stack"))
    refused("lua_adapter_unavailable")


def test_incomplete_or_malformed_worker_receipts_are_refused(monkeypatch):
    event = {"schema": "lua-adapter-event.v1", "kind": "model", "phase": "request", "binding": BOUND, "payload": None}
    for answer, code in [
        ({"schema": "lua-adapter-worker.v1", "event": event, "elapsed_ms": 0, "isolation": ["process"]}, "lua_adapter_isolation_unavailable"),
        ({"schema": "lua-adapter-worker.v1", "event": event, "elapsed_ms": True,
          "isolation": list(adapter_runtime.REQUIRED_ISOLATION)}, "lua_adapter_output_invalid"),
        ({"error": {"injected": "private"}}, "lua_adapter_unavailable"),
        ({"status": "ok"}, "lua_adapter_unavailable"),
    ]:
        monkeypatch.setattr(adapter_runtime.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 0, json.dumps(answer).encode(), b""))
        refused(code)


def test_async_entrypoint_executes_the_same_real_lua_contract():
    async def exercise():
        return await run_adapter_async(kind="source", phase="request", binding=BOUND, payload={"query": "vpn"},
            source="return function(event) event.payload.query=event.payload.query:upper(); return event end")
    result = asyncio.run(exercise())
    assert result.payload == {"query": "VPN"} and result.receipt["kind"] == "source"


def test_imports_do_not_start_workers_network_or_runtime_storage(tmp_path):
    root = str(adapter_runtime.WORKER.parents[2])
    code = """import asyncio, ssl, socket, subprocess, sys
def refused(*args, **kwargs): raise AssertionError('import started an external resource')
socket.socket = refused
subprocess.Popen = refused
sys.path.insert(0, sys.argv[1])
import services.lua_adapters
print('import_only')
"""
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", code, root], cwd=tmp_path,
                               capture_output=True, text=True, timeout=10,
                               env=adapter_runtime.worker_environment())
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "import_only"
    assert list(tmp_path.iterdir()) == []
