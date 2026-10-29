"""Isolated pure-Lua protocol worker, reusing the dataset worker's OS sandbox.

This file is launched directly with Python isolated mode. Its imports and trusted
helper load finish before lockdown. It accepts one JSON request and never starts
providers, opens credentials or grants execution authority.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import runpy
import sys
import time

from lupa import lua54

# Reuse the established isolation and Lua environment; do not maintain a weaker
# second sandbox. run_name keeps the row worker's main entrypoint inactive.
shared = runpy.run_path(str(Path(__file__).resolve().parents[1] / "extensions" / "transform_worker.py"),
                        run_name="lua_adapter_isolation")
Failure = shared["Failure"]
EVENT_SCHEMA = "lua-adapter-event.v1"
REQUIRED_ISOLATION = {"process", "no_new_privileges", "resource_limits", "no_new_descriptors"}
JSON_SUPPORT = """
function()
  local markers = setmetatable({}, {__mode = "k"})
  local function tagged(kind, value)
    if value == nil then value = {} end
    if type(value) ~= "table" then error("JSON constructor needs a table") end
    markers[value] = kind
    return value
  end
  local null = tagged("null")
  local helpers = {
    array = function(value) return tagged("array", value) end,
    object = function(value) return tagged("object", value) end,
    null = null
  }
  return helpers, function(kind) return tagged(kind) end,
    function(value) return markers[value] end
end
"""


def canonical(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def to_lua(value, new, null):
    if value is None:
        return null
    if type(value) is dict:
        table = new("object")
        for key, item in value.items():
            table[key] = to_lua(item, new, null)
        return table
    if type(value) is list:
        table = new("array")
        for index, item in enumerate(value, start=1):
            table[index] = to_lua(item, new, null)
        return table
    return value


def from_lua(value, kind_of, depth=0):
    kind = lua54.lua_type(value)
    if kind is None:
        if value is None or type(value) in {bool, int, str}:
            return value
        if type(value) is float and math.isfinite(value):
            return value
        raise Failure("lua_adapter_output_invalid", "Unsupported JSON value")
    if kind != "table" or depth > 16:
        raise Failure("lua_adapter_output_invalid", "Unsupported JSON table")
    tag = kind_of(value)
    items = list(value.items())
    if tag == "null":
        if items:
            raise Failure("lua_adapter_output_invalid", "A JSON null marker cannot contain data")
        return None
    if tag == "array" or (tag is None and items and all(type(key) is int for key, _ in items)):
        keys = sorted(key for key, _ in items if type(key) is int)
        if len(keys) != len(items) or keys != list(range(1, len(items) + 1)):
            raise Failure("lua_adapter_output_invalid", "Array keys must be contiguous")
        mapping = dict(items)
        return [from_lua(mapping[key], kind_of, depth + 1) for key in keys]
    if tag in {None, "object"} and all(type(key) is str for key, _ in items):
        return {key: from_lua(item, kind_of, depth + 1) for key, item in items}
    raise Failure("lua_adapter_output_invalid", "Unsupported JSON object keys")


def map_event(request, layers, runtime, load_text, env, new, null, kind_of):
    started = time.perf_counter()
    chunk, problem = load_text(request["source"], env)
    if chunk is None:
        raise Failure("lua_adapter_syntax_error", "Invalid Lua source")
    mapper = chunk()
    if lua54.lua_type(mapper) != "function":
        raise Failure("lua_adapter_contract_invalid", "An adapter must return a function")
    # Count JSON payload nesting independently of the outer protocol envelope.
    event = from_lua(mapper(to_lua(request["event"], new, null)), kind_of, depth=-1)
    original = request["event"]
    if type(event) is not dict or set(event) != set(original):
        raise Failure("lua_adapter_output_invalid", "An adapter must return an event")
    if any(canonical(event[key]) != canonical(original[key]) for key in ("schema", "kind", "phase", "binding")):
        raise Failure("lua_adapter_binding_changed", "Immutable binding changed")
    if len(canonical(event["payload"]).encode("utf-8")) > request["limits"]["output_bytes"]:
        raise Failure("lua_adapter_output_too_large", "Output size limit")
    return {"schema": "lua-adapter-worker.v1", "event": event, "isolation": layers,
            "elapsed_ms": round((time.perf_counter() - started) * 1000)}


def main():
    try:
        raw = sys.stdin.buffer.read(shared["REQUEST_LIMIT"] + 1)
        if len(raw) > shared["REQUEST_LIMIT"]:
            raise Failure("lua_adapter_input_too_large", "Input size limit")
        request = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        if type(request) is not dict or set(request) != {"event", "source", "limits"}:
            raise Failure("lua_adapter_invalid_input", "Invalid request")
        event = request["event"]
        if type(event) is not dict or set(event) != {"schema", "kind", "phase", "binding", "payload"} or event["schema"] != EVENT_SCHEMA:
            raise Failure("lua_adapter_invalid_input", "Invalid event")
        shared["_check_input"](event["binding"])
        shared["_check_input"](event["payload"])
        limits = request["limits"]
        runtime, load_text, env = shared["_runtime"](limits["memory_bytes"], limits["instructions"])
        helpers, new, kind_of = runtime.eval(JSON_SUPPORT)()
        env["adapter"] = helpers
        null = helpers["null"]
        layers = shared["lockdown"](limits["cpu_seconds"], limits["address_space_bytes"])
        if not REQUIRED_ISOLATION <= set(layers):
            raise Failure("lua_adapter_isolation_unavailable", "Isolation unavailable")
        answer = map_event(request, layers, runtime, load_text, env, new, null, kind_of)
    except Failure as exc:
        code = ("lua_adapter_invalid_input" if exc.code == "transform_input_invalid" else
                exc.code.replace("transform_", "lua_adapter_"))
        answer = {"error": code}
    except (lua54.LuaMemoryError, MemoryError):
        answer = {"error": "lua_adapter_memory_limit"}
    except lua54.LuaError as exc:
        code = "lua_adapter_instruction_limit" if str(exc).startswith("transform_instruction_limit") else "lua_adapter_failed"
        answer = {"error": code}
    except (ValueError, TypeError, KeyError, OverflowError, UnicodeError, RecursionError):
        answer = {"error": "lua_adapter_output_invalid"}
    sys.stdout.write(json.dumps(answer, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
