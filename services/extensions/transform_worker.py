"""Isolated worker for one Lua row transform. Standard library and lupa only.

Started by ``services.extensions.transform`` as ``python -I`` (isolated mode: no
user site-packages, no ``PYTHON*`` variables, no script directory on the path)
with a minimal environment and only its stdin/stdout/stderr pipes. It reads one
request, prepares the Lua runtime, locks itself down, runs the owner's script
over the rows and writes one JSON answer. Imports finish before the lockdown.

Lockdown layers, each independent of the others:
  * no new privileges, no core dumps, no file writes, no child processes;
  * CPU time and address space limits;
  * no descriptor beyond the three pipes, and no new one (no file or socket);
  * Landlock with an empty rule set (no file access, no TCP) where the kernel
    supports it, and a private network namespace where unprivileged namespaces
    are allowed.
The script sees only pure functions: no io, os, require, load, dofile, debug,
coroutine, pcall, metatables, printing or random numbers. Its steps and Lua
heap are bounded.
"""
from __future__ import annotations

import ctypes
import json
import math
import os
import resource
import sys
import time

from lupa import lua54

REQUEST_LIMIT = 4_500_000
MAX_DEPTH = 16
MESSAGE_LIMIT = 300
HOOK_INTERVAL = 1000
LUA_INTEGER = (-(2**63), 2**63 - 1)
PR_SET_NO_NEW_PRIVS = 38
LANDLOCK_CREATE_RULESET, LANDLOCK_RESTRICT_SELF = 444, 446
CLOSE_RANGE = 436  # close_range(2): one call however large the descriptor table is
SAFE_GLOBALS = ("assert", "error", "ipairs", "next", "pairs", "rawequal", "rawlen", "select",
                "tonumber", "tostring", "type")
SAFE_LIBRARIES = {
    "string": ("byte", "char", "find", "format", "gmatch", "gsub", "len", "lower", "match", "rep",
               "reverse", "sub", "upper"),
    "table": ("concat", "insert", "remove", "sort", "unpack"),
    "math": ("abs", "ceil", "floor", "fmod", "huge", "max", "maxinteger", "min", "mininteger", "pi",
             "sqrt", "tointeger", "type", "ult"),
    "utf8": ("char", "charpattern", "codepoint", "codes", "len", "offset"),
}
# Stops the script after `limit` blocks of HOOK_INTERVAL instructions; the hook is the only debug use.
INSTALL_BUDGET = """
function(limit)
  local count, error = 0, error
  debug.sethook(function()
    count = count + 1
    if count > limit then error("transform_instruction_limit", 0) end
  end, "", %d)
end
""" % HOOK_INTERVAL
# Text chunks only (mode "t"), with the restricted table as their whole environment.
LOAD_TEXT = """
function(source, env)
  local chunk, problem = load(source, "=transform", "t", env)
  if chunk then return chunk, nil end
  return nil, tostring(problem)
end
"""


# Builds the script's whole environment inside Lua, so no value crosses into Python.
BUILD_ENVIRONMENT = """
function(globals, libraries)
  local env = {}
  for _, name in ipairs(globals) do env[name] = _G[name] end
  for library, names in pairs(libraries) do
    local source, copy = _G[library], {}
    for _, name in ipairs(names) do copy[name] = source[name] end
    env[library] = copy
  end
  return env
end
"""


class Failure(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(code)
        self.code = code
        self.message = message[:MESSAGE_LIMIT]


MEMORY = Failure("transform_memory_limit", "The transform used more memory than allowed.")
STEPS = Failure("transform_instruction_limit", "The transform ran more steps than allowed.")


def _landlock(libc) -> bool:
    """Deny every file access and TCP bind/connect the running kernel can restrict."""
    libc.syscall.restype = ctypes.c_long
    abi = libc.syscall(LANDLOCK_CREATE_RULESET, None, ctypes.c_size_t(0), ctypes.c_uint32(1))
    if abi < 1:
        return False
    filesystem = (1 << 13) - 1
    for version, bit in ((2, 13), (3, 14), (5, 15)):
        if abi >= version:
            filesystem |= 1 << bit
    network, scoped = (0b11 if abi >= 4 else 0), (0b11 if abi >= 6 else 0)
    attributes = (ctypes.c_uint64 * 3)(filesystem, network, scoped)
    size = 24 if abi >= 6 else 16 if abi >= 4 else 8
    ruleset = libc.syscall(LANDLOCK_CREATE_RULESET, ctypes.byref(attributes), ctypes.c_size_t(size),
                           ctypes.c_uint32(0))
    if ruleset < 0:
        return False
    try:
        return libc.syscall(LANDLOCK_RESTRICT_SELF, ctypes.c_int(ruleset), ctypes.c_uint32(0)) == 0
    finally:
        os.close(ruleset)


def lockdown(cpu_seconds: int, address_space_bytes: int) -> list[str]:
    """Apply every isolation layer this host allows; return their names."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise Failure("transform_isolation_unavailable", "The transform worker could not be isolated.")
    layers = ["process", "no_new_privileges"]
    try:
        os.unshare(os.CLONE_NEWUSER | os.CLONE_NEWNET)
        layers.append("network_namespace")
    except (AttributeError, OSError):
        pass
    if _landlock(libc):
        layers.append("landlock")
    for limit, value in ((resource.RLIMIT_CORE, 0), (resource.RLIMIT_FSIZE, 0), (resource.RLIMIT_NPROC, 0),
                         (resource.RLIMIT_CPU, cpu_seconds), (resource.RLIMIT_AS, address_space_bytes)):
        resource.setrlimit(limit, (value, value))
    # Only the three pipes stay open, and no descriptor can be added: no new file or socket.
    # close_range does this in one call; walking a container's million-slot table took ~0.3 s.
    if libc.syscall(CLOSE_RANGE, ctypes.c_uint(3), ctypes.c_uint(0xFFFFFFFF), ctypes.c_uint(0)) != 0:
        os.closerange(3, resource.getrlimit(resource.RLIMIT_NOFILE)[1])
    resource.setrlimit(resource.RLIMIT_NOFILE, (3, 3))
    layers += ["resource_limits", "no_new_descriptors"]
    return layers


def _runtime(memory_bytes: int, instructions: int):
    runtime = lua54.LuaRuntime(register_eval=False, register_builtins=False, unpack_returned_tuples=True,
                               max_memory=memory_bytes)
    libraries = runtime.table_from({library: runtime.table_from(names) for library, names in SAFE_LIBRARIES.items()})
    env = runtime.eval(BUILD_ENVIRONMENT)(runtime.table_from(SAFE_GLOBALS), libraries)
    runtime.eval(INSTALL_BUDGET)(max(1, instructions // HOOK_INTERVAL))
    return runtime, runtime.eval(LOAD_TEXT), env


def _check_input(value, depth=0):
    """Rows enter Lua as plain data; integers must fit Lua's 64-bit integers."""
    if depth > MAX_DEPTH:
        raise Failure("transform_input_invalid", "Rows nest too deeply.")
    if type(value) is int and not LUA_INTEGER[0] <= value <= LUA_INTEGER[1]:
        raise Failure("transform_input_invalid", "Row numbers must fit in 64 bits.")
    if isinstance(value, list):
        for item in value:
            _check_input(item, depth + 1)
    elif isinstance(value, dict):
        for item in value.values():
            _check_input(item, depth + 1)


def _plain(value, depth=0):
    """A Lua result as JSON data: tables become arrays (keys 1..n) or objects (text keys)."""
    kind = lua54.lua_type(value)
    if kind is None:
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        raise Failure("transform_output_invalid", "Output values must be text, finite numbers, booleans or tables.")
    if kind != "table":
        raise Failure("transform_output_invalid", f"A {kind} cannot be part of a row.")
    if depth >= MAX_DEPTH:
        raise Failure("transform_output_invalid", "Output tables nest too deeply.")
    items = list(value.items())
    if items and all(type(key) is int for key, _ in items):
        keys = sorted(key for key, _ in items)
        if keys != list(range(1, len(keys) + 1)):
            raise Failure("transform_output_invalid", "Array tables must use the keys 1 to n.")
        mapping = dict(items)
        return [_plain(mapping[key], depth + 1) for key in keys]
    if all(isinstance(key, str) for key, _ in items):
        return {key: _plain(item, depth + 1) for key, item in items}
    raise Failure("transform_output_invalid", "A table must be an array or have only text keys.")


def _lua_failure(exc: BaseException, row: int | None) -> Failure:
    if isinstance(exc, (lua54.LuaMemoryError, MemoryError)):
        return MEMORY
    text = str(exc)
    if text.startswith("transform_instruction_limit"):
        return STEPS
    prefix = f"Row {row}: " if row is not None else ""
    return Failure("transform_failed", prefix + text.split("\nstack traceback:")[0])


def transform(request: dict, layers: list[str], runtime, load_text, env) -> dict:
    started = time.perf_counter()
    chunk, problem = load_text(request["source"], env)
    if chunk is None:
        raise Failure("transform_syntax_error", str(problem))
    row = None
    try:
        mapper = chunk()
        if lua54.lua_type(mapper) != "function":
            raise Failure("transform_contract_invalid", "The script must return function(row).")
        rows, size = [], 0
        for row, record in enumerate(request["rows"], start=1):
            result = mapper(runtime.table_from(record, recursive=True))
            if result is None:
                continue
            plain = _plain(result)
            if not isinstance(plain, dict):
                raise Failure("transform_output_invalid", f"Row {row}: return a table with text keys, or nil.")
            size += len(json.dumps(plain, ensure_ascii=False, allow_nan=False).encode()) + 1
            if size > request["limits"]["output_bytes"]:
                raise Failure("transform_output_too_large", "The transformed rows exceed the dataset size limit.")
            rows.append(plain)
    except Failure:
        raise
    except (lua54.LuaError, MemoryError) as exc:
        raise _lua_failure(exc, row) from None
    except (UnicodeDecodeError, RecursionError, ValueError):
        raise Failure("transform_output_invalid", f"Row {row}: the result is not JSON text.") from None
    return {"rows": rows, "elapsed_ms": round((time.perf_counter() - started) * 1000), "isolation": layers}


def main() -> int:
    raw = sys.stdin.buffer.read(REQUEST_LIMIT + 1)
    try:
        if len(raw) > REQUEST_LIMIT:
            raise Failure("transform_input_too_large", "The rows exceed the transform input limit.")
        request = json.loads(raw)
        _check_input(request["rows"])
        limits = request["limits"]
        runtime, load_text, env = _runtime(limits["memory_bytes"], limits["instructions"])
        layers = lockdown(limits["cpu_seconds"], limits["address_space_bytes"])
        answer = transform(request, layers, runtime, load_text, env)
    except Failure as exc:
        answer = {"error": exc.code, "message": exc.message}
    except (lua54.LuaError, MemoryError) as exc:
        failure = _lua_failure(exc, None)
        answer = {"error": failure.code, "message": failure.message}
    sys.stdout.write(json.dumps(answer, ensure_ascii=False))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
