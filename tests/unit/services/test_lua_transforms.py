"""Lua row transforms: the optional scripting entry point, in an isolated worker.

A transform reshapes text the owner already supplied, one JSON row at a time,
before ordinary dataset admission. These tests run the real worker process.
"""
import json
import os
import subprocess
import sys

import pytest

lupa = pytest.importorskip("lupa.lua54")

from services.extensions.transform import (  # noqa: E402
    TransformError,
    TransformLimits,
    run_transform,
    worker_environment,
)

TICKETS = "\n".join(json.dumps({"subject": f"VPN drops {index}", "fix": f"Renew the lease {index}",
                                 "internal": index % 2 == 0}) for index in range(12))
TO_MESSAGES = """
return function(row)
  if row.subject == nil then return nil end
  return {messages = {
    {role = "user", content = "Ticket: " .. row.subject},
    {role = "assistant", content = row.fix},
  }}
end
"""


def refused(code, source, text=TICKETS, limits=TransformLimits()):
    with pytest.raises(TransformError) as caught:
        run_transform(source, text, limits)
    assert caught.value.code == code, caught.value.message
    return caught.value


def test_a_script_maps_each_row_and_reports_what_it_did():
    result = run_transform(TO_MESSAGES, TICKETS)

    rows = [json.loads(line) for line in result["text"].splitlines()]
    assert len(rows) == 12 and result["input_rows"] == result["output_rows"] == 12
    assert rows[0] == {"messages": [{"role": "user", "content": "Ticket: VPN drops 0"},
                                    {"role": "assistant", "content": "Renew the lease 0"}]}
    assert result["schema"] == "lua-row-transform.v1"
    assert type(result["elapsed_ms"]) is int and result["elapsed_ms"] >= 0
    assert {"process", "no_new_privileges", "resource_limits", "no_new_descriptors"} <= set(result["isolation"])


def test_returning_nil_drops_a_row_and_dropping_everything_is_refused():
    keep_public = "return function(row) if row.internal then return nil end return {text = row.subject} end"
    result = run_transform(keep_public, TICKETS)
    assert result["output_rows"] == 6
    assert all(json.loads(line)["text"].endswith(("1", "3", "5", "7", "9", "11"))
               for line in result["text"].splitlines())

    refused("transform_output_empty", "return function(row) return nil end")


@pytest.mark.parametrize("name", ["io", "os", "require", "dofile", "loadfile", "load", "debug", "package",
                                  "coroutine", "collectgarbage", "setmetatable", "getmetatable", "rawset",
                                  "rawget", "pcall", "xpcall", "print", "python", "string.dump", "math.random",
                                  "_G", "warn"])
def test_the_script_sees_no_file_process_network_or_escape_functions(name):
    probe = f"return function(row) return {{present = tostring({name} ~= nil)}} end"
    result = run_transform(probe, '{"a": 1}')
    assert json.loads(result["text"]) == {"present": "false"}


def test_string_methods_work_but_the_string_metatable_is_not_reachable():
    script = 'return function(row) return {upper = row.subject:upper(), parts = select("#", ("a,b"):find(","))} end'
    result = run_transform(script, '{"subject": "vpn"}')
    assert json.loads(result["text"]) == {"parts": 2, "upper": "VPN"}


def test_bytecode_is_never_loaded():
    # Precompiled chunks start with ESC "Lua": plain-text validation refuses them first...
    refused("transform_script_invalid", "\x1bLua" + "T" * 40)
    # ...and the worker's loader accepts text chunks only, even if one got through.
    from services.extensions import transform_worker as worker
    runtime, load_text, env = worker._runtime(8_000_000, 1_000_000)
    attempt = runtime.eval("function(load_text, env) local chunk, problem = "
                           "load_text(string.dump(function() return 1 end), env) return chunk == nil, problem end")
    rejected, problem = attempt(load_text, env)
    assert rejected is True and "binary" in problem


def test_a_compile_error_names_the_line():
    error = refused("transform_syntax_error", "return function(row)\n  return {a = }\nend")
    assert "2" in error.message


def test_a_script_must_return_a_row_function():
    refused("transform_contract_invalid", "return 42")


def test_a_runtime_error_names_the_row():
    error = refused("transform_failed", 'return function(row) if row.n == 3 then error("bad row") end return row end',
                    "\n".join(json.dumps({"n": index}) for index in range(1, 6)))
    assert error.message.startswith("Row 3:") and "bad row" in error.message


@pytest.mark.parametrize("script, code", [
    ("return function(row) while true do end end", "transform_instruction_limit"),
    ("return function(row) local t = {} for i = 1, 1e9 do t[i] = ('x'):rep(1000) .. i end end",
     "transform_memory_limit"),
    ("return function(row) return {text = ('x'):rep(1500000)} end", "transform_output_too_large"),
])
def test_steps_memory_and_output_are_bounded(script, code):
    refused(code, script, '{"a": 1}\n{"a": 2}')


def test_a_worker_that_outlives_its_wall_clock_is_stopped():
    refused("transform_time_limit", "return function(row) while true do end end", '{"a": 1}',
            TransformLimits(instructions=10**12, wall_seconds=2, cpu_seconds=30))


@pytest.mark.parametrize("script", [
    "return function(row) return {f = function() end} end",
    "return function(row) return {[1] = 'a', [3] = 'b'} end",
    "return function(row) return {[true] = 1} end",
    "return function(row) return {n = 0/0} end",
    "return function(row) return 'text' end",
    "return function(row) local t = {} t.self = t return t end",
])
def test_only_json_data_can_leave_the_worker(script):
    refused("transform_output_invalid", script, '{"a": 1}')


@pytest.mark.parametrize("text, code", [
    ("", "transform_input_too_large"),
    ("not json", "transform_input_invalid"),
    ("[1, 2]", "transform_input_invalid"),
    ("\n".join(['{"a": 1}'] * 10_001), "transform_input_too_large"),
    ('{"big": 1180591620717411303424}', "transform_input_invalid"),
])
def test_rows_must_be_bounded_json_objects(text, code):
    refused(code, "return function(row) return row end", text)


@pytest.mark.parametrize("source, code", [
    ("", "transform_script_missing"),
    ("   \n", "transform_script_missing"),
    ("-- x\n" * 5000, "transform_script_too_large"),
    ("return function(row) return row end\x00", "transform_script_invalid"),
])
def test_scripts_must_be_short_plain_text(source, code):
    refused(code, source)


def test_the_worker_inherits_no_credentials(monkeypatch):
    monkeypatch.setenv("AI_INTERNAL_JOB_TOKEN", "internal-secret")
    monkeypatch.setenv("GROQ_API_KEY", "provider-secret")
    environment = worker_environment()
    assert set(environment) == {"PATH", "LANG", "PYTHONDONTWRITEBYTECODE", "PYTHONNOUSERSITE"}
    script = 'return function(row) return {seen = tostring(row.token)} end'
    assert "secret" not in run_transform(script, '{"a": 1}')["text"]


@pytest.mark.skipif(sys.platform != "linux", reason="Linux isolation layers")
@pytest.mark.parametrize("optional_layers", [True, False], ids=["all-layers", "without-namespace-and-landlock"])
def test_after_lockdown_the_worker_cannot_open_files_sockets_or_processes(tmp_path, optional_layers):
    """Run the real lockdown, then try each escape from Python inside the locked process.

    Containers may refuse unprivileged namespaces and Landlock; the layers that always
    apply (no new descriptors, no child processes, no file writes) must hold alone.
    """
    secret = tmp_path / "secret.txt"
    secret.write_text("private")
    disable = "" if optional_layers else (
        "worker._landlock = lambda libc: False\n"
        "os.unshare = lambda flags: (_ for _ in ()).throw(OSError('namespaces refused'))\n")
    probe = f"""
import json, os, socket, subprocess, sys
sys.path.insert(0, {json.dumps(os.getcwd())})
from services.extensions import transform_worker as worker
{disable}layers = worker.lockdown(5, 768 * 1024 * 1024)
results = {{"layers": layers}}
for name, attempt in (
    ("read", lambda: open({json.dumps(str(secret))}).read()),
    ("write", lambda: open({json.dumps(str(tmp_path / "new.txt"))}, "w").write("x")),
    ("socket", lambda: socket.create_connection(("127.0.0.1", 4000), timeout=1)),
    ("process", lambda: subprocess.run(["/bin/true"], check=True)),
):
    try:
        attempt()
        results[name] = "allowed"
    except Exception as exc:
        results[name] = type(exc).__name__
print(json.dumps(results))
"""
    completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=30,
                               env=worker_environment())
    results = json.loads(completed.stdout.strip().splitlines()[-1])
    assert {"read", "write", "socket", "process"} <= set(results)
    for name in ("read", "write", "socket", "process"):
        assert results[name] != "allowed", (name, results)
    assert not (tmp_path / "new.txt").exists()
    if not optional_layers:
        assert {"network_namespace", "landlock"}.isdisjoint(results["layers"])


@pytest.mark.skipif(sys.platform != "linux", reason="Linux resource limits")
def test_without_any_other_layer_the_worker_can_neither_fork_nor_grow_an_inherited_file(tmp_path):
    """Fork and file-size limits hold on their own, even for a descriptor the worker was handed."""
    inherited = tmp_path / "inherited.log"
    probe = f"""
import json, os, sys
sys.path.insert(0, {json.dumps(os.getcwd())})
from services.extensions import transform_worker as worker
handle = os.open({json.dumps(str(inherited))}, os.O_WRONLY | os.O_CREAT, 0o600)
os.dup2(handle, 2)
worker._landlock = lambda libc: False
os.unshare = lambda flags: (_ for _ in ()).throw(OSError("namespaces refused"))
worker.lockdown(5, 768 * 1024 * 1024)
results = {{}}
def fork():
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
for name, attempt in (("fork", fork), ("grow", lambda: os.write(2, b"x"))):
    try:
        attempt()
        results[name] = "allowed"
    except OSError as exc:
        results[name] = exc.errno
print(json.dumps(results), file=sys.stdout)
"""
    completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=30,
                               env=worker_environment())
    results = json.loads(completed.stdout.strip().splitlines()[-1])
    assert results["fork"] != "allowed" and results["grow"] != "allowed", results
    assert inherited.stat().st_size == 0


def test_a_lua_infinity_cannot_leave_the_worker():
    error = refused("transform_output_invalid", "return function(row) return {n = math.huge} end", '{"a": 1}')
    # The owner is told why, not just that the row was not JSON.
    assert "finite numbers" in error.message


def test_deep_nesting_is_refused_before_it_can_exhaust_the_worker():
    deep = "local t = {} local cur = t for i = 1, 40 do cur.next = {} cur = cur.next end return t"
    refused("transform_output_invalid", f"return function(row) {deep} end", '{"a": 1}')


def test_a_table_with_gaps_or_mixed_keys_is_not_silently_reshaped():
    refused("transform_output_invalid", "return function(row) return {list = {[1] = 'a', [2] = 'b', [4] = 'd'}} end",
            '{"a": 1}')
    refused("transform_output_invalid", "return function(row) return {mixed = {'a', key = 'b'}} end", '{"a": 1}')


def test_output_is_bounded_inside_the_worker_before_it_is_sent():
    limits = TransformLimits(output_bytes=200)
    error = refused("transform_output_too_large", "return function(row) return {text = ('x'):rep(80)} end",
                    "\n".join(['{"a": 1}'] * 5), limits)
    assert error.message == "The transformed rows exceed the dataset size limit."


def test_the_worker_stops_producing_rows_at_the_output_limit():
    """The parent buffers the worker's answer, so the worker itself must stop at the limit."""
    from services.extensions import transform_worker as worker

    runtime, load_text, env = worker._runtime(8_000_000, 1_000_000)
    request = {"source": "return function(row) return {text = ('x'):rep(80)} end",
               "rows": [{"a": index} for index in range(5)], "limits": {"output_bytes": 200}}
    with pytest.raises(worker.Failure) as caught:
        worker.transform(request, ["process"], runtime, load_text, env)
    assert caught.value.code == "transform_output_too_large"


def test_only_transform_codes_from_the_worker_reach_the_caller(monkeypatch):
    import services.extensions.transform as runner

    for answer in ({"error": "provider_secret_leak", "message": "x"}, {"error": 7, "message": "x"},
                   {"error": "transform_failed", "message": None}, {"rows": "not rows"}):
        completed = subprocess.CompletedProcess([], 0, json.dumps(answer).encode(), b"")
        monkeypatch.setattr(runner.subprocess, "run", lambda *args, completed=completed, **kwargs: completed)
        with pytest.raises(TransformError) as caught:
            run_transform("return function(row) return row end", '{"a": 1}')
        assert caught.value.code == "transform_unavailable", answer


def test_the_worker_has_a_wall_clock_limit(monkeypatch):
    import services.extensions.transform as runner

    seen = {}

    def fake_run(*args, **kwargs):
        seen.update(kwargs)
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    refused("transform_time_limit", "return function(row) return row end", '{"a": 1}',
            TransformLimits(wall_seconds=7))
    assert seen["timeout"] == 7 and seen["env"] == worker_environment() and seen["close_fds"] is True


@pytest.mark.skipif(sys.platform != "linux", reason="Linux descriptor limits")
def test_lockdown_closes_inherited_descriptors_without_walking_the_whole_table(tmp_path):
    """Every Lua adapter call pays the lockdown, so it must not scale with RLIMIT_NOFILE.

    Containers commonly allow a million descriptors; closing them one by one took
    about 0.3 s per call. Inherited descriptors must still end up closed.
    """
    probe = f"""
import json, os, resource, sys, time
sys.path.insert(0, {json.dumps(os.getcwd())})
from services.extensions import transform_worker as worker
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
inherited = os.open({json.dumps(str(tmp_path / "inherited"))}, os.O_WRONLY | os.O_CREAT, 0o600)
high = os.dup2(inherited, min(hard - 1, 100_000))
started = time.perf_counter()
worker.lockdown(5, 768 * 1024 * 1024)
elapsed = time.perf_counter() - started
closed = []
for fd in (inherited, high):
    try:
        os.fstat(fd)
        closed.append(False)
    except OSError:
        closed.append(True)
print(json.dumps({{"hard": hard, "elapsed": elapsed, "closed": closed}}))
"""
    completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60,
                               env=worker_environment())
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["closed"] == [True, True], result
    if result["hard"] >= 100_000:
        assert result["elapsed"] < 0.05, result
