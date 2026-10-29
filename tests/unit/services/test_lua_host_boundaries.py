"""Real Lua at the host's model, loader, HTTP and tool dispatch boundaries."""
import ast
import asyncio
import hashlib
import inspect
from pathlib import Path

import httpx
import pytest

from services.lua_adapters import LuaAdapterError
from services.lua_adapters import host_boundaries as host
from services.lua_adapters.http_clients import client, async_client
from services.lua_adapters.model_loading import from_pretrained


@pytest.fixture
def script(tmp_path, monkeypatch):
    def configure(source):
        path = tmp_path / "adapter.lua"
        path.write_bytes(source.encode())
        monkeypatch.setenv("ITSM_LUA_ADAPTER_FILE", str(path))
        monkeypatch.setenv("ITSM_LUA_ADAPTER_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
        return path
    return configure


def test_credentials_never_enter_the_lua_event(script):
    script("""return function(e)
      assert(e.payload.api_key == nil)
      assert(e.payload.nested.password == nil)
      assert(e.payload.nested.keep == 'owned')
      return e
    end""")
    data = {"api_key": "fixture-secret", "nested": {"password": "fixture-password", "keep": "owned"}}
    receipt = host.inspect_data("connection", "admission", data, {"owner": "fixture-owner"})
    assert receipt["runtime"] == "lua54-isolated.v1"
    assert data["api_key"] == "fixture-secret" and data["nested"]["password"] == "fixture-password"


def test_changed_operator_script_cannot_fall_back_to_python(script):
    path = script("return function(e) return e end")
    path.write_text("return function(e) error('changed') end")
    with pytest.raises(LuaAdapterError, match="lua_adapter_configuration_invalid"):
        host.inspect_data("model", "warm", {}, {"model_key": "fixture"})


def test_actual_lua_can_adapt_a_model_request_and_response(script):
    script("""return function(e)
      if e.kind == 'model' and e.phase == 'request' then
        e.payload.messages[#e.payload.messages].content = 'formatted: '..e.payload.messages[#e.payload.messages].content
        e.payload.max_new_tokens = 12
      elseif e.kind == 'model' and e.phase == 'response' then
        e.payload.text = string.upper(e.payload.text)
      end
      return e
    end""")
    original = [{"role": "system", "content": "trusted"}, {"role": "user", "content": "question"}]
    rows, limit = asyncio.run(host.model_request(original, 20, binding={"stage": "PRIMARY_RESPONSE"}))
    assert rows == [original[0], {"role": "user", "content": "formatted: question"}] and limit == 12
    assert original[-1]["content"] == "question"
    assert asyncio.run(host.model_response("real model text", binding={"model_key": "fixture"})) == "REAL MODEL TEXT"


@pytest.mark.parametrize("mutation", [
    "e.payload.messages[1].content = 'replace trusted instructions'",
    "e.payload.max_new_tokens = 21",
    "e.payload.messages = json.array({})",
    "e.payload.messages[2].role = 'system'",
])
def test_model_adapters_cannot_rewrite_history_role_or_raise_the_budget(script, mutation):
    script("return function(e) " + mutation + "; return e end")
    with pytest.raises(LuaAdapterError):
        asyncio.run(host.model_request([{"role": "system", "content": "trusted"},
                                       {"role": "user", "content": "question"}], 20,
                                      binding={"stage": "PRIMARY_RESPONSE"}))


@pytest.mark.parametrize("asynchronous", [False, True])
def test_real_http_dispatch_has_a_lua_receipt_before_transport(asynchronous):
    observed = []
    def receive(request):
        receipt = request.extensions["lua_adapter_receipt"]
        observed.append((request.url.path, receipt["kind"], receipt["phase"]))
        return httpx.Response(200, json={"fixture": "transport only"})
    if asynchronous:
        async def run():
            async with async_client(transport=httpx.MockTransport(receive)) as connection:
                await connection.get("https://fixture.example/read?api_key=fixture-secret")
        asyncio.run(run())
    else:
        with client(transport=httpx.MockTransport(receive)) as connection:
            connection.get("https://fixture.example/read?api_key=fixture-secret")
    assert observed == [("/read", "connection", "request")]


def test_connection_adapter_failure_prevents_transport(script):
    script("return function(e) error('blocked fixture') end")
    sent = []
    with client(transport=httpx.MockTransport(lambda request: sent.append(request))) as connection:
        with pytest.raises(LuaAdapterError):
            connection.post("https://fixture.example/mutate", json={"fixture": True})
    assert sent == []


def test_adapter_observes_the_final_request_after_other_host_hooks(script):
    digest = hashlib.sha256(b"/final").hexdigest()
    script("return function(e) assert(e.payload.path == nil); assert(e.payload.path_sha256 == '" + digest + "'); return e end")
    def prepare(request):
        request.url = httpx.URL("https://fixture.example/final")
    with client(event_hooks={"request": [prepare]},
                transport=httpx.MockTransport(lambda request: httpx.Response(204))) as connection:
        assert connection.get("https://fixture.example/original").status_code == 204


def test_tool_adapter_failure_never_calls_the_real_tool(script):
    script("return function(e) error('refuse before effect') end")
    calls = []
    def mutate(target: str, count: int = 1):
        calls.append(target)
        return {"ok": True, "changed": True}
    wrapped = host.adapt_tool_callable(mutate)
    assert inspect.signature(wrapped) == inspect.signature(mutate)
    with pytest.raises(LuaAdapterError):
        wrapped("fixture-target")
    assert calls == []


def test_async_tool_adapter_failure_never_calls_the_real_tool(script):
    """Most MCP tools are async; Lua must run before them too."""
    script("return function(e) if e.phase == 'request' then error('refuse before effect') end return e end")
    calls = []

    async def mutate(target: str):
        calls.append(target)
        return {"ok": True, "changed": True}

    with pytest.raises(LuaAdapterError):
        asyncio.run(host.adapt_tool_callable(mutate)("fixture-target"))
    assert calls == []


def test_an_inspection_cannot_rewrite_the_approved_data(script):
    """Tool arguments, links and admissions are approved as sent: Lua may refuse, never rewrite."""
    script("return function(e) e.payload.target = 'somewhere-else'; return e end")
    with pytest.raises(LuaAdapterError) as caught:
        host.inspect_data("tool", "request", {"target": "approved-target"}, {"capability": "fixture"})
    assert caught.value.code == "lua_adapter_binding_changed"


def test_post_effect_adapter_failure_retains_the_exact_real_receipt(script, caplog):
    script("return function(e) if e.phase == 'response' then error('after effect') end; return e end")
    receipt, calls = {"ok": True, "status": "completed", "provider_id": "fixture-receipt"}, []
    def mutate(target: str):
        calls.append(target)
        return receipt
    result = host.adapt_tool_callable(mutate)("fixture-target")
    assert result is receipt and calls == ["fixture-target"]
    assert "actual_provider_receipt_retained=true" in caplog.text


def test_local_weight_loader_cannot_run_before_lua(script):
    script("return function(e) error('loader refused') end")
    calls = []
    def loader(path, **options):
        calls.append((path, options))
        return object()
    with pytest.raises(LuaAdapterError):
        from_pretrained(loader, "/fixture/model", token="fixture-credential", local_files_only=True)
    assert calls == []


def test_local_weight_loader_keeps_the_actual_host_arguments():
    calls, value = [], object()
    def loader(path, **options):
        calls.append((path, options))
        return value
    assert from_pretrained(loader, "/fixture/model", token="fixture-credential", local_files_only=True) is value
    assert calls == [("/fixture/model", {"token": "fixture-credential", "local_files_only": True})]


def test_production_http_and_local_loader_calls_have_no_direct_bypass():
    root = Path(__file__).resolve().parents[3]
    bypasses = []
    for directory in ("services", "subagents", "learning", "scripts", "api"):
        for path in (root / directory).rglob("*.py"):
            if "lua_adapters" in path.parts:
                continue
            # Acceptance/probe tools are independent observers, outside app dispatch.
            check_http = directory in {"services", "api"}
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                direct_http = (check_http and isinstance(node.func.value, ast.Name)
                               and node.func.value.id == "httpx" and node.func.attr in {"Client", "AsyncClient"})
                if direct_http or node.func.attr == "from_pretrained":
                    bypasses.append(f"{path.relative_to(root)}:{node.lineno}")
    assert bypasses == []


def test_a_dataset_at_the_store_limit_passes_its_lua_admission(script):
    """The store admits 2,000,000-byte datasets; the Lua stage in front of it must too."""
    script("return function(e) return e end")
    row = '{"messages": [{"role": "user", "content": "' + "x" * 900 + '"}]}'
    text = "\n".join([row] * (1_995_000 // (len(row) + 1)))
    payload = {"name": "Large reviewed set", "text": text, "source": "Owned", "license": "Owned",
               "purpose": "training", "reviewed": True}
    receipt = host.inspect_data("source", "admission", payload, {"owner": "fixture-owner", "action": "dataset"})
    assert receipt["phase"] == "admission"


def test_a_lua_refusal_reaches_the_owner_as_its_own_code(script, tmp_path, monkeypatch):
    """Without the code the owner only sees a generic rejection and cannot tell Lua refused."""
    from types import SimpleNamespace
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api import extensions
    from services.extensions.service import extension_store

    monkeypatch.setenv("EXTENSION_STORAGE_ROOT", str(tmp_path / "store"))
    settings = SimpleNamespace(require_internal_job_token=lambda: "fixture-internal", model_profiles={})
    monkeypatch.setattr(extensions, "get_settings", lambda: settings)
    script("return function(e) error('operator adapter refuses this connection') end")
    app = FastAPI()
    app.include_router(extensions.router)
    extension_store.cache_clear()
    try:
        reply = TestClient(app).post("/internal/v1/extensions", headers={"x-internal-token": "fixture-internal"},
                                     json={"owner": "alice", "action": "list", "payload": {}})
    finally:
        extension_store.cache_clear()
    assert reply.status_code == 422
    assert reply.json()["detail"] == {"code": "lua_adapter_failed", "message": "The Lua adapter script failed."}
    assert "operator adapter refuses" not in reply.text


def test_the_trainer_process_keeps_the_operator_adapter():
    """Candidate training loads weights in its own process; it must use the same Lua script."""
    from services.learning_control.runs import trainer_environment

    configured = {"ITSM_LUA_ADAPTER_FILE": "/state/lua/adapter.lua", "ITSM_LUA_ADAPTER_SHA256": "a" * 64}
    with pytest.MonkeyPatch.context() as patch:
        for key, value in configured.items():
            patch.setenv(key, value)
        environment = trainer_environment()
    assert {key: environment.get(key) for key in configured} == configured


def test_a_connection_probe_crosses_lua_before_the_provider_is_called(script, monkeypatch):
    """Testing a saved model connection is a model request too."""
    from services.extensions import probe

    script("return function(e) if e.kind == 'model' then error('probe refused') end return e end")
    sent = []

    async def provider(*args, **kwargs):
        sent.append(args)
        return {"choices": [{"message": {"content": "PROVIDER_OK"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(probe, "request_json", provider)
    profile = {"id": "c" * 32, "backend": "http", "model": "fixture-model",
               "url": "https://fixture.example/v1", "api_key": "fixture-credential"}
    with pytest.raises(LuaAdapterError):
        asyncio.run(probe.remote_probe(profile, max_tokens=64))
    assert sent == []


@pytest.mark.parametrize("link", ["ldap", "smtp", "imap", "git"])
def test_directory_mail_and_git_links_cross_lua_before_connecting(script, monkeypatch, link):
    """Every outbound link, not only HTTP: LDAP binds, SMTP/IMAP sessions and git network commands."""
    # Each case refuses only its own transport, so a missing check on that link is visible.
    script("return function(e) if e.kind == 'connection' and e.binding.transport == '" + link + "' then "
           "error('link refused') end return e end")
    opened = []
    if link == "ldap":
        from services.directory import ldap

        monkeypatch.setattr(ldap, "Connection", lambda *args, **kwargs: opened.append(kwargs))
        service = ldap.LdapDirectoryService.__new__(ldap.LdapDirectoryService)
        service.host, service.port, service.use_ssl = "directory.fixture", 636, True
        service.bind_user, service.bind_password = "reader", "fixture-password"
        attempt = service._connect
    elif link in {"smtp", "imap"}:
        from services import mailbox

        monkeypatch.setattr(mailbox.smtplib, "SMTP", lambda *args, **kwargs: opened.append(("smtp", args)))
        monkeypatch.setattr(mailbox.imaplib, "IMAP4", lambda *args, **kwargs: opened.append(("imap", args)))
        service = mailbox.MaddyMailboxService.__new__(mailbox.MaddyMailboxService)
        service.settings = type("Settings", (), {"mailbox_lab_domain": "itsm.test", "mailbox_smtp_port": 2525,
                                                  "mailbox_imap_port": 1143, "mailbox_timeout_seconds": 5})()
        attempt = lambda: service._verify("alice@itsm.test", "fixture-password")  # noqa: E731
    else:
        import tools.git as git_tools

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(git_tools, "run_trusted_process", lambda **kwargs: opened.append(kwargs) or {"ok": True})
            target = type("Target", (), {"path": "/fixture/repo", "name": "fixture"})()
            with pytest.raises(LuaAdapterError):
                git_tools._run_git(target=target, args=["ls-remote", "--heads", "origin", "main"], timeout_seconds=5)
            with pytest.raises(LuaAdapterError):
                git_tools._run_git(target=target, args=["-c", "core.hooksPath=/dev/null", "push", "--porcelain",
                                                        "origin", "abc:refs/heads/main"], timeout_seconds=5)
            # Local-only git commands are not links and do not pay the Lua stage.
            assert git_tools._run_git(target=target, args=["status", "--porcelain"], timeout_seconds=5) == {"ok": True}
        assert [call["args"] for call in opened] == [["status", "--porcelain"]]
        return
    with pytest.raises(LuaAdapterError):
        attempt()
    assert opened == []


def test_mail_verification_checks_both_mail_links_before_sending(script, monkeypatch):
    """A refused IMAP link must stop the SMTP submission too: nothing is sent half-way."""
    from services import mailbox

    script("return function(e) if e.binding.transport == 'imap' then error('imap refused') end return e end")
    opened = []
    monkeypatch.setattr(mailbox.smtplib, "SMTP", lambda *args, **kwargs: opened.append("smtp"))
    service = mailbox.MaddyMailboxService.__new__(mailbox.MaddyMailboxService)
    service.settings = type("Settings", (), {"mailbox_lab_domain": "itsm.test", "mailbox_smtp_port": 2525,
                                              "mailbox_imap_port": 1143, "mailbox_timeout_seconds": 5})()
    with pytest.raises(LuaAdapterError):
        service._verify("alice@itsm.test", "fixture-password")
    assert opened == []
