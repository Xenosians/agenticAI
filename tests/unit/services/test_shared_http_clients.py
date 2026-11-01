"""Outbound calls reuse their connections, and every request still crosses Lua.

Opening a new HTTPS connection costs a TCP and TLS handshake (33-52 ms to the
model providers from the development host) before a request is even sent. The
shared connections must not change who is allowed where: each request is still
inspected by the Lua connection adapter, and each event loop gets its own pool.
"""
import asyncio
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest

pytest.importorskip("lupa.lua54")

from services.extensions import transport  # noqa: E402
from services.lua_adapters import http_clients  # noqa: E402


@contextmanager
def provider():
    """A local chat-completions endpoint that records which connection each request used."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):  # noqa: N802 - http.server API
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            seen.append(self.client_address[1])
            body = json.dumps({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/chat/completions", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.fixture
def lua_requests(monkeypatch):
    monkeypatch.setenv("EXTENSION_ALLOWED_ORIGINS", "http://127.0.0.1")
    inspected = []
    original = http_clients.inspect_data_async

    async def recording(kind, phase, payload, binding):
        inspected.append((kind, phase, payload["method"]))
        return await original(kind, phase, payload, binding)

    monkeypatch.setattr(http_clients, "inspect_data_async", recording)
    return inspected


def allow(monkeypatch, url):
    origin = url.split("/v1/")[0]
    monkeypatch.setenv("EXTENSION_ALLOWED_ORIGINS", origin)


def test_consecutive_provider_calls_share_one_connection(lua_requests, monkeypatch):
    with provider() as (url, seen):
        allow(monkeypatch, url)

        async def three_calls():
            for _ in range(3):
                assert (await transport.request_json(url, payload={"messages": []}))["choices"]
            await http_clients.close_shared_clients()

        asyncio.run(three_calls())
    assert len(seen) == 3
    assert len(set(seen)) == 1, "each call opened a new connection"
    assert lua_requests == [("connection", "request", "POST")] * 3


def test_each_event_loop_gets_its_own_connections(lua_requests, monkeypatch):
    with provider() as (url, seen):
        allow(monkeypatch, url)

        async def one_call():
            await transport.request_json(url, payload={"messages": []})
            await http_clients.close_shared_clients()

        asyncio.run(one_call())
        asyncio.run(one_call())
    assert len(seen) == 2
    assert lua_requests == [("connection", "request", "POST")] * 2


def test_closing_releases_the_connections_and_a_later_call_still_works(lua_requests, monkeypatch):
    with provider() as (url, seen):
        allow(monkeypatch, url)

        async def call_close_call():
            await transport.request_json(url, payload={"messages": []})
            await http_clients.close_shared_clients()
            await transport.request_json(url, payload={"messages": []})
            await http_clients.close_shared_clients()

        asyncio.run(call_close_call())
    assert len(seen) == 2 and seen[0] != seen[1]


def test_shared_clients_keep_the_configured_trust(monkeypatch, tmp_path):
    monkeypatch.setenv("ITSM_HTTP_CA_BUNDLE", str(tmp_path / "missing.pem"))

    async def build():
        with pytest.raises(ValueError, match="^transport_ca_bundle_invalid$"):
            async with http_clients.shared_async_client(timeout=5):
                pass

    asyncio.run(build())
