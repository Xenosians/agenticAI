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
import time

import pytest

pytest.importorskip("lupa.lua54")

from services.extensions import transport  # noqa: E402
from services.lua_adapters import http_clients  # noqa: E402


@contextmanager
def provider(idle_timeout: float | None = None):
    """A local chat-completions endpoint that records which connection each request used.

    With ``idle_timeout`` it behaves like the live provider proxy: it advertises
    ``Keep-Alive: timeout=N``, and a request that arrives on a connection idle for
    longer gets no response; the connection is closed (the server already gave it up).
    """
    seen = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.answered_at = None

        def handle_one_request(self):
            if idle_timeout is not None and self.answered_at is not None:
                if not self.rfile.peek(1):  # The client closed the connection.
                    self.close_connection = True
                    return
                if time.monotonic() - self.answered_at > idle_timeout:
                    self.close_connection = True  # Closed without a response.
                    return
            super().handle_one_request()
            self.answered_at = time.monotonic()

        def do_POST(self):  # noqa: N802 - http.server API
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            seen.append(self.client_address[1])
            body = json.dumps({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if idle_timeout is not None:
                self.send_header("Keep-Alive", f"timeout={idle_timeout:g}")
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


def test_a_connection_idle_as_long_as_a_server_keeps_it_is_not_reused(lua_requests, monkeypatch):
    # Node.js and Apache servers drop a connection idle for 5 seconds; the live
    # provider proxy advertises "Keep-Alive: timeout=5". A request sent on such a
    # connection got no response, and the job failed with "Server disconnected
    # without sending a response" (provider_connection_failed).
    with provider(idle_timeout=5.0) as (url, seen):
        allow(monkeypatch, url)

        async def call_wait_call():
            await transport.request_json(url, payload={"messages": []})
            await asyncio.sleep(0.2)
            await transport.request_json(url, payload={"messages": []})  # Reused: same connection.
            await asyncio.sleep(5.2)
            reply = await transport.request_json(url, payload={"messages": []})
            await http_clients.close_shared_clients()
            return reply

        assert asyncio.run(call_wait_call())["choices"]
    assert len(seen) == 3 and seen[0] == seen[1] != seen[2]


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
