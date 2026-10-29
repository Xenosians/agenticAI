"""Real private-CA HTTP/WSS checks; no external provider or global trust edits."""
import asyncio
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import ssl
import threading

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
import httpx
import pytest

from services.lua_adapters.http_clients import async_client, client
from services.transport_tls import configured_context


@pytest.fixture
def certificates(tmp_path, monkeypatch):
    monkeypatch.delenv("ITSM_HTTP_CA_BUNDLE", raising=False)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    monkeypatch.delenv("ITSM_LUA_ADAPTER_FILE", raising=False)
    monkeypatch.delenv("ITSM_LUA_ADAPTER_SHA256", raising=False)
    now = datetime.now(timezone.utc)
    ca_key, server_key = (ec.generate_private_key(ec.SECP256R1()) for _ in range(2))
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "isolated transport fixture")])
    ca = (x509.CertificateBuilder().subject_name(ca_name).issuer_name(ca_name)
          .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
          .not_valid_before(now - timedelta(minutes=1)).not_valid_after(now + timedelta(hours=1))
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .sign(ca_key, hashes.SHA256()))
    server = (x509.CertificateBuilder()
              .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")]))
              .issuer_name(ca_name).public_key(server_key.public_key())
              .serial_number(x509.random_serial_number()).not_valid_before(now - timedelta(minutes=1))
              .not_valid_after(now + timedelta(hours=1))
              .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
              .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
              .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
              .sign(ca_key, hashes.SHA256()))
    ca_file, cert_file, key_file = (tmp_path / name for name in ("ca.pem", "server.pem", "server.key"))
    ca_file.write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    cert_file.write_bytes(server.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(server_key.private_bytes(serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_file, key_file)
    return ca_file, context


@contextmanager
def https_server(context):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            body = b'{"observed":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        assert not thread.is_alive()


def test_explicit_ca_works_without_proxy_environment_and_keeps_hostname_verification(certificates, monkeypatch):
    ca, server_context = certificates
    monkeypatch.setenv("SSL_CERT_FILE", str(ca))
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    with https_server(server_context) as (port, requests):
        # Reproduce why merely mounting SSL_CERT_FILE cannot fix trust_env=False.
        with client(trust_env=False, timeout=3) as transport:
            with pytest.raises(httpx.ConnectError):
                transport.get(f"https://localhost:{port}/untrusted")
        assert requests == []
        monkeypatch.setenv("ITSM_HTTP_CA_BUNDLE", str(ca))
        with client(trust_env=False, timeout=3) as transport:
            assert transport.get(f"https://localhost:{port}/verified").json() == {"observed": True}
            with pytest.raises(httpx.ConnectError):
                transport.get(f"https://127.0.0.1:{port}/wrong-host")
        assert requests == ["/verified"]
    context = configured_context()
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname


def test_async_transport_uses_the_same_explicit_trust(certificates, monkeypatch):
    ca, server_context = certificates
    monkeypatch.setenv("ITSM_HTTP_CA_BUNDLE", str(ca))
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    with https_server(server_context) as (port, requests):
        async def probe():
            async with async_client(trust_env=False, timeout=3) as transport:
                assert (await transport.get(f"https://localhost:{port}/async")).json() == {"observed": True}
        asyncio.run(probe())
        assert requests == ["/async"]


@pytest.mark.parametrize("value", ["", "relative.pem", "/fixture-ca-does-not-exist.pem"])
def test_invalid_explicit_bundle_refuses_without_default_fallback(monkeypatch, value):
    monkeypatch.setenv("ITSM_HTTP_CA_BUNDLE", value)
    with pytest.raises(ValueError, match="^transport_ca_bundle_invalid$"):
        client(trust_env=False)


def test_malformed_bundle_refuses_and_explicit_caller_context_is_preserved(tmp_path, monkeypatch):
    malformed = tmp_path / "invalid.pem"
    malformed.write_text("not a certificate")
    monkeypatch.setenv("ITSM_HTTP_CA_BUNDLE", str(malformed))
    with pytest.raises(ValueError, match="^transport_ca_bundle_invalid$"):
        client(trust_env=False)
    # A caller-owned SSLContext is a separate explicit trust policy.
    context = ssl.create_default_context()
    with client(trust_env=False, verify=context):
        pass


def test_real_wss_progress_join_and_lease_use_deployment_ca(certificates, monkeypatch):
    from services.job_progress import JobProgress
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve

    ca, server_context = certificates
    monkeypatch.setenv("ITSM_HTTP_CA_BUNDLE", str(ca))

    # Internal callbacks must not inherit an operator's unrelated shell proxy.
    monkeypatch.setenv("WSS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")

    async def probe():
        observed = []
        progress_received = asyncio.Event()

        async def handler(socket):
            assert socket.request.headers["X-Internal-Token"] == "fixture-internal"
            async for raw in socket:
                frame = json.loads(raw)
                observed.append(frame[3])
                await socket.send(json.dumps([frame[0], frame[1], frame[2], "phx_reply",
                                               {"status": "ok", "response": {}}]))
                if frame[3] == "progress":
                    progress_received.set()

        async with serve(handler, "127.0.0.1", 0, ssl=server_context) as server:
            port = server.sockets[0].getsockname()[1]
            # System roots alone refuse this issuer, so the real client below
            # can pass only if the selected context reaches websockets.connect.
            with pytest.raises(ssl.SSLCertVerificationError):
                async with connect(f"wss://localhost:{port}", ssl=ssl.create_default_context(), proxy=None):
                    pass
            publisher = JobProgress(f"https://localhost:{port}", "fixture-internal", "a" * 32, 1)
            task = asyncio.create_task(publisher.run())
            try:
                publisher.publish("planning")
                async with asyncio.timeout(8):
                    await progress_received.wait()
                    assert await publisher.renew_lease() == {"status": "ok", "response": {}}
                assert observed[:3] == ["phx_join", "progress", "renew_lease"]
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(probe())
