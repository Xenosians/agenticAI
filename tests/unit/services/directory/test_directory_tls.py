"""The directory connection verifies the server's TLS certificate before the bind sends a password.

Real TLS handshakes against a local server signed by a throwaway private CA: the right CA and
name connect; system trust alone, or a wrong name, is refused before any LDAP message is sent.
"""
from datetime import datetime, timedelta, timezone
import socket
import ssl
import threading

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from ldap3 import Connection
from ldap3.core.exceptions import LDAPSocketOpenError
import pytest

from config import Settings
from services.directory.ldap import LdapDirectoryService

SERVER_NAME = "directory.fixture"


@pytest.fixture(autouse=True)
def plain_environment(monkeypatch):
    """No machine profile, deployment bundle or OpenSSL overrides leak into these checks."""
    for name in ("ITSM_LUA_ADAPTER_FILE", "ITSM_LUA_ADAPTER_SHA256", "ITSM_HTTP_CA_BUNDLE",
                 "SSL_CERT_FILE", "SSL_CERT_DIR"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def directory_tls(tmp_path):
    """A TLS listener on 127.0.0.1 whose certificate names directory.fixture, signed by a private CA."""
    now = datetime.now(timezone.utc)
    ca_key, server_key = (ec.generate_private_key(ec.SECP256R1()) for _ in range(2))
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "directory fixture CA")])
    ca = (x509.CertificateBuilder().subject_name(ca_name).issuer_name(ca_name)
          .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
          .not_valid_before(now - timedelta(minutes=1)).not_valid_after(now + timedelta(hours=1))
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .sign(ca_key, hashes.SHA256()))
    server = (x509.CertificateBuilder()
              .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, SERVER_NAME)]))
              .issuer_name(ca_name).public_key(server_key.public_key())
              .serial_number(x509.random_serial_number()).not_valid_before(now - timedelta(minutes=1))
              .not_valid_after(now + timedelta(hours=1))
              .add_extension(x509.SubjectAlternativeName([x509.DNSName(SERVER_NAME)]), critical=False)
              .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
              .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
              .sign(ca_key, hashes.SHA256()))
    ca_file, cert_file, key_file = (tmp_path / name for name in ("ca.pem", "server.pem", "server.key"))
    ca_file.write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    cert_file.write_bytes(server.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(server_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                  serialization.NoEncryption()))
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_file, key_file)
    listener = socket.create_server(("127.0.0.1", 0))
    listener.settimeout(0.2)
    stop = threading.Event()

    def accept():
        while not stop.is_set():
            try:
                raw, _ = listener.accept()
            except OSError:
                continue
            try:
                with context.wrap_socket(raw, server_side=True) as tls:
                    tls.settimeout(1)
                    tls.recv(1)  # hold the session until the client closes it
            except (OSError, ssl.SSLError):
                pass

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    yield listener.getsockname()[1], ca_file
    stop.set()
    thread.join(timeout=3)
    listener.close()


def directory(port, **overrides):
    values = {"ad_host": "127.0.0.1", "ad_port": port, "ad_use_ssl": True, "ad_bind_user": "reader",
              "ad_bind_password": "fixture-password", "ad_base_dn": "DC=fixture"}
    return LdapDirectoryService(Settings(_env_file=None, **(values | overrides)))


def handshake(service):
    connection = Connection(service._create_server(), auto_bind=False, receive_timeout=3)
    try:
        connection.open(read_server_info=False)
    finally:
        connection.unbind()


def test_trusted_ca_and_certificate_name_connect(directory_tls):
    port, ca_file = directory_tls
    handshake(directory(port, ad_tls_ca_file=ca_file, ad_tls_server_name=SERVER_NAME))


def test_the_deployment_ca_bundle_is_trusted_when_no_directory_ca_is_set(directory_tls, monkeypatch):
    port, ca_file = directory_tls
    monkeypatch.setenv("ITSM_HTTP_CA_BUNDLE", str(ca_file))
    handshake(directory(port, ad_tls_server_name=SERVER_NAME))


def test_an_untrusted_certificate_is_refused(directory_tls):
    port, _ = directory_tls
    with pytest.raises(LDAPSocketOpenError):
        handshake(directory(port, ad_tls_server_name=SERVER_NAME))


def test_a_certificate_for_another_name_is_refused(directory_tls):
    port, ca_file = directory_tls
    with pytest.raises(LDAPSocketOpenError):
        handshake(directory(port, ad_tls_ca_file=ca_file, ad_tls_server_name="other.fixture"))


def test_tls_settings_require_verification_and_send_the_expected_name():
    tls = directory(636, ad_host="dc.example.test")._create_server().tls
    assert tls.validate == ssl.CERT_REQUIRED and tls.ca_certs_file is None
    assert tls.valid_names is None and tls.sni == "dc.example.test"
    plain = directory(389, ad_use_ssl=False)._create_server()
    assert plain.ssl is False
