import json
import subprocess
from types import SimpleNamespace

import httpx
import pytest

from config import Settings
from services.mailbox import (
    CONFIG_PATH, CONTAINER, MaddyMailboxService, MailboxProviderError,
    PhoenixMailboxCredentialStore, validate_lab_email,
)


def settings(tmp_path, **overrides):
    return Settings(_env_file=None, mailbox_lab_enabled=True,
                    mailbox_provisioning_lock_path=tmp_path / "create.lock", **overrides)


class Store:
    def __init__(self):
        self.password = None
        self.calls = 0

    def validate_configuration(self):
        pass

    def persist(self, *, email, password):
        self.password = password
        self.calls += 1
        return {"mailbox_record_id": "mailbox:unit", "credential_id": "credential:unit",
                "credential_expires_at": "2026-10-20T00:00:00Z"}


class Provider(MaddyMailboxService):
    def __init__(self, settings, store):
        super().__init__(settings, store)
        self.credentials, self.mailboxes, self.calls = set(), set(), []
        self.fail_on = None
        self.verified = False

    def _admin(self, group, action, email=None, *, password=None):
        self.calls.append((group, action, email))
        if self.fail_on == (group, action):
            raise MailboxProviderError("Private process output must not escape")
        identities = self.credentials if group == "creds" else self.mailboxes
        if action == "list":
            return "\n".join(identities)
        identities.add(email)
        if password:
            self.generated_password = password
        return ""

    def _verify(self, email, password):
        assert self.credential_store.password == password  # Persist before network readback.
        if self.fail_on == "verify":
            raise MailboxProviderError("Private SMTP response must not escape")
        self.verified = True


@pytest.mark.parametrize("email", ["a@example.com", "A@itsm.test", "a+alias@itsm.test", "--help@itsm.test",
                                  "a..b@itsm.test", "a.@itsm.test", "a@itsm.test\n", "a@other.test",
                                  "a;cmd@itsm.test", " a@itsm.test"])
def test_email_cannot_expand_authority(email):
    with pytest.raises(ValueError):
        validate_lab_email(email, "itsm.test")


@pytest.mark.parametrize("domain", ["example.com", "test", "a.test\n", "UPPER.test", "a..test", "-a.test"])
def test_settings_only_allow_exact_test_domain(domain):
    with pytest.raises(ValueError):
        Settings(_env_file=None, mailbox_lab_domain=domain)


def test_success_requires_persistence_and_readback_and_never_returns_secret(tmp_path):
    store = Store()
    service = Provider(settings(tmp_path), store)
    result = service.create("alice@itsm.test")
    assert result.ok and result.delivery_verified and result.smtp_verified and result.imap_verified
    assert result.credential_persisted and not result.retry_safe and not result.public_delivery
    assert store.calls == 1 and len(store.password) >= 32
    assert service.verified
    assert store.password not in result.model_dump_json()
    assert "password" not in type(result).model_fields


@pytest.mark.parametrize("existing", ["credentials", "mailboxes"])
def test_existing_identity_is_never_overwritten(tmp_path, existing):
    store = Store()
    service = Provider(settings(tmp_path), store)
    getattr(service, existing).add("alice@itsm.test")
    result = service.create("alice@itsm.test")
    assert not result.ok and result.status == "conflict" and not result.changed
    assert not any(call[1] == "create" for call in service.calls)
    assert store.calls == 0 and not hasattr(service, "generated_password")


@pytest.mark.parametrize("failure", [("creds", "create"), ("imap-acct", "create"), "verify"])
def test_uncertain_mutation_never_blindly_retries_or_reports_success(tmp_path, failure):
    service = Provider(settings(tmp_path), Store())
    service.fail_on = failure
    result = service.create("alice@itsm.test")
    assert not result.ok and result.status == "outcome_unknown" and not result.retry_safe
    assert not result.delivery_verified
    assert "Private" not in result.model_dump_json()
    assert sum(call[1] == "create" for call in service.calls) <= 2
    if failure == ("creds", "create"):
        assert result.changed is None and result.credential_exists is None
    if failure == ("imap-acct", "create"):
        assert result.changed and result.mailbox_exists is None


def test_store_failure_after_mailbox_creation_is_explicitly_unknown(tmp_path):
    store = Store()
    def fail(**kwargs):
        raise RuntimeError("private response")
    store.persist = fail
    service = Provider(settings(tmp_path), store)
    result = service.create("alice@itsm.test")
    assert result.status == "outcome_unknown" and result.changed and result.mailbox_exists
    assert not result.credential_persisted and not result.retry_safe
    assert "private response" not in result.model_dump_json()


def test_missing_store_configuration_prevents_password_generation_and_mutation(tmp_path):
    store = Store()
    store.validate_configuration = lambda: (_ for _ in ()).throw(RuntimeError("missing"))
    service = Provider(settings(tmp_path), store)
    result = service.create("alice@itsm.test")
    assert not result.ok and result.status == "denied" and result.retry_safe
    assert service.calls == [] and not hasattr(service, "generated_password")


def test_competing_creates_for_same_mailbox_generate_only_one_credential(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    store = Store()
    service = Provider(settings(tmp_path), store)
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(lambda _: service.create("alice@itsm.test"), range(2)))
    assert sorted(result.status for result in results) == ["conflict", "executed"]
    assert store.calls == 1
    assert service.calls.count(("creds", "create", "alice@itsm.test")) == 1
    assert service.calls.count(("imap-acct", "create", "alice@itsm.test")) == 1


def test_presence_read_is_not_misreported_as_authenticated_delivery(tmp_path):
    service = Provider(settings(tmp_path), Store())
    service.credentials.add("alice@itsm.test")
    service.mailboxes.add("alice@itsm.test")
    result = service.status("alice@itsm.test")
    assert result.ok and result.mailbox_exists and result.credential_exists
    assert not result.smtp_verified and not result.imap_verified and not result.delivery_verified
    assert not result.changed and not service.verified


def test_admin_password_only_uses_stdin_not_argv_or_exception(tmp_path, monkeypatch):
    service = MaddyMailboxService(settings(tmp_path), Store())
    recorded = []
    def run(argv, **kwargs):
        recorded.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="private secret output", stderr="private")
    monkeypatch.setattr(subprocess, "run", run)
    assert service._admin("creds", "create", "alice@itsm.test", password="Secret_unit") == ""
    argv, kwargs = recorded[0]
    assert argv == ["docker", "exec", "-i", CONTAINER, "/bin/maddy", "-config", CONFIG_PATH,
                    "creds", "create", "alice@itsm.test"]
    assert kwargs["input"] == "Secret_unit\n" and kwargs["shell"] is False
    assert "Secret_unit" not in repr(argv)


def test_admin_timeout_discards_secret_bearing_process_details(tmp_path, monkeypatch):
    def fail(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 1, output="secret", stderr="secret")
    monkeypatch.setattr(subprocess, "run", fail)
    service = MaddyMailboxService(settings(tmp_path), Store())
    with pytest.raises(MailboxProviderError) as error:
        service._admin("creds", "create", "alice@itsm.test", password="secret")
    assert "secret" not in str(error.value)


def test_mailbox_store_uses_internal_transport_and_whitelists_response(tmp_path, monkeypatch):
    actual_client = httpx.Client
    received = []
    def handler(request):
        received.append(request)
        return httpx.Response(201, json={"mailbox_record_id": "mailbox:1", "credential_id": "credential:1",
                                        "credential_expires_at": "2026-10-20T00:00:00Z", "password": "never return"})
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr("services.mailbox.httpx.Client", lambda **kw: actual_client(transport=transport, **kw))
    store = PhoenixMailboxCredentialStore(settings(tmp_path, phoenix_base_url="http://127.0.0.1:4000",
                                                     itsm_internal_job_token="unit-internal-token"))
    result = store.persist(email="alice@itsm.test", password="unit-password")
    assert "password" not in result
    assert received[0].url.path == "/api/internal/v1/provisioned-mailboxes"
    assert received[0].headers["x-internal-token"] == "unit-internal-token"
    assert json.loads(received[0].content) == {"provider": "maddy", "email": "alice@itsm.test",
        "temporary_password": "unit-password", "verification": {"smtp": False, "imap": False}}


def test_remote_plain_http_vault_is_denied_before_dispatch(tmp_path):
    store = PhoenixMailboxCredentialStore(settings(tmp_path, phoenix_base_url="http://example.com",
                                                     itsm_internal_job_token="unit-token"))
    with pytest.raises(RuntimeError, match="HTTPS"):
        store.validate_configuration()


@pytest.mark.parametrize("found", [True, False])
def test_network_readback_requires_the_submitted_message_id(tmp_path, monkeypatch, found):
    recorded = {}
    class SMTP:
        def __init__(self, host, port, *, timeout):
            assert host == "127.0.0.1" and port == 1587
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def login(self, email, password):
            recorded["smtp_login"] = (email, password)
        def send_message(self, message):
            assert message["From"] == message["To"] == "alice@itsm.test"
            recorded["message_id"] = str(message["Message-ID"])
            return {}
    class IMAP:
        error = ValueError
        def __init__(self, host, port, *, timeout):
            assert host == "127.0.0.1" and port == 1143
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def login(self, email, password):
            assert (email, password) == recorded["smtp_login"]
            return "OK", []
        def select(self, mailbox, readonly=False):
            assert mailbox == "INBOX" and readonly
            return "OK", []
        def search(self, charset, header, name, message_id):
            assert (charset, header, name) == (None, "HEADER", "Message-ID")
            assert message_id == recorded["message_id"]
            return "OK", [b"1" if found else b""]
    monkeypatch.setattr("services.mailbox.smtplib.SMTP", SMTP)
    monkeypatch.setattr("services.mailbox.imaplib.IMAP4", IMAP)
    service = MaddyMailboxService(settings(tmp_path), Store())
    if found:
        service._verify("alice@itsm.test", "unit-password")
    else:
        with pytest.raises(MailboxProviderError, match="not read back"):
            service._verify("alice@itsm.test", "unit-password")
