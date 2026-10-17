from dataclasses import dataclass, field

import httpx
import pytest

from config import Settings
from services.credentials import CredentialPersistenceError, CredentialPersistenceResult, PhoenixCredentialStore
from services.directory.mock import MockDirectoryService
from services.directory.password_reset import (
    LdapPasswordResetService,
    MockPasswordResetService,
    ResetTarget,
    is_service_account,
    prepare_password_reset,
)
from tools.account.catalog import ACCOUNT_TOOLS, format_password_reset_result
from tools.registry import TOOLS


@dataclass
class RecordingStore:
    calls: list = field(default_factory=list)
    fail: bool = False

    def persist_reset_credential(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise CredentialPersistenceError("Phoenix rejected credential persistence.")
        return CredentialPersistenceResult(credential_id="cred-1", credential_expires_at="2026-10-18T00:00:00Z")


def _settings(**overrides) -> Settings:
    return Settings(directory_backend="mock", account_temporary_password_length=24, **overrides)


def test_policy_freezes_the_exact_target_and_rejects_unknown_accounts():
    directory = MockDirectoryService()
    ready = prepare_password_reset(directory, _settings(), user_id="jdoe")
    assert ready["ok"] is True and ready["requires_approval"] is True
    assert ready["execution_arguments"] == {
        "user_id": "jdoe", "expected_object_guid": "mock:jdoe", "expected_dn": "CN=jdoe,CN=Mock",
    }
    assert prepare_password_reset(directory, _settings(), user_id="ghost")["status"] == "denied"
    assert prepare_password_reset(directory, _settings(), user_id=" jdoe")["status"] == "denied"


def test_reset_generates_the_secret_after_approval_and_never_returns_it():
    store = RecordingStore()
    service = MockPasswordResetService(MockDirectoryService(), store, _settings())
    result = service.reset_password("jdoe", expected_object_guid="mock:jdoe", expected_dn="CN=jdoe,CN=Mock")
    assert result.ok is True and result.credential_persisted is True and result.must_change_at_next_logon
    secret = store.calls[0]["temporary_password"]
    assert len(secret) == 24 and store.calls[0]["directory_user_id"] == "jdoe"
    assert secret not in result.model_dump_json()
    assert secret not in format_password_reset_result(result.model_dump())


def test_reset_refuses_a_target_that_changed_after_approval():
    store = RecordingStore()
    service = MockPasswordResetService(MockDirectoryService(), store, _settings())
    result = service.reset_password("jdoe", expected_object_guid="mock:asmith", expected_dn="CN=jdoe,CN=Mock")
    assert result.status == "denied" and store.calls == []


def test_failed_credential_handoff_is_reported_as_incomplete_without_retry():
    service = MockPasswordResetService(MockDirectoryService(), RecordingStore(fail=True), _settings())
    result = service.reset_password("jdoe", expected_object_guid="mock:jdoe", expected_dn="CN=jdoe,CN=Mock")
    assert result.ok is False and result.status == "incomplete" and result.retry_safe is False


def _target(**overrides) -> ResetTarget:
    values = dict(user_id="svc-itsm", dn="CN=svc-itsm,CN=Users,DC=itsm,DC=local", object_guid="g",
                  protected=False, usn_changed=1, pwd_last_set=1)
    values.update(overrides)
    return ResetTarget(**values)


@pytest.mark.parametrize("settings", [
    {"ad_write_bind_user": "ITSM\\svc-itsm"},
    {"ad_bind_user": "svc-itsm@itsm.local"},
    {"ad_write_bind_dn": "cn=SVC-ITSM,cn=users,dc=itsm,dc=local"},
])
def test_application_service_accounts_are_never_reset(settings):
    assert is_service_account(_target(), Settings(directory_backend="mock", **settings))


def test_ordinary_accounts_are_not_service_accounts():
    assert not is_service_account(_target(user_id="jdoe", dn="CN=jdoe,CN=Users,DC=itsm,DC=local"),
                                  Settings(directory_backend="mock", ad_write_bind_user="ITSM\\svc-itsm"))


class _Entry:
    def __init__(self, dn, attrs):
        self.entry_dn = dn
        self._attrs = attrs

    def __getitem__(self, name):
        class Attr:
            raw_values = [self._attrs[name]] if name in self._attrs else []
        return Attr()


class _Connection:
    """Simulated directory: the modify call changes pwdLastSet and the USN."""

    def __init__(self, admin=False):
        guid = bytes(range(16))
        self.state = {"sAMAccountName": b"jdoe", "objectGUID": guid, "uSNChanged": b"10", "pwdLastSet": b"5"}
        if admin:
            self.state["adminCount"] = b"1"
        self.entries = []
        self.modified = []

    def search(self, **kwargs):
        self.entries = [_Entry("CN=jdoe,CN=Users,DC=itsm,DC=local", dict(self.state))]

    def modify(self, dn, changes):
        self.modified.append((dn, sorted(changes)))
        self.state.update(uSNChanged=b"11", pwdLastSet=b"0")
        return True

    def unbind(self):
        pass


class _Directory:
    base_dn = "DC=itsm,DC=local"

    def __init__(self, connection):
        self.connection = connection

    def _connect_write(self):
        return self.connection


GUID = "03020100-0504-0706-0809-0a0b0c0d0e0f"


def test_ldap_reset_sets_password_requires_change_and_verifies():
    connection = _Connection()
    store = RecordingStore()
    service = LdapPasswordResetService(_Directory(connection), store, _settings())
    result = service.reset_password("jdoe", expected_object_guid=GUID, expected_dn="CN=jdoe,CN=Users,DC=itsm,DC=local")
    assert result.ok is True, result
    assert connection.modified == [("CN=jdoe,CN=Users,DC=itsm,DC=local", ["pwdLastSet", "unicodePwd"])]
    assert store.calls[0]["directory_user_id"] == "jdoe"


def test_ldap_reset_refuses_protected_accounts_even_after_approval():
    connection = _Connection(admin=True)
    service = LdapPasswordResetService(_Directory(connection), RecordingStore(), _settings())
    result = service.reset_password("jdoe", expected_object_guid=GUID, expected_dn="CN=jdoe,CN=Users,DC=itsm,DC=local")
    assert result.status == "denied" and connection.modified == []


def test_ldap_reset_refuses_a_replaced_account():
    connection = _Connection()
    service = LdapPasswordResetService(_Directory(connection), RecordingStore(), _settings())
    result = service.reset_password("jdoe", expected_object_guid="other-guid",
                                    expected_dn="CN=jdoe,CN=Users,DC=itsm,DC=local")
    assert result.status == "denied" and connection.modified == []


def test_catalog_entry_keeps_the_password_out_of_model_arguments():
    tool = TOOLS["reset_password"]
    assert tool is ACCOUNT_TOOLS["reset_password"]
    assert tool["requires_approval"] is True and tool["risk"] == "high"
    assert list(tool["parameters"]) == ["user_id"]
    assert tool["trusted_policy_arguments"] == ["expected_object_guid", "expected_dn"]


def test_credential_handoff_posts_only_to_the_internal_endpoint(monkeypatch):
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["token"] = request.headers.get("x-internal-token")
        return httpx.Response(201, json={"credential_id": "c1", "credential_expires_at": "t", "directory_user_id": "jdoe"})

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    settings = Settings(directory_backend="mock", phoenix_base_url="http://127.0.0.1:4000",
                        itsm_internal_job_token="t" * 32)
    result = PhoenixCredentialStore(settings).persist_reset_credential(directory_user_id="jdoe",
                                                                        temporary_password="x" * 20)
    assert result == CredentialPersistenceResult(credential_id="c1", credential_expires_at="t")
    assert seen["url"] == "http://127.0.0.1:4000/api/internal/v1/directory-credentials"
    assert seen["token"] == "t" * 32
