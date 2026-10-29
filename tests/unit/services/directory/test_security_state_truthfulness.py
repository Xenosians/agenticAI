"""Existing production LDAP adapters must not turn unreadable flags into zero."""
from types import SimpleNamespace

import pytest

from services.directory.ldap import DirectoryReadError, LdapDirectoryService
from services.directory.account_lifecycle import LdapAccountLifecycleService
from tools.account.mcp import register_account_lifecycle_tools


class Entry(dict):
    entry_dn = "CN=audit_user,DC=example,DC=test"


def entry(**values):
    return Entry({key: SimpleNamespace(raw_values=value) for key, value in values.items()})


class Connection:
    def __init__(self, *, modify_error=None):
        self.writes = []
        self.result = {"result": 0}
        self.unbound = False
        self.modify_error = modify_error

    def modify(self, dn, changes):
        self.writes.append((dn, changes))
        if self.modify_error:
            raise self.modify_error
        return True

    def unbind(self):
        self.unbound = True


class Directory(LdapDirectoryService):
    def __init__(self, *entries, modify_error=None):
        # Deliberately no Settings instance, directory connection or .env access.
        self.connection = Connection(modify_error=modify_error)
        self.entries = list(entries)

    def _connect(self):
        return self.connection

    def _connect_write(self):
        return self.connection

    def _find_user(self, connection, user_id):
        assert connection is self.connection
        assert user_id == "audit_user"
        return self.entries.pop(0)


@pytest.mark.parametrize("values", [None, [], [None], [b"not-an-integer"], [b"-1"], [b"512.0"], [b"512", b"514"], [True], [512.0], [b"\xff"], [str(2**64).encode()]])
@pytest.mark.parametrize("field", ["userAccountControl", "lockoutTime"])
def test_security_integer_must_be_observed_and_well_formed(field, values):
    observed = entry(**{field: values}) if values is not None else Entry()
    with pytest.raises(DirectoryReadError, match="readable"):
        LdapDirectoryService._get_raw_integer_attribute(observed, field)


def test_optional_metadata_fallback_is_explicit_and_separate_from_security_state():
    assert LdapDirectoryService._get_raw_integer_attribute(Entry(), "primaryGroupID", 0) == 0
    for field in ("userAccountControl", "lockoutTime"):
        with pytest.raises(DirectoryReadError):
            LdapDirectoryService._get_raw_integer_attribute(Entry(), field, 0)


@pytest.mark.parametrize("field", ["userAccountControl", "lockoutTime"])
def test_account_status_cannot_report_enabled_unlocked_when_either_flag_is_missing(field):
    observed = entry(userAccountControl=[b"512"], lockoutTime=[b"0"])
    del observed[field]
    directory = Directory(observed)
    result = directory.account_status("audit_user")
    assert result["ok"] is False
    assert result["enabled"] is None and result["locked"] is None
    assert directory.connection.writes == []
    assert directory.connection.unbound


@pytest.mark.parametrize("enabled,locked", [(True, False), (False, False), (True, True)])
def test_account_status_preserves_explicit_observed_flags(enabled, locked):
    directory = Directory(entry(userAccountControl=[b"512" if enabled else b"514"], lockoutTime=[b"123" if locked else b"0"]))
    result = directory.account_status("audit_user")
    assert result["ok"] is True
    assert result["enabled"] is enabled and result["locked"] is locked


@pytest.mark.parametrize("operation", ["enable_user", "disable_user"])
def test_lifecycle_refuses_missing_uac_without_writing(operation):
    directory = Directory(Entry())
    result = getattr(LdapAccountLifecycleService(directory), operation)("audit_user")
    assert result.ok is False and result.status == "error"
    assert result.enabled is None and result.changed is False
    assert result.retry_safe is True
    assert directory.connection.writes == []


@pytest.mark.parametrize("operation,before", [("enable_user", 514), ("disable_user", 512)])
def test_lifecycle_unreadable_post_write_state_stays_unknown(operation, before):
    directory = Directory(entry(userAccountControl=[str(before).encode()]), Entry())
    result = getattr(LdapAccountLifecycleService(directory), operation)("audit_user")
    assert result.ok is False and result.status == "outcome_unknown"
    assert result.enabled is None and result.changed is None
    assert result.mutation_performed is True
    assert result.verification_ok is False and result.retry_safe is False
    assert len(directory.connection.writes) == 1


def test_unlock_refuses_missing_lockout_time_without_writing():
    directory = Directory(Entry())
    result = directory.unlock_user("audit_user")
    assert result["ok"] is False and result["status"] == "error"
    assert result["changed"] is False and result["retry_safe"] is True
    assert directory.connection.writes == []


def test_unlock_cannot_verify_an_absent_attribute_after_modify():
    directory = Directory(entry(lockoutTime=[b"123"]), Entry())
    result = directory.unlock_user("audit_user")
    assert result["ok"] is False and result["status"] == "outcome_unknown"
    assert result["changed"] is None and result["retry_safe"] is False
    assert result["mutation_performed"] is True
    assert len(directory.connection.writes) == 1


def test_unlock_success_requires_explicit_zero_read_back():
    directory = Directory(entry(lockoutTime=[b"123"]), entry(lockoutTime=[b"0"]))
    result = directory.unlock_user("audit_user")
    assert result["ok"] is True and result["verification_ok"] is True
    assert result["changed"] is True
    assert len(directory.connection.writes) == 1


@pytest.mark.parametrize("verified", [True, False])
def test_unlock_mcp_serialization_preserves_real_ldap_readback_evidence(verified):
    from mcp_server import MutationResult

    after = entry(lockoutTime=[b"0"]) if verified else Entry()
    directory = Directory(entry(lockoutTime=[b"123"]), after)
    observed = directory.unlock_user("audit_user")
    payload = MutationResult(**observed).model_dump(mode="json")

    assert payload["status"] == ("executed" if verified else "outcome_unknown")
    assert payload["ok"] is verified
    assert payload["changed"] is (True if verified else None)
    assert payload["mutation_performed"] is True
    assert payload["retry_safe"] is False
    # The LDAP adapter reports True only on verified readback; absence is not
    # evidence of verification and must stay unknown at the MCP boundary.
    assert payload["verification_ok"] is observed.get("verification_ok")
    assert len(directory.connection.writes) == 1


def test_unlock_mcp_schema_preserves_explicit_negative_and_absent_evidence():
    from mcp_server import MutationResult

    unknown = MutationResult(
        ok=False, status="outcome_unknown", changed=None,
        mutation_performed=None, verification_ok=False, retry_safe=False,
    ).model_dump(mode="json")
    assert unknown["verification_ok"] is False
    assert unknown["retry_safe"] is False
    assert unknown["mutation_performed"] is None
    unspecified = MutationResult(ok=False, status="error").model_dump(mode="json")
    assert all(unspecified[key] is None for key in (
        "mutation_performed", "verification_ok", "retry_safe",
    ))


def test_unlock_noop_requires_explicit_observed_zero():
    directory = Directory(entry(lockoutTime=[b"0"]))
    result = directory.unlock_user("audit_user")
    assert result["ok"] is True and result["verification_ok"] is True
    assert result["changed"] is False and directory.connection.writes == []


@pytest.mark.parametrize("operation,observed", [("unlock_user", entry(lockoutTime=[b"123"])), ("disable_user", entry(userAccountControl=[b"512"]))])
def test_transport_failure_after_mutation_attempt_never_reports_no_change(operation, observed):
    directory = Directory(observed, modify_error=OSError("connection interrupted"))
    adapter = directory if operation == "unlock_user" else LdapAccountLifecycleService(directory)
    result = getattr(adapter, operation)("audit_user")
    result = result if isinstance(result, dict) else result.model_dump()
    assert result["ok"] is False and result["status"] == "outcome_unknown"
    assert result["changed"] is None and result["retry_safe"] is False
    assert len(directory.connection.writes) == 1


COMPUTED = "msDS-User-Account-Control-Computed"


def test_never_locked_account_is_unlocked_by_the_computed_state():
    # Active Directory and Samba omit lockoutTime until the first lockout;
    # the domain controller still constructs the authoritative state.
    directory = Directory(entry(userAccountControl=[b"512"], **{COMPUTED: [b"8388608"]}))
    result = directory.account_status("audit_user")
    assert result["ok"] is True
    assert (result["enabled"], result["locked"]) == (True, False)


@pytest.mark.parametrize("computed,lockout,locked", [
    ([b"16"], None, True),
    ([b"0"], [b"133000000000000000"], False),  # the lockout duration has elapsed
    ([b"16"], [b"133000000000000000"], True),
])
def test_computed_lockout_bit_is_authoritative(computed, lockout, locked):
    values = {"userAccountControl": [b"512"], COMPUTED: computed}
    if lockout is not None:
        values["lockoutTime"] = lockout
    result = Directory(entry(**values)).account_status("audit_user")
    assert result["ok"] is True and result["locked"] is locked


@pytest.mark.parametrize("computed", [[b"x"], [b"-16"], [b"16", b"0"], [None], [str(2**32).encode()]])
def test_malformed_computed_state_is_unknown_even_with_a_lockout_time(computed):
    directory = Directory(entry(userAccountControl=[b"512"], lockoutTime=[b"0"], **{COMPUTED: computed}))
    result = directory.account_status("audit_user")
    assert result["ok"] is False
    assert result["enabled"] is None and result["locked"] is None


def test_unlock_of_a_never_locked_account_is_an_observed_noop():
    directory = Directory(entry(**{COMPUTED: [b"0"]}))
    result = directory.unlock_user("audit_user")
    assert result["ok"] is True and result["changed"] is False
    assert result["verification_ok"] is True
    assert directory.connection.writes == []


def test_unlock_success_requires_zero_read_back_and_a_cleared_lockout_bit():
    directory = Directory(entry(lockoutTime=[b"123"], **{COMPUTED: [b"16"]}),
                          entry(lockoutTime=[b"0"], **{COMPUTED: [b"0"]}))
    result = directory.unlock_user("audit_user")
    assert result["ok"] is True and result["changed"] is True
    assert len(directory.connection.writes) == 1


def test_unlock_is_unknown_when_the_directory_still_reports_a_lockout():
    directory = Directory(entry(lockoutTime=[b"123"], **{COMPUTED: [b"16"]}),
                          entry(lockoutTime=[b"0"], **{COMPUTED: [b"16"]}))
    result = directory.unlock_user("audit_user")
    assert result["ok"] is False and result["status"] == "outcome_unknown"
    assert result["changed"] is None and result["retry_safe"] is False


def test_account_reads_request_the_computed_state():
    from services.directory.ldap import ACCOUNT_SEARCH_ATTRIBUTES

    assert COMPUTED in ACCOUNT_SEARCH_ATTRIBUTES

    class Recorder:
        entries = []

        def search(self, **request):
            self.request = request

    directory = Directory()
    directory.base_dn = "DC=example,DC=test"
    recorder = Recorder()
    assert LdapDirectoryService._find_user(directory, recorder, "audit_user") is None
    assert COMPUTED in recorder.request["attributes"]


def test_mcp_lifecycle_wrapper_preserves_uncertain_outcome_fields():
    functions = {}
    server = SimpleNamespace(tool=lambda: lambda function: functions.setdefault(function.__name__, function))
    directory = Directory(entry(userAccountControl=[b"514"]), Entry())
    register_account_lifecycle_tools(server, LdapAccountLifecycleService(directory))
    result = functions["enable_user"]("audit_user")
    assert result.status == "outcome_unknown" and result.changed is None
    assert result.verification_ok is False and result.retry_safe is False
