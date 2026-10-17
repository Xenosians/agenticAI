"""Approved password reset for one existing directory account.

Trusted policy resolves and freezes the exact target before approval and refuses
protected or service accounts. After approval, trusted code generates the
temporary password, sets it with "must change at next logon", verifies the
change by read-back, and hands the credential to Phoenix for encrypted storage.
The password never reaches the model, the approval, or the tool result.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ldap3 import MODIFY_REPLACE
from ldap3.utils.conv import escape_filter_chars
from pydantic import BaseModel

from config import Settings
from services.credentials import CredentialPersistenceError, PhoenixCredentialStore

from .account_creation import generate_temporary_password
from .base import DirectoryService
from .ldap import LdapDirectoryService
from .mock import MockDirectoryService

TARGET_ATTRIBUTES = [
    "sAMAccountName",
    "objectGUID",
    "adminCount",
    "isCriticalSystemObject",
    "uSNChanged",
    "pwdLastSet",
]


class PasswordResetResult(BaseModel):
    ok: bool
    status: str
    changed: bool = False
    operation: str = "reset_password"
    user_id: str | None = None
    must_change_at_next_logon: bool | None = None
    credential_persisted: bool = False
    credential_id: str | None = None
    credential_expires_at: str | None = None
    retry_safe: bool = True
    message: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class ResetTarget:
    user_id: str
    dn: str
    object_guid: str
    protected: bool
    usn_changed: int
    pwd_last_set: int


def _raw(entry: Any, name: str) -> bytes | None:
    try:
        values = entry[name].raw_values
    except Exception:
        return None
    return values[0] if values else None


def _raw_int(entry: Any, name: str) -> int:
    value = _raw(entry, name)
    return int(value) if value else 0


def read_reset_target(connection: Any, base_dn: str, user_id: str) -> ResetTarget | None:
    connection.search(
        search_base=base_dn,
        search_filter=f"(&(objectClass=user)(sAMAccountName={escape_filter_chars(user_id)}))",
        attributes=TARGET_ATTRIBUTES,
    )
    if len(connection.entries) != 1:
        return None
    entry = connection.entries[0]
    guid = _raw(entry, "objectGUID")
    critical = (_raw(entry, "isCriticalSystemObject") or b"").upper() == b"TRUE"
    return ResetTarget(
        user_id=(_raw(entry, "sAMAccountName") or user_id.encode()).decode(),
        dn=str(entry.entry_dn),
        object_guid=str(uuid.UUID(bytes_le=guid)) if guid and len(guid) == 16 else "",
        protected=_raw_int(entry, "adminCount") == 1 or critical,
        usn_changed=_raw_int(entry, "uSNChanged"),
        pwd_last_set=_raw_int(entry, "pwdLastSet"),
    )


def _account_name(value: str | None) -> str:
    """Account part of DOMAIN\\name, name@domain or a bare name."""
    name = (value or "").strip().rsplit("\\", 1)[-1].split("@", 1)[0]
    return name.casefold()


def is_service_account(target: ResetTarget, settings: Settings) -> bool:
    """The directory identities this application binds with must never be reset here."""
    dns = {(settings.ad_bind_dn or "").casefold(), (settings.ad_write_bind_dn or "").casefold()} - {""}
    names = {_account_name(settings.ad_bind_user), _account_name(settings.ad_write_bind_user)} - {""}
    return target.dn.casefold() in dns or target.user_id.casefold() in names


def _denied(error: str) -> dict[str, Any]:
    return {"ok": False, "status": "denied", "risk": "high", "requires_approval": True, "error": error}


def prepare_password_reset(directory: DirectoryService, settings: Settings, *, user_id: str) -> dict[str, Any]:
    """Read-only pre-approval policy; freezes the exact target account."""
    if not isinstance(user_id, str) or not user_id.strip() or user_id != user_id.strip():
        return _denied("user_id must be an exact, non-empty account name.")
    if isinstance(directory, MockDirectoryService):
        if directory._normalize_user(user_id) not in directory.users:
            return _denied(f"User '{user_id}' was not found.")
        frozen = {"expected_object_guid": f"mock:{user_id}", "expected_dn": f"CN={user_id},CN=Mock"}
    elif isinstance(directory, LdapDirectoryService):
        connection = directory._connect()
        try:
            target = read_reset_target(connection, directory.base_dn, user_id)
        finally:
            connection.unbind()
        if target is None or not target.object_guid:
            return _denied(f"User '{user_id}' was not found.")
        if target.protected:
            return _denied("Protected administrative accounts cannot be reset through this workflow.")
        if is_service_account(target, settings):
            return _denied("The application's own directory service accounts cannot be reset.")
        frozen = {"expected_object_guid": target.object_guid, "expected_dn": target.dn}
    else:
        return _denied("Password reset is not supported for this directory backend.")
    return {
        "ok": True,
        "status": "ready",
        "risk": "high",
        "requires_approval": True,
        "execution_arguments": {"user_id": user_id, **frozen},
        "error": None,
    }


def _persist(credential_store: PhoenixCredentialStore, user_id: str, password: str) -> PasswordResetResult:
    try:
        persisted = credential_store.persist_reset_credential(directory_user_id=user_id, temporary_password=password)
    except (CredentialPersistenceError, RuntimeError):
        return PasswordResetResult(
            ok=False, status="incomplete", changed=True, user_id=user_id, must_change_at_next_logon=True,
            retry_safe=False,
            error=("The password was reset, but the temporary credential could not be stored. "
                   "Run another approved reset to issue a usable credential."),
        )
    return PasswordResetResult(
        ok=True, status="executed", changed=True, user_id=user_id, must_change_at_next_logon=True,
        credential_persisted=True, credential_id=persisted.credential_id,
        credential_expires_at=persisted.credential_expires_at, retry_safe=False,
        message=(f"Password for '{user_id}' was reset. The temporary credential is encrypted by Phoenix "
                 "and must be changed at next sign-in."),
    )


class PasswordResetService(ABC):
    @abstractmethod
    def reset_password(self, user_id: str, *, expected_object_guid: str, expected_dn: str) -> PasswordResetResult:
        raise NotImplementedError


class MockPasswordResetService(PasswordResetService):
    def __init__(self, directory: MockDirectoryService, credential_store: PhoenixCredentialStore,
                 settings: Settings) -> None:
        self.directory = directory
        self.credential_store = credential_store
        self.settings = settings

    def reset_password(self, user_id: str, *, expected_object_guid: str, expected_dn: str) -> PasswordResetResult:
        if expected_object_guid != f"mock:{user_id}" or expected_dn != f"CN={user_id},CN=Mock":
            return PasswordResetResult(ok=False, status="denied", user_id=user_id,
                                       error="The approved target no longer matches this account.")
        result = self.directory.reset_password(user_id)
        if not result.get("ok"):
            return PasswordResetResult(ok=False, status="error", user_id=user_id, error=result.get("error"))
        password = generate_temporary_password(self.settings.account_temporary_password_length)
        return _persist(self.credential_store, user_id, password)


class LdapPasswordResetService(PasswordResetService):
    def __init__(self, directory: LdapDirectoryService, credential_store: PhoenixCredentialStore,
                 settings: Settings) -> None:
        self.directory = directory
        self.credential_store = credential_store
        self.settings = settings

    def reset_password(self, user_id: str, *, expected_object_guid: str, expected_dn: str) -> PasswordResetResult:
        connection = None
        attempted = False
        try:
            connection = self.directory._connect_write()
            target = read_reset_target(connection, self.directory.base_dn, user_id)
            if (target is None or target.object_guid != expected_object_guid
                    or target.dn.casefold() != expected_dn.casefold()):
                return PasswordResetResult(ok=False, status="denied", user_id=user_id,
                                           error="The approved target no longer matches this account.")
            if target.protected or is_service_account(target, self.settings):
                return PasswordResetResult(ok=False, status="denied", user_id=user_id,
                                           error="This account is protected from password reset.")
            password = generate_temporary_password(self.settings.account_temporary_password_length)
            attempted = True
            changed = connection.modify(target.dn, {
                "unicodePwd": [(MODIFY_REPLACE, [f'"{password}"'.encode("utf-16-le")])],
                "pwdLastSet": [(MODIFY_REPLACE, [0])],
            })
            if not changed:
                return PasswordResetResult(ok=False, status="error", user_id=user_id,
                                           error="Active Directory rejected the password reset.")
            after = read_reset_target(connection, self.directory.base_dn, user_id)
            if (after is None or after.object_guid != target.object_guid or after.pwd_last_set != 0
                    or after.usn_changed <= target.usn_changed):
                return PasswordResetResult(
                    ok=False, status="outcome_unknown", changed=True, user_id=user_id, retry_safe=False,
                    error="The reset was accepted but read-back could not verify the change.",
                )
        except Exception:
            return PasswordResetResult(
                ok=False, status="outcome_unknown" if attempted else "error", changed=attempted,
                user_id=user_id, retry_safe=not attempted,
                error="The directory password reset failed." if not attempted
                else "The directory password reset outcome is unknown; check the account before retrying.",
            )
        finally:
            if connection is not None:
                connection.unbind()
        return _persist(self.credential_store, target.user_id, password)


def build_password_reset_service(directory: DirectoryService, credential_store: PhoenixCredentialStore,
                                 settings: Settings) -> PasswordResetService:
    if isinstance(directory, MockDirectoryService):
        return MockPasswordResetService(directory, credential_store, settings)
    if isinstance(directory, LdapDirectoryService):
        return LdapPasswordResetService(directory, credential_store, settings)
    raise RuntimeError(f"Unsupported DirectoryService implementation for password reset: {type(directory).__name__}")
