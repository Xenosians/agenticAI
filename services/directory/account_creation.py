from __future__ import annotations

import re
import secrets
import string
import unicodedata
from abc import ABC, abstractmethod
from typing import Any

from ldap3 import MODIFY_REPLACE
from pydantic import BaseModel

from config import Settings
from services.credentials import (
    CredentialPersistenceError,
    PhoenixCredentialStore,
)

from .base import DirectoryService
from .ldap import LdapDirectoryService
from .mock import MockDirectoryService


IDENTITY_POLICY_VERSION = "corporate-account-identity.v1"
NORMAL_ACCOUNT_UAC = 0x0200
DISABLED_NORMAL_ACCOUNT_UAC = 0x0202


class AccountCreateResult(BaseModel):
    ok: bool
    status: str
    changed: bool = False
    operation: str = "account_create"
    user_id: str | None = None
    email: str | None = None
    account_created: bool | None = None
    verification_ok: bool = False
    credential_persisted: bool = False
    account_record_id: str | None = None
    credential_id: str | None = None
    credential_expires_at: str | None = None
    retry_safe: bool = True
    message: str | None = None
    error: str | None = None


class AccountCreationService(ABC):
    @abstractmethod
    def create_account(
        self,
        given_name: str,
        family_name: str,
        *,
        department: str | None = None,
        role: str | None = None,
        expected_username: str,
        expected_email: str,
        expected_container_dn: str | None = None,
        identity_policy_version: str = IDENTITY_POLICY_VERSION,
    ) -> AccountCreateResult:
        raise NotImplementedError


def _clean_required(value: str, field: str, max_length: int = 100) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string.")
    if value != value.strip() or not value:
        raise ValueError(f"{field} must be a non-empty exact string.")
    if len(value) > max_length:
        raise ValueError(f"{field} exceeds {max_length} characters.")
    return value


def _clean_optional(value: str | None, field: str, max_length: int = 200) -> str | None:
    if value is None:
        return None
    return _clean_required(value, field, max_length)


def _slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", ascii_value.lower())


def _email_domain(settings: Settings) -> str:
    configured = settings.account_email_domain
    if configured:
        return configured

    base_dn = settings.ad_base_dn or ""
    parts = []
    for component in base_dn.split(","):
        component = component.strip()
        if component.lower().startswith("dc=") and len(component) > 3:
            parts.append(component[3:])

    if parts:
        return ".".join(parts).lower()

    raise RuntimeError(
        "ACCOUNT_EMAIL_DOMAIN is required when it cannot be derived from AD_BASE_DN."
    )


def _container_dn(settings: Settings) -> str | None:
    if settings.ad_account_container_dn:
        return settings.ad_account_container_dn
    if settings.ad_base_dn:
        return f"CN=Users,{settings.ad_base_dn}"
    return None


def _username_base(given_name: str, family_name: str) -> str:
    given = _slug(given_name)
    family = _slug(family_name)
    if not given or not family:
        raise ValueError("Names must contain characters usable in a corporate username.")
    return f"{given[0]}{family}"[:20]


def _user_exists(directory: DirectoryService, user_id: str) -> bool:
    if isinstance(directory, MockDirectoryService):
        return directory._normalize_user(user_id) in directory.users

    if isinstance(directory, LdapDirectoryService):
        connection = None
        try:
            connection = directory._connect()
            return directory._find_user(connection, user_id) is not None
        finally:
            if connection is not None:
                connection.unbind()

    result = directory.account_status(user_id)
    return bool(result.get("ok"))


def prepare_account_create(
    directory: DirectoryService,
    settings: Settings,
    *,
    given_name: str,
    family_name: str,
    department: str | None = None,
    role: str | None = None,
) -> dict[str, Any]:
    given_name = _clean_required(given_name, "given_name")
    family_name = _clean_required(family_name, "family_name")
    department = _clean_optional(department, "department")
    role = _clean_optional(role, "role")

    base = _username_base(given_name, family_name)
    selected: str | None = None

    for suffix in range(1, 100):
        if suffix == 1:
            candidate = base
        else:
            suffix_text = str(suffix)
            candidate = f"{base[:20-len(suffix_text)]}{suffix_text}"

        if not _user_exists(directory, candidate):
            selected = candidate
            break

    if selected is None:
        return {
            "ok": False,
            "status": "denied",
            "risk": "high",
            "requires_approval": True,
            "error": "No available username could be derived under the identity policy.",
        }

    email = f"{selected}@{_email_domain(settings)}"

    execution_arguments: dict[str, Any] = {
        "given_name": given_name,
        "family_name": family_name,
        "expected_username": selected,
        "expected_email": email,
        "identity_policy_version": IDENTITY_POLICY_VERSION,
    }

    if department is not None:
        execution_arguments["department"] = department
    if role is not None:
        execution_arguments["role"] = role

    container = _container_dn(settings)
    if container is not None:
        execution_arguments["expected_container_dn"] = container

    return {
        "ok": True,
        "status": "ready",
        "risk": "high",
        "requires_approval": True,
        "execution_arguments": execution_arguments,
        "error": None,
    }




def _approved_identity_still_valid(
    settings: Settings,
    *,
    given_name: str,
    family_name: str,
    expected_username: str,
    expected_email: str,
    expected_container_dn: str | None,
) -> bool:
    base = _username_base(given_name, family_name)
    username_ok = expected_username == base
    if not username_ok:
        for suffix in range(2, 100):
            suffix_text = str(suffix)
            candidate = f"{base[:20-len(suffix_text)]}{suffix_text}"
            if expected_username == candidate:
                username_ok = True
                break

    if not username_ok:
        return False

    if expected_email != f"{expected_username}@{_email_domain(settings)}":
        return False

    current_container = _container_dn(settings)
    if expected_container_dn != current_container:
        return False

    return True

def generate_temporary_password(length: int) -> str:
    if length < 16:
        raise ValueError("Temporary password length must be at least 16.")

    upper = secrets.choice(string.ascii_uppercase)
    lower = secrets.choice(string.ascii_lowercase)
    digit = secrets.choice(string.digits)
    symbol = secrets.choice("!@#$%^&*()-_=+")
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()-_=+"
    remaining = [secrets.choice(alphabet) for _ in range(length - 4)]
    chars = [upper, lower, digit, symbol, *remaining]

    # secrets.SystemRandom is backed by the OS CSPRNG.
    secrets.SystemRandom().shuffle(chars)
    return "".join(chars)


class MockAccountCreationService(AccountCreationService):
    def __init__(
        self,
        directory: MockDirectoryService,
        credential_store: PhoenixCredentialStore,
        settings: Settings,
    ) -> None:
        self.directory = directory
        self.credential_store = credential_store
        self.settings = settings

    def create_account(
        self,
        given_name: str,
        family_name: str,
        *,
        department: str | None = None,
        role: str | None = None,
        expected_username: str,
        expected_email: str,
        expected_container_dn: str | None = None,
        identity_policy_version: str = IDENTITY_POLICY_VERSION,
    ) -> AccountCreateResult:
        if identity_policy_version != IDENTITY_POLICY_VERSION:
            return AccountCreateResult(
                ok=False,
                status="denied",
                retry_safe=True,
                error="The approved identity policy version is no longer current.",
            )

        if not _approved_identity_still_valid(
            self.settings,
            given_name=given_name,
            family_name=family_name,
            expected_username=expected_username,
            expected_email=expected_email,
            expected_container_dn=expected_container_dn,
        ):
            return AccountCreateResult(
                ok=False,
                status="denied",
                retry_safe=True,
                error="The approved corporate identity no longer matches current policy.",
            )

        if _user_exists(self.directory, expected_username):
            return AccountCreateResult(
                ok=False,
                status="conflict",
                user_id=expected_username,
                email=expected_email,
                account_created=False,
                retry_safe=True,
                error="The approved username is no longer available.",
            )

        password = generate_temporary_password(self.settings.account_temporary_password_length)

        normalized = self.directory._normalize_user(expected_username)
        self.directory.users[normalized] = {
            "enabled": True,
            "locked": False,
            "password_reset_count": 0,
            "given_name": given_name,
            "family_name": family_name,
            "email": expected_email,
            "department": department,
            "role": role,
            "temporary_password_set": True,
            "force_password_change": True,
        }

        state = self.directory.users.get(normalized)
        verified = bool(
            state
            and state.get("email") == expected_email
            and state.get("enabled") is True
        )

        if not verified:
            return AccountCreateResult(
                ok=False,
                status="outcome_unknown",
                changed=True,
                user_id=normalized,
                email=expected_email,
                account_created=True,
                verification_ok=False,
                retry_safe=False,
                error="Mock directory account was written but verification failed.",
            )

        return _persist_created_account(
            credential_store=self.credential_store,
            user_id=normalized,
            email=expected_email,
            given_name=given_name,
            family_name=family_name,
            department=department,
            role=role,
            password=password,
        )


class LdapAccountCreationService(AccountCreationService):
    def __init__(
        self,
        directory: LdapDirectoryService,
        credential_store: PhoenixCredentialStore,
        settings: Settings,
    ) -> None:
        self.directory = directory
        self.credential_store = credential_store
        self.settings = settings

    def create_account(
        self,
        given_name: str,
        family_name: str,
        *,
        department: str | None = None,
        role: str | None = None,
        expected_username: str,
        expected_email: str,
        expected_container_dn: str | None = None,
        identity_policy_version: str = IDENTITY_POLICY_VERSION,
    ) -> AccountCreateResult:
        if identity_policy_version != IDENTITY_POLICY_VERSION:
            return AccountCreateResult(
                ok=False,
                status="denied",
                retry_safe=True,
                error="The approved identity policy version is no longer current.",
            )

        if not _approved_identity_still_valid(
            self.settings,
            given_name=given_name,
            family_name=family_name,
            expected_username=expected_username,
            expected_email=expected_email,
            expected_container_dn=expected_container_dn,
        ):
            return AccountCreateResult(
                ok=False,
                status="denied",
                retry_safe=True,
                error="The approved corporate identity no longer matches current policy.",
            )

        if not self.settings.ad_use_ssl:
            return AccountCreateResult(
                ok=False,
                status="denied",
                retry_safe=True,
                error="Live Active Directory account creation requires LDAPS.",
            )

        container_dn = expected_container_dn or _container_dn(self.settings)
        if not container_dn:
            return AccountCreateResult(
                ok=False,
                status="denied",
                retry_safe=True,
                error="No Active Directory account container is configured.",
            )

        connection = None
        account_created = False
        password = generate_temporary_password(self.settings.account_temporary_password_length)
        display_name = f"{given_name} {family_name}"
        # Use the trusted, policy-derived account identifier as the RDN.
        # This avoids placing free-form display-name text into a DN component.
        user_dn = f"CN={expected_username},{container_dn}"

        try:
            connection = self.directory._connect_write()

            if self.directory._find_user(connection, expected_username) is not None:
                return AccountCreateResult(
                    ok=False,
                    status="conflict",
                    user_id=expected_username,
                    email=expected_email,
                    account_created=False,
                    retry_safe=True,
                    error="The approved username is no longer available.",
                )

            attributes: dict[str, Any] = {
                "sAMAccountName": expected_username,
                "userPrincipalName": expected_email,
                "mail": expected_email,
                "givenName": given_name,
                "sn": family_name,
                "displayName": display_name,
                "userAccountControl": DISABLED_NORMAL_ACCOUNT_UAC,
            }
            if department is not None:
                attributes["department"] = department
            if role is not None:
                attributes["title"] = role

            added = connection.add(
                user_dn,
                object_class=["top", "person", "organizationalPerson", "user"],
                attributes=attributes,
            )

            if not added:
                return AccountCreateResult(
                    ok=False,
                    status="error",
                    user_id=expected_username,
                    email=expected_email,
                    account_created=False,
                    retry_safe=True,
                    error="Active Directory rejected account creation.",
                )

            account_created = True

            quoted_password = f'"{password}"'.encode("utf-16-le")
            password_set = connection.modify(
                user_dn,
                {
                    "unicodePwd": [(MODIFY_REPLACE, [quoted_password])],
                    "pwdLastSet": [(MODIFY_REPLACE, [0])],
                },
            )

            if not password_set:
                return AccountCreateResult(
                    ok=False,
                    status="incomplete",
                    changed=True,
                    user_id=expected_username,
                    email=expected_email,
                    account_created=True,
                    verification_ok=False,
                    retry_safe=False,
                    error=(
                        "The directory account exists, but the temporary password "
                        "could not be set. Do not retry account creation blindly."
                    ),
                )

            enabled = connection.modify(
                user_dn,
                {"userAccountControl": [(MODIFY_REPLACE, [NORMAL_ACCOUNT_UAC])]},
            )

            if not enabled:
                return AccountCreateResult(
                    ok=False,
                    status="incomplete",
                    changed=True,
                    user_id=expected_username,
                    email=expected_email,
                    account_created=True,
                    verification_ok=False,
                    retry_safe=False,
                    error=(
                        "The directory account exists with a password, but enabling "
                        "it failed. Do not retry account creation blindly."
                    ),
                )

            verified = self.directory._find_user(connection, expected_username)
            if verified is None:
                return AccountCreateResult(
                    ok=False,
                    status="outcome_unknown",
                    changed=True,
                    user_id=expected_username,
                    email=expected_email,
                    account_created=True,
                    verification_ok=False,
                    retry_safe=False,
                    error=(
                        "The directory mutation completed but read-back verification "
                        "could not find the account."
                    ),
                )

        except Exception:
            return AccountCreateResult(
                ok=False,
                status="outcome_unknown" if account_created else "error",
                changed=account_created,
                user_id=expected_username,
                email=expected_email,
                account_created=account_created,
                verification_ok=False,
                retry_safe=not account_created,
                error=(
                    "Active Directory account creation failed after a mutation may "
                    "have occurred."
                    if account_created
                    else "Active Directory account creation failed before a verified mutation."
                ),
            )
        finally:
            if connection is not None:
                connection.unbind()

        return _persist_created_account(
            credential_store=self.credential_store,
            user_id=expected_username,
            email=expected_email,
            given_name=given_name,
            family_name=family_name,
            department=department,
            role=role,
            password=password,
        )


def _persist_created_account(
    *,
    credential_store: PhoenixCredentialStore,
    user_id: str,
    email: str,
    given_name: str,
    family_name: str,
    department: str | None,
    role: str | None,
    password: str,
) -> AccountCreateResult:
    try:
        persisted = credential_store.persist_provisioned_account(
            directory_user_id=user_id,
            email=email,
            given_name=given_name,
            family_name=family_name,
            department=department,
            role=role,
            temporary_password=password,
        )
    except (CredentialPersistenceError, RuntimeError):
        return AccountCreateResult(
            ok=False,
            status="incomplete",
            changed=True,
            user_id=user_id,
            email=email,
            account_created=True,
            verification_ok=True,
            credential_persisted=False,
            retry_safe=False,
            error=(
                "The account was created and verified, but the encrypted credential "
                "could not be persisted. Do not retry account creation blindly; "
                "reset the password through a separately approved recovery action."
            ),
        )

    return AccountCreateResult(
        ok=True,
        status="executed",
        changed=True,
        user_id=user_id,
        email=email,
        account_created=True,
        verification_ok=True,
        credential_persisted=True,
        account_record_id=persisted.account_record_id,
        credential_id=persisted.credential_id,
        credential_expires_at=persisted.credential_expires_at,
        retry_safe=False,
        message=(
            f"Corporate account '{user_id}' was created and its temporary credential "
            "was encrypted and persisted by Phoenix."
        ),
    )


def build_account_creation_service(
    directory: DirectoryService,
    credential_store: PhoenixCredentialStore,
    settings: Settings,
) -> AccountCreationService:
    if isinstance(directory, MockDirectoryService):
        return MockAccountCreationService(directory, credential_store, settings)
    if isinstance(directory, LdapDirectoryService):
        return LdapAccountCreationService(directory, credential_store, settings)
    raise RuntimeError(
        "Unsupported DirectoryService implementation for account creation: "
        f"{type(directory).__name__}"
    )
