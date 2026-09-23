from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from config import Settings


class CredentialPersistenceError(RuntimeError):
    """Safe, non-secret credential persistence failure."""


@dataclass(frozen=True)
class ProvisionedAccountPersistenceResult:
    account_record_id: str
    credential_id: str
    credential_expires_at: str


class PhoenixCredentialStore:
    """
    Internal AI -> Phoenix credential handoff.

    Plaintext exists only in the caller and HTTP request body long enough
    for Phoenix to encrypt it. The response never contains the secret.
    """

    def __init__(self, settings: Settings) -> None:
        # Resolve deployment secrets lazily at the moment of persistence.
        # This keeps read-only/model-eval startup independent of Phoenix.
        self._settings = settings

    @staticmethod
    def _require_safe_transport(base_url: str) -> None:
        parsed = urlsplit(base_url)
        host = (parsed.hostname or "").lower()
        local_hosts = {"127.0.0.1", "localhost", "::1"}

        if parsed.scheme == "https":
            return

        if parsed.scheme == "http" and host in local_hosts:
            return

        raise RuntimeError(
            "Credential persistence requires HTTPS unless Phoenix is "
            "running on localhost."
        )

    def persist_provisioned_account(
        self,
        *,
        directory_user_id: str,
        email: str,
        given_name: str,
        family_name: str,
        department: str | None,
        role: str | None,
        temporary_password: str,
    ) -> ProvisionedAccountPersistenceResult:
        base_url = self._settings.require_phoenix_base_url()
        token = self._settings.require_internal_job_token()
        timeout = self._settings.phoenix_http_timeout_seconds
        self._require_safe_transport(base_url)

        url = f"{base_url}/api/internal/v1/provisioned-accounts"

        payload = {
            "directory_user_id": directory_user_id,
            "email": email,
            "given_name": given_name,
            "family_name": family_name,
            "department": department,
            "role": role,
            "temporary_password": temporary_password,
        }

        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    url,
                    headers={
                        "x-internal-token": token,
                    },
                    json=payload,
                )
        except httpx.HTTPError as exc:
            raise CredentialPersistenceError(
                "Phoenix credential persistence transport failed."
            ) from exc

        if response.status_code not in range(200, 300):
            raise CredentialPersistenceError(
                "Phoenix rejected credential persistence."
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise CredentialPersistenceError(
                "Phoenix returned an invalid credential persistence response."
            ) from exc

        if not isinstance(body, dict):
            raise CredentialPersistenceError(
                "Phoenix returned an invalid credential persistence response."
            )

        required = (
            "account_record_id",
            "credential_id",
            "credential_expires_at",
        )

        values: dict[str, str] = {}
        for key in required:
            value = body.get(key)
            if not isinstance(value, str) or not value.strip():
                raise CredentialPersistenceError(
                    "Phoenix returned an incomplete credential persistence response."
                )
            values[key] = value.strip()

        return ProvisionedAccountPersistenceResult(
            account_record_id=values["account_record_id"],
            credential_id=values["credential_id"],
            credential_expires_at=values["credential_expires_at"],
        )


def build_credential_store(settings: Settings) -> PhoenixCredentialStore:
    return PhoenixCredentialStore(settings)
