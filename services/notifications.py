"""Approved notifications to registered application users through the backend mail outbox."""

from __future__ import annotations

import hashlib
import re
import secrets

import httpx

from config import Settings
from services.credentials.phoenix import PhoenixCredentialStore


NOTIFICATION_PATH = "/api/internal/v1/notifications"
MAX_MESSAGE_LENGTH = 4000
MAX_SUBJECT_LENGTH = 160
EMAIL_PATTERN = re.compile(r"[^\s@]+@[^\s@]+\.[^\s@]+")
DEDUPE_KEY_PATTERN = re.compile(r"[0-9a-f]{64}")
CONTROL_PATTERN = re.compile(r"[\x00-\x1f\x7f]")
# Failures raised before any byte of the request left this process.
NOT_SENT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)


def validate_notification(email: object, message: object, subject: object = None) -> str | None:
    """Why the request cannot be sent, or None. Phoenix repeats these checks."""
    if not isinstance(email, str) or len(email) > 320 or not EMAIL_PATTERN.fullmatch(email.strip()):
        return "email must be one exact email address."
    if not isinstance(message, str) or not message.strip():
        return "message must be non-empty text."
    if len(message) > MAX_MESSAGE_LENGTH:
        return f"message must be at most {MAX_MESSAGE_LENGTH} characters."
    if subject is not None and (not isinstance(subject, str) or not subject.strip()
                                or len(subject) > MAX_SUBJECT_LENGTH or CONTROL_PATTERN.search(subject)):
        return f"subject must be one line of at most {MAX_SUBJECT_LENGTH} characters."
    return None


def new_dedupe_key(email: str, message: str, subject: str | None = None) -> str:
    # A fresh nonce per prepared request: replaying one approval reuses its key,
    # while a later request with the same text is a new notification.
    nonce = secrets.token_hex(16)
    return hashlib.sha256("\n".join((email, subject or "", message, nonce)).encode()).hexdigest()


def expected_outbox_id(dedupe_key: str) -> str:
    # Phoenix namespaces the key and derives the outbox id from it.
    return hashlib.sha256(("user_notification:" + dedupe_key).encode()).hexdigest()


class UserNotificationService:

    def __init__(self, settings: Settings, *, transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings
        self._transport = transport

    def validate_configuration(self) -> None:
        PhoenixCredentialStore._require_safe_transport(self.settings.require_phoenix_base_url())
        self.settings.require_internal_job_token()

    def notify(self, *, email: str, message: str, subject: str | None = None, dedupe_key: str | None = None) -> dict:
        def result(ok: bool, status: str, error_code: str | None = None, error: str | None = None, **extra) -> dict:
            return {"ok": ok, "status": status, "provider": "phoenix_outbox", "operation": "notify_user",
                    "email": email, "queued": False, "error_code": error_code, "error": error, **extra}

        if not isinstance(dedupe_key, str) or not DEDUPE_KEY_PATTERN.fullmatch(dedupe_key):
            return result(False, "denied", "missing_trusted_preparation",
                          "notify_user needs the dedupe key prepared by trusted policy before approval.")
        invalid = validate_notification(email, message, subject)
        if invalid:
            return result(False, "denied", "invalid_arguments", invalid)
        try:
            self.validate_configuration()
        except RuntimeError as exc:
            return result(False, "error", "not_configured", str(exc))

        payload = {"email": email, "message": message, "dedupe_key": dedupe_key}
        if subject is not None:
            payload["subject"] = subject
        outbox_id = expected_outbox_id(dedupe_key)
        try:
            # trust_env=False: the internal token never goes through an environment proxy.
            with httpx.Client(transport=self._transport, timeout=self.settings.phoenix_http_timeout_seconds,
                              trust_env=False) as client:
                response = client.post(self.settings.require_phoenix_base_url() + NOTIFICATION_PATH,
                                       headers={"x-internal-token": self.settings.require_internal_job_token()},
                                       json=payload)
        except NOT_SENT_ERRORS:
            return result(False, "error", "provider_unavailable", "The backend could not be reached; nothing was queued.")
        except httpx.HTTPError:
            return result(False, "outcome_unknown", "outcome_unknown",
                          "The backend did not answer; the notification may or may not be queued.", outbox_id=outbox_id)

        try:
            body = response.json()
        except ValueError:
            body = None
        body = body if isinstance(body, dict) else {}
        code = response.status_code
        delivery_status = body.get("delivery_status")
        matching_receipt = body.get("outbox_id") == outbox_id
        reconciliation_required = body.get("reconciliation_required")
        if code in (200, 202):
            expected = {"pending": (202, True), "processing": (202, False), "sent": (200, False)}
            if (isinstance(delivery_status, str) and delivery_status in expected
                    and (code, body.get("queued")) == expected[delivery_status]
                    and type(body.get("queued")) is bool
                    and reconciliation_required is False and matching_receipt):
                return result(True, "executed", queued=body["queued"], outbox_id=outbox_id,
                              delivery_status=delivery_status, reconciliation_required=False, retry_safe=False)
            return result(False, "outcome_unknown", "outcome_unknown",
                          "The backend response did not confirm this notification's delivery state.",
                          outbox_id=outbox_id, reconciliation_required=True, retry_safe=False)
        if code == 404 and body.get("error") == "recipient_not_found":
            return result(False, "denied", "recipient_not_found", "No registered application user has that email address.")
        if code == 404:
            return result(False, "error", "endpoint_unavailable",
                          "The backend does not offer the notification endpoint; nothing was queued.")
        if code == 422:
            field = body.get("field")
            return result(False, "denied", "invalid_arguments",
                          f"The backend rejected the {field if isinstance(field, str) else 'request'}.")
        if code == 409:
            if (matching_receipt and body.get("queued") is False
                    and delivery_status == "outcome_unknown"
                    and reconciliation_required is True
                    and body.get("error") == "notification_delivery_unknown"):
                return result(False, "outcome_unknown", "notification_delivery_unknown",
                              "The existing notification's delivery is unknown; reconcile its outbox before another send.",
                              outbox_id=outbox_id, delivery_status=delivery_status,
                              reconciliation_required=True, retry_safe=False)
            if (matching_receipt and body.get("queued") is False and delivery_status == "dead"
                    and reconciliation_required is False
                    and body.get("error") == "notification_delivery_failed"):
                return result(False, "error", "notification_delivery_failed",
                              "The existing notification failed and is not queued. Recovery requires an explicit operator decision.",
                              outbox_id=outbox_id, delivery_status=delivery_status,
                              reconciliation_required=False, retry_safe=False)
            if body.get("error") != "dedupe_conflict":
                return result(False, "outcome_unknown", "outcome_unknown",
                              "The backend returned an unrecognized notification conflict; check its outbox.",
                              outbox_id=outbox_id, reconciliation_required=True, retry_safe=False)
            return result(False, "denied", "dedupe_conflict",
                          "A different notification was already queued with this dedupe key.")
        if code in (401, 403):
            return result(False, "error", "provider_auth_failed", "The backend rejected the internal credential.")
        if code >= 500:
            return result(False, "outcome_unknown", "outcome_unknown",
                          f"The backend failed (HTTP {code}); the notification may or may not be queued.",
                          outbox_id=outbox_id)
        return result(False, "error", "provider_error", f"The backend refused the notification (HTTP {code}).")
