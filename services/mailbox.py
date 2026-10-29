"""Governed local Maddy mailboxes; directory identities and Brevo stay separate."""
from __future__ import annotations

from services.lua_adapters.http_clients import client as lua_http_client

import imaplib
import os
import re
import secrets
import smtplib
import subprocess
import threading
from contextlib import contextmanager
from email.message import EmailMessage
from pathlib import Path
from typing import Iterator

import httpx
from pydantic import BaseModel

from config import Settings
from services.credentials.phoenix import PhoenixCredentialStore


CONTAINER = "itsm-mailbox-lab"
CONFIG_PATH = "/data/maddy.conf"
_CREATE_LOCK = threading.Lock()


class MailboxResult(BaseModel):
    ok: bool
    status: str
    operation: str
    provider: str = "maddy"
    email: str | None = None
    changed: bool | None = False
    credential_exists: bool | None = None
    mailbox_exists: bool | None = None
    credential_persisted: bool = False
    mailbox_record_id: str | None = None
    credential_id: str | None = None
    credential_expires_at: str | None = None
    smtp_verified: bool = False
    imap_verified: bool = False
    delivery_verified: bool = False
    public_delivery: bool = False
    retry_safe: bool = True
    message: str | None = None
    error: str | None = None


def validate_lab_email(email: str, domain: str) -> str:
    if not isinstance(email, str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9._-]{0,63}@" + re.escape(domain), email
    ) or ".." in email or email.split("@", 1)[0].endswith("."):
        raise ValueError("Supply an exact lowercase email at the configured .test lab domain.")
    return email


class MailboxProviderError(RuntimeError):
    """Only fixed, non-secret messages may cross the provider boundary."""


class PhoenixMailboxCredentialStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def validate_configuration(self) -> None:
        PhoenixCredentialStore._require_safe_transport(self.settings.require_phoenix_base_url())
        self.settings.require_internal_job_token()

    def persist(self, *, email: str, password: str) -> dict[str, str]:
        self.validate_configuration()
        try:
            with lua_http_client(timeout=self.settings.phoenix_http_timeout_seconds) as client:
                response = client.post(
                    self.settings.require_phoenix_base_url() + "/api/internal/v1/provisioned-mailboxes",
                    headers={"x-internal-token": self.settings.require_internal_job_token()},
                    json={"provider": "maddy", "email": email, "temporary_password": password,
                          "verification": {"smtp": False, "imap": False}},
                )
            if response.status_code not in range(200, 300):
                raise MailboxProviderError("Phoenix rejected encrypted mailbox credential persistence.")
            body = response.json()
            required = ("mailbox_record_id", "credential_id", "credential_expires_at")
            if not isinstance(body, dict) or any(
                not isinstance(body.get(key), str) or not body[key].strip() for key in required
            ):
                raise MailboxProviderError("Phoenix returned incomplete mailbox credential metadata.")
            return {key: body[key] for key in required}
        except (httpx.HTTPError, ValueError):
            raise MailboxProviderError("Mailbox credential persistence did not return a verified response.") from None


@contextmanager
def _provisioning_lock(path: Path) -> Iterator[None]:
    """Serialize creates across threads and service processes sharing this lab."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _CREATE_LOCK, path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


class MaddyMailboxService:
    def __init__(self, settings: Settings, credential_store: PhoenixMailboxCredentialStore | None = None) -> None:
        self.settings = settings
        self.credential_store = credential_store or PhoenixMailboxCredentialStore(settings)

    def _admin(self, group: str, action: str, email: str | None = None, *, password: str | None = None) -> str:
        if (group, action) not in {("creds", "list"), ("creds", "create"),
                                   ("imap-acct", "list"), ("imap-acct", "create")}:
            raise MailboxProviderError("Mailbox administration operation is not allowed.")
        if (action == "create") != (email is not None):
            raise MailboxProviderError("Mailbox administration arguments are invalid.")
        if email is not None:
            validate_lab_email(email, self.settings.mailbox_lab_domain)
        if (group == "creds" and action == "create") != (password is not None):
            raise MailboxProviderError("Mailbox credential input is invalid.")
        argv = ["docker", "exec"]
        if password is not None:
            argv.append("-i")
        argv.extend([CONTAINER, "/bin/maddy", "-config", CONFIG_PATH, group, action])
        if email is not None:
            argv.append(email)
        try:
            completed = subprocess.run(
                argv, input=(password + "\n") if password is not None else None,
                capture_output=True, text=True, shell=False,
                timeout=self.settings.mailbox_timeout_seconds, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            raise MailboxProviderError("Maddy administration did not return a verified response.") from None
        if completed.returncode != 0:
            # Neither stdout nor stderr is surfaced: the admin process handled a secret.
            raise MailboxProviderError("Maddy rejected the mailbox administration operation.")
        return completed.stdout if action == "list" else ""

    def _identity_state(self, email: str) -> tuple[bool, bool]:
        credentials = self._admin("creds", "list").splitlines()
        mailboxes = self._admin("imap-acct", "list").splitlines()
        return email in credentials, email in mailboxes

    def _validate(self, email: str) -> None:
        validate_lab_email(email, self.settings.mailbox_lab_domain)
        if not self.settings.mailbox_lab_enabled:
            raise MailboxProviderError("The local mailbox lab is not enabled. Configure and start the Maddy lab first.")

    def status(self, email: str) -> MailboxResult:
        try:
            self._validate(email)
            credential_exists, mailbox_exists = self._identity_state(email)
            return MailboxResult(
                ok=True, status="executed", operation="mailbox_status", email=email,
                credential_exists=credential_exists, mailbox_exists=mailbox_exists,
                message="Local Maddy identity read back. Presence does not prove authenticated delivery or public reachability.",
            )
        except (ValueError, MailboxProviderError) as exc:
            return MailboxResult(ok=False, status="denied", operation="mailbox_status", error=str(exc))

    def _verify(self, email: str, password: str) -> None:
        """One authenticated SMTP submission, then read its unique Message-ID via IMAP."""
        message_id = "<" + secrets.token_hex(20) + "@" + self.settings.mailbox_lab_domain + ">"
        message = EmailMessage()
        message["From"] = email
        message["To"] = email
        message["Subject"] = "ITSM mailbox provisioning verification"
        message["Message-ID"] = message_id
        message.set_content("This local test message verifies SMTP submission and IMAP delivery.")
        from services.lua_adapters.host_boundaries import inspect_link

        inspect_link("smtp", f"smtp://127.0.0.1:{self.settings.mailbox_smtp_port}", operation="submit")
        inspect_link("imap", f"imap://127.0.0.1:{self.settings.mailbox_imap_port}", operation="read")
        try:
            with smtplib.SMTP("127.0.0.1", self.settings.mailbox_smtp_port,
                              timeout=self.settings.mailbox_timeout_seconds) as smtp:
                smtp.login(email, password)
                refused = smtp.send_message(message)
                if refused:
                    raise MailboxProviderError("Maddy did not accept the local verification message.")
            with imaplib.IMAP4("127.0.0.1", self.settings.mailbox_imap_port,
                               timeout=self.settings.mailbox_timeout_seconds) as imap:
                if imap.login(email, password)[0] != "OK" or imap.select("INBOX", readonly=True)[0] != "OK":
                    raise MailboxProviderError("Maddy IMAP authentication or inbox selection failed.")
                status, data = imap.search(None, "HEADER", "Message-ID", message_id)
                if status != "OK" or not data or not data[0].strip():
                    raise MailboxProviderError("The submitted verification message was not read back from IMAP.")
        except (OSError, smtplib.SMTPException, imaplib.IMAP4.error):
            raise MailboxProviderError("Authenticated mailbox delivery verification did not complete.") from None

    def create(self, email: str) -> MailboxResult:
        result = MailboxResult(ok=False, status="denied", operation="mailbox_create", email=email)
        mutation_started = False
        try:
            self._validate(email)
            self.credential_store.validate_configuration()
            with _provisioning_lock(self.settings.mailbox_provisioning_lock_path):
                credentials, mailbox = self._identity_state(email)
                result.credential_exists, result.mailbox_exists = credentials, mailbox
                if credentials or mailbox:
                    result.status = "conflict"
                    result.error = "The email already has credentials or mailbox storage; creation never replaces either."
                    return result
                # This function is invoked by MCP only after the normal gateway approval.
                password = secrets.token_urlsafe(32)
                mutation_started = True
                result.changed = None
                result.credential_exists = None
                self._admin("creds", "create", email, password=password)
                result.changed = True
                result.credential_exists = True
                result.mailbox_exists = None
                self._admin("imap-acct", "create", email)
                result.mailbox_exists = True
                persisted = self.credential_store.persist(email=email, password=password)
                result.credential_persisted = True
                for key, value in persisted.items():
                    setattr(result, key, value)
                self._verify(email, password)
                result.smtp_verified = result.imap_verified = result.delivery_verified = True
                result.ok, result.status, result.retry_safe = True, "executed", False
                result.message = "Local mailbox created; authenticated SMTP self-delivery was read back through IMAP. The .test address cannot receive public service verification mail."
                return result
        except Exception as exc:
            result.status = "outcome_unknown" if mutation_started else "denied"
            result.retry_safe = not mutation_started
            # Do not serialize exception text from HTTP/process/SMTP libraries or stores.
            result.error = (
                "Mailbox provisioning may have changed state but did not finish verified. Do not retry creation; inspect mailbox status and recover through an operator."
                if mutation_started else
                str(exc) if isinstance(exc, (ValueError, MailboxProviderError)) else
                "Mailbox provisioning prerequisites are unavailable."
            )
            return result
