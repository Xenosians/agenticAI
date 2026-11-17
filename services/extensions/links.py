"""Each account's own Proxmox and Tailscale connection (owner decision, 17 November).

"because this is an agnostic program or user dependant not machine dependant
make everything like that besides the UI component": a link belongs to the
Janus account that saved it, not to the computer Janus runs on. It is stored
in the owner-scoped extension store with its secret encrypted, listed without
the secret, and used for that owner's chat, tools and dashboard. The machine
profile's link is only the default for an account that saved none.

A link is validated before it is stored: an HTTPS origin without credentials,
path or loopback host, an exact token identity, and bounded secrets. Reaching
the provider is not checked here; the first read says whether it answers.
"""
from __future__ import annotations

import ipaddress
import json
import re
from urllib.parse import urlsplit

KINDS = ("proxmox", "tailscale")
MAX_SECRET = 4096
TOKEN_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}@[A-Za-z0-9_.-]{1,64}![A-Za-z0-9_.-]{1,64}")
CLIENT_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}")
SERVER_NAME = re.compile(r"(?=.{1,253}$)[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
                         r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*")
# The fields each kind accepts; the secret field is stored encrypted, apart from the rest.
FIELDS = {"proxmox": {"url", "token_id", "token_secret", "verify_tls", "server_name"},
          "tailscale": {"client_id", "client_secret"}}
SECRET_FIELD = {"proxmox": "token_secret", "tailscale": "client_secret"}


class LinkError(ValueError):
    """A link was refused: a stable code and a fixed message that never repeats a secret."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def public_detail(self) -> dict:
        return {"code": self.code, "message": self.message}


def _https_origin(value) -> str:
    if not isinstance(value, str) or len(value) > 300:
        raise LinkError("link_url_invalid", "Enter the HTTPS address of the server, without a path.")
    parts = urlsplit(value.strip())
    try:
        port = parts.port
    except ValueError:
        port = -1
    host = (parts.hostname or "").lower()
    if (parts.scheme != "https" or not host or parts.username or parts.password or port == -1
            or parts.path not in ("", "/") or parts.query or parts.fragment):
        raise LinkError("link_url_invalid", "Enter the HTTPS address of the server, without a path.")
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if loopback or host == "localhost" or host.endswith(".localhost"):
        raise LinkError("link_url_invalid", "Enter the server's own address, not a loopback alias.")
    return value.strip().rstrip("/")


def _validated(kind: str, link: dict, stored: dict | None) -> tuple[dict, str]:
    """The link's public fields and its secret (the stored one when the new one is empty)."""
    if kind not in KINDS:
        raise LinkError("link_kind_invalid", "Choose Proxmox or Tailscale.")
    if not isinstance(link, dict) or not set(link) <= FIELDS[kind]:
        raise LinkError("link_field_invalid", "The link has a field Janus does not accept.")
    secret = link.get(SECRET_FIELD[kind], "")
    if not isinstance(secret, str) or len(secret) > MAX_SECRET:
        raise LinkError("link_secret_invalid", "The secret is not valid text of at most 4096 characters.")
    if not secret:
        if stored is None:
            raise LinkError("link_secret_required", "Enter the secret; Janus has none stored for this link.")
        secret = stored[SECRET_FIELD[kind]]
    if kind == "proxmox":
        token_id = link.get("token_id")
        if not isinstance(token_id, str) or not TOKEN_ID.fullmatch(token_id):
            raise LinkError("link_token_id_invalid", "Enter the API token as user@realm!name.")
        verify_tls = link.get("verify_tls", True)
        if type(verify_tls) is not bool:
            raise LinkError("link_verify_tls_invalid", "verify_tls must be true or false.")
        server_name = link.get("server_name") or None
        if server_name is not None and (not isinstance(server_name, str) or not SERVER_NAME.fullmatch(server_name)):
            raise LinkError("link_server_name_invalid", "Enter the certificate's host name, or leave it empty.")
        return ({"url": _https_origin(link.get("url")), "token_id": token_id, "verify_tls": verify_tls,
                 "server_name": server_name}, secret)
    client_id = link.get("client_id")
    if not isinstance(client_id, str) or not CLIENT_ID.fullmatch(client_id):
        raise LinkError("link_client_id_invalid", "Enter the OAuth client ID.")
    return {"client_id": client_id}, secret


def _public(kind: str, data: dict) -> dict:
    if kind == "proxmox":
        return {"kind": kind, "host": urlsplit(data["url"]).hostname, "token_id": data["token_id"],
                "verify_tls": data["verify_tls"], "server_name": data["server_name"],
                "credential_configured": True}
    return {"kind": kind, "client_id": data["client_id"], "credential_configured": True}


class OwnerLinks:
    """Each owner's links, in the extension store's database (encrypted secret, absent row = none)."""

    def __init__(self, store):
        self.store = store
        with store.connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS owner_link (
                    owner TEXT NOT NULL, kind TEXT NOT NULL, data TEXT NOT NULL, secret BLOB NOT NULL,
                    PRIMARY KEY(owner, kind))
            """)

    def save(self, owner: str, kind, link) -> dict:
        if not isinstance(owner, str) or not owner or len(owner) > 200:
            raise LinkError("link_owner_invalid", "The account is not valid.")
        stored = self.get(owner, kind) if kind in KINDS else None
        data, secret = _validated(kind, link, stored)
        with self.store.connect() as db:
            db.execute("INSERT OR REPLACE INTO owner_link(owner, kind, data, secret) VALUES (?,?,?,?)",
                       (owner, kind, json.dumps(data, sort_keys=True), self.store.cipher.encrypt(secret.encode())))
        return _public(kind, data)

    def get(self, owner: str, kind: str) -> dict | None:
        """The owner's link with its decrypted secret, for the host only; None when there is none."""
        with self.store.connect() as db:
            row = db.execute("SELECT data, secret FROM owner_link WHERE owner=? AND kind=?",
                             (owner, kind)).fetchone()
        if row is None:
            return None
        return {**json.loads(row["data"]), SECRET_FIELD[kind]: self.store.cipher.decrypt(row["secret"]).decode()}

    def list(self, owner: str) -> list[dict]:
        with self.store.connect() as db:
            rows = db.execute("SELECT kind, data FROM owner_link WHERE owner=? ORDER BY kind", (owner,)).fetchall()
        return [_public(row["kind"], json.loads(row["data"])) for row in rows]

    def remove(self, owner: str, kind) -> dict:
        if kind not in KINDS:
            raise LinkError("link_kind_invalid", "Choose Proxmox or Tailscale.")
        with self.store.connect() as db:
            removed = db.execute("DELETE FROM owner_link WHERE owner=? AND kind=?", (owner, kind)).rowcount == 1
        return {"kind": kind, "removed": removed}


def owner_links() -> OwnerLinks:
    from services.extensions.service import extension_store

    return OwnerLinks(extension_store())
