"""Database credentials stored per account (owner decisions 18 Nov item 2, 23 Nov item 2).

A database guest's superuser password is generated when the owner's approved create runs and is kept
only here: encrypted with the extension store's key, scoped to the account that approved it, listed
without the secret. Nothing else (plans, approvals, Proxmox requests, receipts, logs) ever holds it.
"""
from __future__ import annotations

import re

NAME = re.compile(r"[a-z][a-z0-9-]{0,61}[a-z0-9]")
NODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
ENGINES = ("postgresql", "mariadb")


class DatabaseCredentialError(ValueError):
    """A credential could not be stored or found; the message never repeats a secret."""


class DatabaseCredentials:
    """Each owner's database logins, in the extension store's database (encrypted password)."""

    def __init__(self, store):
        self.store = store
        with store.connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS database_credential (
                    owner TEXT NOT NULL, node TEXT NOT NULL, vmid INTEGER NOT NULL, name TEXT NOT NULL,
                    engine TEXT NOT NULL, address TEXT NOT NULL, port INTEGER NOT NULL, username TEXT NOT NULL,
                    secret BLOB NOT NULL, PRIMARY KEY(owner, node, vmid))
            """)

    def save(self, owner: str, *, node: str, vmid: int, name: str, engine: str, address: str, port: int,
             username: str, password: str) -> dict:
        if not isinstance(owner, str) or not owner or len(owner) > 200:
            raise DatabaseCredentialError("The account is not valid.")
        if not isinstance(node, str) or not NODE.fullmatch(node) or type(vmid) is not int or vmid < 1 \
                or not isinstance(name, str) or not NAME.fullmatch(name) or engine not in ENGINES \
                or type(port) is not int or not 1 <= port <= 65535 or not isinstance(username, str) \
                or not username or not isinstance(password, str) or not 24 <= len(password) <= 128:
            raise DatabaseCredentialError("The database login is not valid.")
        with self.store.connect() as db:
            db.execute("INSERT OR REPLACE INTO database_credential VALUES (?,?,?,?,?,?,?,?,?)",
                       (owner, node, vmid, name, engine, address, port, username,
                        self.store.cipher.encrypt(password.encode())))
        return self._public((node, vmid, name, engine, address, port, username))

    def get(self, owner: str, node: str, vmid: int) -> dict | None:
        """The owner's login for one guest with its decrypted password (host use only), or None."""
        with self.store.connect() as db:
            row = db.execute("SELECT node, vmid, name, engine, address, port, username, secret "
                             "FROM database_credential WHERE owner=? AND node=? AND vmid=?",
                             (owner, node, vmid)).fetchone()
        if row is None:
            return None
        return {**self._public(tuple(row)[:7]), "password": self.store.cipher.decrypt(row["secret"]).decode()}

    def list(self, owner: str) -> list[dict]:
        with self.store.connect() as db:
            rows = db.execute("SELECT node, vmid, name, engine, address, port, username FROM database_credential "
                              "WHERE owner=? ORDER BY node, vmid", (owner,)).fetchall()
        return [self._public(tuple(row)) for row in rows]

    def remove(self, owner: str, node: str, vmid: int) -> bool:
        with self.store.connect() as db:
            return db.execute("DELETE FROM database_credential WHERE owner=? AND node=? AND vmid=?",
                              (owner, node, vmid)).rowcount == 1

    @staticmethod
    def _public(row: tuple) -> dict:
        node, vmid, name, engine, address, port, username = row
        return {"node": node, "vmid": vmid, "name": name, "engine": engine, "address": address, "port": port,
                "username": username, "credential_configured": True}


def database_credentials() -> DatabaseCredentials:
    from services.extensions.service import extension_store

    return DatabaseCredentials(extension_store())
