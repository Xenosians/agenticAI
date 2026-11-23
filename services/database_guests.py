"""Database guests from official OCI images (owner decisions 20 Nov 2026 item 5, 23 Nov 2026 item 2).

"database management - PostgreSQL/MariaDB guests from OCI images, health through a read-only login
stored per account, dumps and restores each approved by the owner"; 23 Nov: "create and start in one
approval, then read its health".

Proxmox VE 9 pulls an OCI image into a storage's container templates (``oci-registry-pull``, skopeo on
the node) and creates an application container from it; the image's entrypoint runs as PID 1. A
database guest here is:

* the official image of a known engine at a pinned tag (never ``latest``);
* unprivileged, in the lab pool, tagged ``janus-db`` and ``db-<engine>`` so the database inventory and
  the change guard recognise it;
* on the guest NAT subnet 10.77.0.0/24, which the tailnet bridge routes, with a fixed address;
* with its data on its own mount point, so backups and restores carry the data.

Live findings (hummingbird, PVE 9.2 / pve-container 6.1.20) that shape the steps:

* 20 Nov: a create carrying ``tags`` was refused (HTTP 403, VM.Config.Options on ``/vms/<id>``). PVE's
  ``assert_tag_permissions`` checks ``/vms/<id>`` without the pool, and a guest that does not exist yet
  inherits nothing from the pool. So the guest is created untagged and tagged by a second request, once
  it is a pool member.
* 23 Nov: ``restore_oci_archive`` replaces ``env`` (and the entrypoint) with the image's own, so the
  first-boot settings go in that same second request, merged into the image's environment.

No password is ever part of a plan, an approval or a Proxmox request. The superuser password is
generated when the approved change runs and kept only in the account's encrypted store; the guest gets
its SCRAM-SHA-256 verifier (PostgreSQL stores a password that is already a verifier as-is) or its
mysql_native_password hash (the MariaDB image's ``MARIADB_ROOT_PASSWORD_HASH``).
"""
from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import hmac
import ipaddress
import os
import re

BRIDGED_SUBNET = ipaddress.ip_network("10.77.0.0/24")
TAG_PATTERN = re.compile(r"[0-9][0-9A-Za-z._-]{0,63}")
HOSTNAME = re.compile(r"[a-z][a-z0-9-]{0,61}[a-z0-9]")
NODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
STORAGE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}")
VNET = re.compile(r"[a-z][a-z0-9]{1,7}")
ENV_NAME = re.compile(r"\w+")
PASSWORD = re.compile(r"[A-Za-z0-9_-]{24,128}")
SCRAM_ITERATIONS = 4096


class DatabaseGuestRefused(ValueError):
    """The database guest cannot be planned as asked; nothing was sent."""


@dataclass(frozen=True)
class Engine:
    image: str
    port: int
    mount: str
    superuser: str
    # The image's variable that takes the superuser credential; only a verifier or hash goes there.
    credential_variable: str
    # Fixed first-boot settings (never secret).
    settings: tuple[tuple[str, str], ...] = ()


DATABASE_ENGINES = {
    # A fresh volume holds lost+found and initdb refuses a non-empty directory, so the cluster lives in a
    # subdirectory of the mount point.
    "postgresql": Engine("docker.io/library/postgres", 5432, "/var/lib/postgresql/data", "postgres",
                         "POSTGRES_PASSWORD", (("PGDATA", "/var/lib/postgresql/data/pgdata"),)),
    "mariadb": Engine("docker.io/library/mariadb", 3306, "/var/lib/mysql", "root", "MARIADB_ROOT_PASSWORD_HASH"),
}


def oci_template_volume(storage: str, reference: str) -> str:
    """The volume Proxmox lists after ``oci-registry-pull``: the image name and tag, '_' for ':'.

    Proxmox normalises the reference to a file name (PVE::Storage::normalize_content_filename) after
    dropping the registry and namespace; live on 20 Nov 2026 docker.io/library/postgres:17-alpine was
    listed as local:vztmpl/postgres_17-alpine.tar.
    """
    name = reference.rsplit("/", 1)[-1]
    return f"{storage}:vztmpl/{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}.tar"


def generate_password() -> str:
    """A random superuser password (URL-safe, 43 characters); generated only when an approved change runs."""
    return base64.urlsafe_b64encode(os.urandom(32)).decode().rstrip("=")


def scram_sha256_verifier(password: str, *, salt: bytes | None = None, iterations: int = SCRAM_ITERATIONS) -> str:
    """PostgreSQL's stored form of ``password`` (RFC 5802/7677); it cannot be used to log in by itself."""
    salt = os.urandom(16) if salt is None else salt
    salted = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    stored_key = hashlib.sha256(hmac.new(salted, b"Client Key", hashlib.sha256).digest()).digest()
    server_key = hmac.new(salted, b"Server Key", hashlib.sha256).digest()

    def encode(raw: bytes) -> str:
        return base64.b64encode(raw).decode()

    return f"SCRAM-SHA-256${iterations}:{encode(salt)}${encode(stored_key)}:{encode(server_key)}"


def mysql_native_hash(password: str) -> str:
    """MariaDB's mysql_native_password hash, as ``SELECT PASSWORD(...)`` prints it."""
    return "*" + hashlib.sha1(hashlib.sha1(password.encode()).digest()).hexdigest().upper()


def first_boot_environment(engine: str, password: str) -> dict[str, str]:
    """The variables the guest gets before its first start: settings plus the credential's verifier."""
    spec = DATABASE_ENGINES.get(engine)
    if spec is None:
        raise DatabaseGuestRefused(f"The database engine is one of {', '.join(sorted(DATABASE_ENGINES))}.")
    if not isinstance(password, str) or not PASSWORD.fullmatch(password):
        raise DatabaseGuestRefused("The generated password must be 24-128 URL-safe characters.")
    verifier = scram_sha256_verifier(password) if engine == "postgresql" else mysql_native_hash(password)
    return {**dict(spec.settings), spec.credential_variable: verifier}


def merged_environment(current: str | None, ours: dict[str, str]) -> str:
    """The image's NUL-separated environment with ours set: the image's entries first, ours last."""
    if any(not ENV_NAME.fullmatch(name) or any(char in value for char in "\0\n\r")
           for name, value in ours.items()):
        raise DatabaseGuestRefused("An environment entry is NAME=value on one line.")
    kept = [entry for entry in (current or "").split("\0") if entry and entry.split("=", 1)[0] not in ours]
    return "\0".join([*kept, *(f"{name}={value}" for name, value in ours.items())])


def environment_names(value: str | None) -> dict[str, str]:
    """A NUL-separated environment as a mapping (for read-back checks; never logged)."""
    return dict(entry.split("=", 1) for entry in (value or "").split("\0") if "=" in entry)


def database_guest_plan(*, engine: str, version: str, hostname: str, node: str, vmid: int, storage: str,
                        template_storage: str, pool: str, vnet: str, address: str, gateway: str, disk_gib: int,
                        memory_mib: int, cores: int, data_gib: int | None = None) -> dict:
    """The exact pull, create and configure requests for one database guest, validated; nothing is sent."""
    spec = DATABASE_ENGINES.get(engine)
    if spec is None:
        raise DatabaseGuestRefused(f"The database engine is one of {', '.join(sorted(DATABASE_ENGINES))}.")
    if not isinstance(version, str) or not TAG_PATTERN.fullmatch(version):
        raise DatabaseGuestRefused("Name a pinned image tag that starts with a digit, for example 17-alpine; "
                                   "a floating tag such as latest is refused.")
    if not isinstance(hostname, str) or not HOSTNAME.fullmatch(hostname):
        raise DatabaseGuestRefused("The hostname is lowercase letters, digits and hyphens.")
    for label, value, pattern in (("node", node, NODE), ("storage", storage, STORAGE),
                                  ("template storage", template_storage, STORAGE), ("pool", pool, STORAGE),
                                  ("VNet", vnet, VNET)):
        if not isinstance(value, str) or not pattern.fullmatch(value):
            raise DatabaseGuestRefused(f"Name one exact {label}.")
    if type(vmid) is not int or not 100 <= vmid <= 999_999_999:
        raise DatabaseGuestRefused("Name one free VMID.")
    for label, value, low, high in (("disk", disk_gib, 2, 1024), ("memory", memory_mib, 256, 262_144),
                                    ("cores", cores, 1, 64)):
        if type(value) is not int or not low <= value <= high:
            raise DatabaseGuestRefused(f"The {label} must be between {low} and {high}.")
    data = disk_gib if data_gib is None else data_gib
    if type(data) is not int or not 1 <= data <= 4096:
        raise DatabaseGuestRefused("The data volume must be between 1 and 4096 GiB.")
    try:
        interface = ipaddress.ip_interface(address)
        gateway_address = ipaddress.ip_address(gateway)
    except ValueError:
        raise DatabaseGuestRefused("Give the guest's address with its prefix and the gateway address.") from None
    if interface.network != BRIDGED_SUBNET or gateway_address not in BRIDGED_SUBNET or \
            interface.ip in (BRIDGED_SUBNET.network_address, BRIDGED_SUBNET.broadcast_address, gateway_address):
        raise DatabaseGuestRefused(f"A database guest gets a free host address in {BRIDGED_SUBNET}, the subnet "
                                   "the tailnet bridge routes, with that subnet's gateway.")
    reference = f"{spec.image}:{version}"
    template = oci_template_volume(template_storage, reference)
    # No tags (refused for a pool token before the guest exists) and no env (the image's replaces it):
    # both are set by the configure step after creation.
    create = {
        "vmid": str(vmid), "ostemplate": template, "hostname": hostname, "cores": str(cores),
        "memory": str(memory_mib), "swap": "0", "rootfs": f"{storage}:{disk_gib}", "unprivileged": "1",
        "pool": pool, "onboot": "1",
        "net0": f"name=eth0,bridge={vnet},ip={interface.with_prefixlen},gw={gateway_address},firewall=1",
        "mp0": f"{storage}:{data},mp={spec.mount},backup=1",
        "description": f"Janus database guest ({engine} {version}); managed through approvals.",
    }
    return {"engine": engine, "version": version, "reference": reference, "template": template,
            "template_storage": template_storage, "node": node, "vmid": vmid, "hostname": hostname,
            "address": str(interface.ip), "port": spec.port, "superuser": spec.superuser,
            "tags": f"janus-db,db-{engine}", "create": create}
