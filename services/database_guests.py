"""Database guests from official OCI images (owner decision 20 Nov 2026, ~14:10 WIB, item 5).

"database management - PostgreSQL/MariaDB guests from OCI images, health through a read-only login
stored per account, dumps and restores each approved by the owner".

Proxmox VE 9 pulls an OCI image into a storage's container templates (``oci-registry-pull``, skopeo on
the node) and creates an application container from it; the image's entrypoint runs as PID 1 and its
ENV comes along. A database guest here is:

* the official image of a known engine at a pinned tag (never ``latest``);
* unprivileged, in the lab pool, tagged ``janus-db`` and ``db-<engine>`` so the database inventory and
  the change guard recognise it;
* on the guest NAT subnet 10.77.0.0/24, which the tailnet bridge routes (item 2), with a fixed address;
* with its data directory on its own mount point, so backups and restores carry the data.

No password is ever part of a plan or a Proxmox request: the images refuse to start without one only
when they initialise an empty data directory, so the plan passes the engines' documented
"generate a random root/superuser password" switches and Janus sets the real passwords afterwards
through the guest's console, writing them only to the account's encrypted store.
"""
from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import re

BRIDGED_SUBNET = ipaddress.ip_network("10.77.0.0/24")
TAG_PATTERN = re.compile(r"[0-9][0-9A-Za-z._-]{0,63}")
HOSTNAME = re.compile(r"[a-z][a-z0-9-]{0,61}[a-z0-9]")
NODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
STORAGE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}")
VNET = re.compile(r"[a-z][a-z0-9]{1,7}")


class DatabaseGuestRefused(ValueError):
    """The database guest cannot be planned as asked; nothing was sent."""


@dataclass(frozen=True)
class Engine:
    image: str
    port: int
    data_directory: str
    # Environment that lets the official image initialise without a password in the request.
    first_boot_env: tuple[tuple[str, str], ...]


DATABASE_ENGINES = {
    "postgresql": Engine("docker.io/library/postgres", 5432, "/var/lib/postgresql/data",
                         # The image's documented switch; Janus replaces it with a real password at once.
                         (("POSTGRES_HOST_AUTH_METHOD", "scram-sha-256"), ("POSTGRES_INITDB_ARGS", "--auth-local=trust"))),
    "mariadb": Engine("docker.io/library/mariadb", 3306, "/var/lib/mysql",
                      (("MARIADB_RANDOM_ROOT_PASSWORD", "1"),)),
}


def oci_template_volume(storage: str, reference: str) -> str:
    """The volume Proxmox lists after ``oci-registry-pull``: the image name and tag, '_' for ':'.

    Proxmox normalises the reference to a file name (PVE::Storage::normalize_content_filename) after
    dropping the registry and namespace; live on 20 Nov 2026 docker.io/library/postgres:17-alpine was
    listed as local:vztmpl/postgres_17-alpine.tar.
    """
    name = reference.rsplit("/", 1)[-1]
    return f"{storage}:vztmpl/{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}.tar"


def database_guest_plan(*, engine: str, version: str, hostname: str, node: str, vmid: int, storage: str,
                        template_storage: str, pool: str, vnet: str, address: str, gateway: str, disk_gib: int,
                        memory_mib: int, cores: int, data_gib: int | None = None) -> dict:
    """The exact pull and create requests for one database guest, validated; nothing is sent here."""
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
    create = {
        "vmid": str(vmid), "ostemplate": template, "hostname": hostname, "cores": str(cores),
        "memory": str(memory_mib), "swap": "0", "rootfs": f"{storage}:{disk_gib}", "unprivileged": "1",
        "pool": pool, "tags": f"janus-db,db-{engine}", "onboot": "1",
        "net0": f"name=eth0,bridge={vnet},ip={interface.with_prefixlen},gw={gateway_address},firewall=1",
        "mp0": f"{storage}:{data},mp={spec.data_directory},backup=1",
        "env": "\0".join(f"{key}={value}" for key, value in spec.first_boot_env),
        "description": f"Janus database guest ({engine} {version}); managed through approvals.",
    }
    return {"engine": engine, "version": version, "reference": reference, "template": template,
            "template_storage": template_storage, "node": node, "vmid": vmid, "hostname": hostname,
            "address": str(interface.ip), "port": spec.port, "create": create}
