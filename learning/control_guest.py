"""The ``janus-control`` Proxmox container (owner decision 26 November 2026).

"a dedicated janus-control Ubuntu LXC, 2 cores, 2 GiB RAM, 16 GiB disk" on ``hummingbird``. It runs
the learning control plane — the durable job store, the memory and the authenticated WebSocket hub —
while GPU training and inference stay on the owner's machine. Like ``janus-trainer`` it reaches the
internet through the ``janusnat`` SDN zone and is reached over the tailnet as ``tag:janus-control``.

An unprivileged container has no ``/dev/net/tun``, so Tailscale runs in userspace. The daemon binds
loopback and ``tailscale serve`` publishes it on the guest's HTTPS tailnet name; the host kernel never
holds the tailnet address.

This module only builds requests and the first-boot script. The live steps run through the governed
Proxmox service with the owner's approval and are read back.
"""
from __future__ import annotations

import ipaddress
import re
import shlex

CONTROL_PROFILE = {"hostname": "janus-control", "cores": 2, "memory_mib": 2048, "swap_mib": 1024,
                   "disk_gib": 16, "storage": "local-lvm"}
CONTROL_TAG = "tag:janus-control"
CONTROL_TAGS = "janus-control"
CONTROL_POOL = "janus"
CONTROL_HOST = 11                                    # .11 of the NAT subnet; the gateway is .1
NAMESERVERS = "1.1.1.1 9.9.9.9"                      # not the host's: it may resolve through its own tailnet
SERVICE_ROOT = "/opt/janus/control-plane"
DATABASE = "/var/lib/janus/jobs.sqlite3"
TOKENS = "/etc/janus/worker-tokens.json"
SERVICE_PORT = 8765

PUBLIC_KEY = re.compile(r"^(ssh-ed25519|ecdsa-sha2-nistp256|ssh-rsa) [A-Za-z0-9+/=]{16,}( [A-Za-z0-9@._-]{1,64})?$")
NAME = re.compile(r"^[a-z][a-z0-9]{1,7}$")           # SDN zone and VNet ids: letters/digits, at most 8
UBUNTU_TEMPLATE = re.compile(
    r"^[A-Za-z][A-Za-z0-9_.-]{0,63}:vztmpl/ubuntu-2[0-9]\.[0-9]{2}-standard_[0-9A-Za-z_.+-]+\.tar\.(zst|xz|gz)$")
PATH = re.compile(r"/[A-Za-z0-9._/-]{1,200}")


class ControlGuestError(ValueError):
    pass


def _subnet(cidr: str) -> ipaddress.IPv4Network:
    network = ipaddress.ip_network(cidr)
    if network.version != 4 or not network.is_private or network.prefixlen != 24:
        raise ControlGuestError("the NAT subnet is one private IPv4 /24")
    return network


def control_address(cidr: str) -> tuple[str, str]:
    """The control plane's fixed address (with its prefix) and the gateway the NAT zone provides."""
    hosts = list(_subnet(cidr).hosts())
    return f"{hosts[CONTROL_HOST - 1]}/24", str(hosts[0])


def container_arguments(*, template: str, public_key: str, vnet: str, cidr: str) -> dict:
    """Keyword arguments of ``ProxmoxService.guest_create_lxc`` for the approved control container."""
    if not PUBLIC_KEY.match(public_key or ""):
        raise ControlGuestError("only an OpenSSH public key line is sent to Proxmox")
    if not NAME.match(vnet):
        raise ControlGuestError("the VNet name is 2-8 lowercase letters and digits")
    if not UBUNTU_TEMPLATE.match(template or ""):
        raise ControlGuestError("the control plane uses an Ubuntu standard template from the node's storage")
    profile = CONTROL_PROFILE
    address, gateway = control_address(cidr)
    return {"hostname": profile["hostname"], "template": template, "cores": profile["cores"],
            "memory_mib": profile["memory_mib"], "swap_mib": profile["swap_mib"], "disk_gib": profile["disk_gib"],
            "storage": profile["storage"], "pool": CONTROL_POOL,
            "description": "Janus learning control plane (owner decision 26 Nov 2026); removable any time",
            "net0": f"name=eth0,bridge={vnet},ip={address},gw={gateway},firewall=1", "nameserver": NAMESERVERS,
            "ssh_public_keys": public_key, "nesting": True,
            # A root shell on the console until first boot is done; guest_set_options(console_mode="tty") ends it.
            "console_mode": "shell", "on_boot": True}


def bootstrap_script(*, auth_key_file: str) -> str:
    """First boot inside the container: tailnet enrolment, Python, the janus service account and directories.

    The one-time auth key arrives as a file and is deleted after ``tailscale up``; nothing echoes it.
    """
    if not PATH.fullmatch(auth_key_file or "") or ".." in auth_key_file:
        raise ControlGuestError("the auth key file is one absolute path")
    key = shlex.quote(auth_key_file)
    return "\n".join([
        "#!/bin/sh",
        "set -eu",
        "export DEBIAN_FRONTEND=noninteractive",
        "apt-get update -q",
        "apt-get install -y -q --no-install-recommends ca-certificates curl gnupg python3 python3-venv "
        "rsync openssh-server",
        "systemctl enable --now ssh",
        "curl -fsSL https://tailscale.com/install.sh | sh",
        # No /dev/net/tun in an unprivileged container: userspace networking, which still accepts incoming
        # tailnet connections (sshd) through tailscaled's own network stack.
        "if grep -q '^FLAGS=' /etc/default/tailscaled; then "
        "sed -i 's/^FLAGS=.*/FLAGS=\"--tun=userspace-networking\"/' /etc/default/tailscaled; "
        "else echo 'FLAGS=\"--tun=userspace-networking\"' >> /etc/default/tailscaled; fi",
        "systemctl enable tailscaled",
        "systemctl restart tailscaled",
        f"tailscale up --auth-key=file:{auth_key_file} --advertise-tags={CONTROL_TAG} --hostname=janus-control",
        f"rm -f {key}",
        "id janus >/dev/null 2>&1 || useradd --system --create-home --home-dir /var/lib/janus "
        "--shell /usr/sbin/nologin janus",
        f"install -d -o janus -g janus -m 750 {shlex.quote(SERVICE_ROOT)} /var/lib/janus",
        "install -d -o root -g janus -m 750 /etc/janus",
    ]) + "\n"


def service_unit(*, service_root: str = SERVICE_ROOT, database: str = DATABASE, tokens: str = TOKENS,
                 port: int = SERVICE_PORT) -> str:
    """The systemd unit: loopback bind, one explicit database and token file, restart on failure."""
    for value in (service_root, database, tokens):
        if not PATH.fullmatch(value) or ".." in value:
            raise ControlGuestError("the unit's paths are absolute")
    if type(port) is not int or not 0 < port <= 65535:
        raise ControlGuestError("the service port is between 1 and 65535")
    return "\n".join([
        "[Unit]",
        "Description=Janus learning control plane",
        "After=network-online.target tailscaled.service",
        "Wants=network-online.target",
        "",
        "[Service]",
        "Type=simple",
        "User=janus",
        "Group=janus",
        f"WorkingDirectory={service_root}",
        "Environment=PYTHONUNBUFFERED=1",
        f"ExecStart={service_root}/.venv/bin/python -m learning.control_plane "
        f"--database {database} --tokens {tokens} --host 127.0.0.1 --port {port}",
        "Restart=on-failure",
        "RestartSec=2",
        "StateDirectory=janus",
        "NoNewPrivileges=yes",
        "PrivateTmp=yes",
        "ProtectHome=yes",
        "ProtectSystem=strict",
        "ReadWritePaths=/var/lib/janus",
        "RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
    ]) + "\n"


__all__ = ["CONTROL_PROFILE", "CONTROL_TAG", "CONTROL_TAGS", "CONTROL_POOL", "CONTROL_HOST",
           "ControlGuestError", "bootstrap_script", "container_arguments", "control_address", "service_unit"]
