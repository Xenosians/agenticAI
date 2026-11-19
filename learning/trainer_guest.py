"""The ``janus-trainer`` Proxmox container (owner decision 18 November 2026, 23:30 WIB).

"a Debian container named janus-trainer with 4 cores, 3.5 GiB RAM, 4 GiB swap and a 48 GiB disk on
local-lvm (you can remove it any time)". It reaches the internet through a Proxmox SDN Simple zone with
source NAT (hummingbird's only uplink is Wi-Fi, which cannot be bridged) and is reached by Janus over the
tailnet as ``tag:janus-trainer``.

The container has a fixed address in the NAT subnet: Proxmox serves SDN DHCP only through dnsmasq
installed on every node, which its API cannot install. An unprivileged container has no /dev/net/tun,
so Tailscale runs in userspace and still accepts the tailnet connections Janus makes (SSH).

This module only builds requests and the first-boot script; the live steps run through the governed
Proxmox service with the owner's approval and are read back.
"""
from __future__ import annotations

import ipaddress
import re
import shlex

TRAINER_PROFILE = {"hostname": "janus-trainer", "cores": 4, "memory_mib": 3584, "swap_mib": 4096, "disk_gib": 48,
                   "storage": "local-lvm"}
TRAINER_TAG = "tag:janus-trainer"
# Janus-owned guests (VMIDs 991-999) live in the pool Janus may create guests in (owner decision 16 Nov 2026).
TRAINER_POOL = "janus"
TRAINER_TAGS = "janus-trainer"
PUBLIC_KEY = re.compile(r"^(ssh-ed25519|ecdsa-sha2-nistp256|ssh-rsa) [A-Za-z0-9+/=]{16,}( [A-Za-z0-9@._-]{1,64})?$")
NAME = re.compile(r"^[a-z][a-z0-9]{1,7}$")          # SDN zone and VNet ids: letters/digits, at most 8
TRAINER_HOST = 10                                   # the trainer is .10 of the NAT subnet; the gateway is .1
NAMESERVERS = "1.1.1.1 9.9.9.9"                     # not the host's: it may resolve through its own tailnet


class TrainerGuestError(ValueError):
    pass


def _subnet(cidr: str) -> ipaddress.IPv4Network:
    network = ipaddress.ip_network(cidr)
    if network.version != 4 or not network.is_private or network.prefixlen != 24:
        raise TrainerGuestError("the NAT subnet is one private IPv4 /24")
    return network


def trainer_address(cidr: str) -> tuple[str, str]:
    """The trainer's fixed address (with its prefix) and the gateway the NAT zone provides."""
    hosts = list(_subnet(cidr).hosts())
    return f"{hosts[TRAINER_HOST - 1]}/24", str(hosts[0])


def container_arguments(*, template: str, public_key: str, vnet: str, cidr: str) -> dict:
    """Keyword arguments of ``ProxmoxService.guest_create_lxc`` for the approved trainer container."""
    if not PUBLIC_KEY.match(public_key or ""):
        raise TrainerGuestError("only an OpenSSH public key line is sent to Proxmox")
    if not NAME.match(vnet):
        raise TrainerGuestError("the VNet name is 2-8 lowercase letters and digits")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}:vztmpl/debian-1[2-9]-standard_[0-9A-Za-z_.+-]+\.tar\.(zst|xz|gz)",
                        template):
        raise TrainerGuestError("the trainer uses a Debian standard template from the node's storage")
    profile = TRAINER_PROFILE
    address, gateway = trainer_address(cidr)
    return {"hostname": profile["hostname"], "template": template, "cores": profile["cores"],
            "memory_mib": profile["memory_mib"], "swap_mib": profile["swap_mib"], "disk_gib": profile["disk_gib"],
            "storage": profile["storage"], "pool": TRAINER_POOL,
            "description": "Janus training worker (owner decision 18 Nov 2026); removable any time",
            "net0": f"name=eth0,bridge={vnet},ip={address},gw={gateway},firewall=1", "nameserver": NAMESERVERS,
            "ssh_public_keys": public_key, "nesting": True,
            # A root shell on the console until first boot is done; guest_set_options(console_mode="tty") ends it.
            # No tags here: Proxmox checks tag rights on /vms/<id> alone, which only covers the guest once it is
            # a member of the pool, so TRAINER_TAGS is set by guest_set_options right after creation.
            "console_mode": "shell", "on_boot": True}


def cpu_requirements(text: str) -> tuple[str, str]:
    """The worker's pinned torch as its CPU build, and the other requirements without the CUDA-only ones."""
    torch, rest = None, []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("bitsandbytes"):           # QLoRA needs CUDA; a CPU worker trains plain LoRA
            continue
        pinned = re.fullmatch(r"torch==([0-9][0-9A-Za-z.]*)", line)
        if pinned:
            torch = f"torch=={pinned.group(1)}+cpu"
            continue
        rest.append(line)
    if torch is None:
        raise TrainerGuestError("the worker's requirements pin torch")
    return torch, "\n".join(rest) + "\n"


def nat_zone_requests(*, zone: str, vnet: str, cidr: str):
    if not NAME.match(zone) or not NAME.match(vnet):
        raise TrainerGuestError("zone and VNet names are 2-8 lowercase letters and digits")
    network = _subnet(cidr)
    _address, gateway = trainer_address(cidr)
    return (
        ("POST", "/cluster/sdn/zones", {"zone": zone, "type": "simple", "ipam": "pve"}),
        ("POST", "/cluster/sdn/vnets", {"vnet": vnet, "zone": zone}),
        ("POST", f"/cluster/sdn/vnets/{vnet}/subnets",
         {"subnet": str(network), "type": "subnet", "gateway": gateway, "snat": "1"}),
        # SDN changes stay pending until applied to the nodes' network configuration.
        ("PUT", "/cluster/sdn", {}),
    )


def bootstrap_script(*, auth_key_file: str, torch_requirement: str, python_requirements: str) -> str:
    """First boot inside the container: tailnet enrolment, the janus user, Python and the worker's libraries.

    The one-time auth key arrives as a file and is deleted after ``tailscale up``; nothing echoes it.
    """
    if not re.fullmatch(r"torch==[0-9][0-9A-Za-z.]*\+cpu", torch_requirement):
        raise TrainerGuestError("the trainer installs one pinned CPU build of torch")
    key = shlex.quote(auth_key_file)
    requirements = shlex.quote(python_requirements)
    return "\n".join([
        "#!/bin/sh",
        "set -eu",
        "export DEBIAN_FRONTEND=noninteractive",
        "apt-get update -q",
        "apt-get install -y -q --no-install-recommends ca-certificates curl gnupg python3 python3-venv rsync "
        "openssh-server systemd-container",
        "curl -fsSL https://tailscale.com/install.sh | sh",
        # No /dev/net/tun in an unprivileged container: userspace networking, which still accepts incoming
        # tailnet connections (sshd) through tailscaled's own network stack.
        "if grep -q '^FLAGS=' /etc/default/tailscaled; then "
        "sed -i 's/^FLAGS=.*/FLAGS=\"--tun=userspace-networking\"/' /etc/default/tailscaled; "
        "else echo 'FLAGS=\"--tun=userspace-networking\"' >> /etc/default/tailscaled; fi",
        "systemctl enable tailscaled",
        "systemctl restart tailscaled",
        f"tailscale up --auth-key=file:{auth_key_file} --advertise-tags={TRAINER_TAG} --hostname=janus-trainer",
        f"rm -f {key}",
        "id janus >/dev/null 2>&1 || useradd --create-home --shell /bin/bash janus",
        "install -d -o janus -g janus -m 700 /home/janus/.ssh /home/janus/jobs /home/janus/runs /home/janus/janus-worker",
        "install -o janus -g janus -m 600 /root/.ssh/authorized_keys /home/janus/.ssh/authorized_keys",
        "loginctl enable-linger janus",
        "su - janus -c 'python3 -m venv ~/janus-worker/.venv'",
        "su - janus -c '~/janus-worker/.venv/bin/pip install --quiet --upgrade pip'",
        # The pinned CPU build of serving's torch first, so the PyPI install below finds it already satisfied.
        f"su - janus -c \"~/janus-worker/.venv/bin/pip install --quiet --index-url https://download.pytorch.org/whl/cpu "
        f"{shlex.quote(torch_requirement)}\"",
        f"printf '%s\\n' {requirements} > /home/janus/janus-worker/requirements-cpu.txt",
        "chown janus:janus /home/janus/janus-worker/requirements-cpu.txt",
        "su - janus -c '~/janus-worker/.venv/bin/pip install --quiet -r ~/janus-worker/requirements-cpu.txt'",
        "systemctl disable --now ssh.socket 2>/dev/null || true",
        "systemctl enable --now ssh",
        "echo janus-trainer-ready",
        "",
    ])
