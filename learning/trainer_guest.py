"""The ``janus-trainer`` Proxmox container (owner decision 18 November 2026, 23:30 WIB).

"a Debian container named janus-trainer with 4 cores, 3.5 GiB RAM, 4 GiB swap and a 48 GiB disk on
local-lvm (you can remove it any time)". It reaches the internet through a Proxmox SDN Simple zone with
source NAT (hummingbird's only uplink is Wi-Fi, which cannot be bridged) and is reached by Janus over the
tailnet as ``tag:janus-trainer``.

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
PUBLIC_KEY = re.compile(r"^(ssh-ed25519|ecdsa-sha2-nistp256|ssh-rsa) [A-Za-z0-9+/=]{16,}( [A-Za-z0-9@._-]{1,64})?$")
NAME = re.compile(r"^[a-z][a-z0-9]{1,7}$")          # SDN zone and VNet ids: letters/digits, at most 8


class TrainerGuestError(ValueError):
    pass


def container_request(*, vmid: int, template: str, public_key: str, vnet: str) -> dict[str, str]:
    if not PUBLIC_KEY.match(public_key or ""):
        raise TrainerGuestError("only an OpenSSH public key line is sent to Proxmox")
    if not NAME.match(vnet):
        raise TrainerGuestError("the VNet name is 2-8 lowercase letters and digits")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}:vztmpl/debian-1[2-9]-standard_[0-9A-Za-z_.+-]+\.tar\.(zst|xz|gz)",
                        template):
        raise TrainerGuestError("the trainer uses a Debian standard template from the node's storage")
    profile = TRAINER_PROFILE
    return {"vmid": str(vmid), "ostemplate": template, "hostname": profile["hostname"],
            "cores": str(profile["cores"]), "memory": str(profile["memory_mib"]), "swap": str(profile["swap_mib"]),
            "rootfs": f"{profile['storage']}:{profile['disk_gib']}", "unprivileged": "1", "features": "nesting=1",
            "net0": f"name=eth0,bridge={vnet},ip=dhcp,firewall=1", "ssh-public-keys": public_key,
            "tags": "janus-trainer", "onboot": "1",
            "description": "Janus training worker (owner decision 18 Nov 2026); removable any time"}


def nat_zone_requests(*, zone: str, vnet: str, cidr: str):
    if not NAME.match(zone) or not NAME.match(vnet):
        raise TrainerGuestError("zone and VNet names are 2-8 lowercase letters and digits")
    network = ipaddress.ip_network(cidr)
    if not network.is_private or network.prefixlen != 24:
        raise TrainerGuestError("the NAT subnet is one private /24")
    hosts = list(network.hosts())
    return (
        ("POST", "/cluster/sdn/zones", {"zone": zone, "type": "simple", "dhcp": "dnsmasq", "ipam": "pve"}),
        ("POST", "/cluster/sdn/vnets", {"vnet": vnet, "zone": zone}),
        ("POST", f"/cluster/sdn/vnets/{vnet}/subnets",
         {"subnet": str(network), "type": "subnet", "gateway": str(hosts[0]), "snat": "1",
          "dhcp-range": f"start-address={hosts[99]},end-address={hosts[199]}"}),
    )


def bootstrap_script(*, auth_key_file: str, python_requirements: str) -> str:
    """First boot inside the container: tailnet enrolment, the janus user, Python and the worker's libraries.

    The one-time auth key arrives as a file and is deleted after ``tailscale up``; nothing echoes it.
    """
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
        "systemctl enable --now tailscaled",
        f"tailscale up --auth-key=file:{auth_key_file} --advertise-tags={TRAINER_TAG} --hostname=janus-trainer",
        f"rm -f {key}",
        "id janus >/dev/null 2>&1 || useradd --create-home --shell /bin/bash janus",
        "install -d -o janus -g janus -m 700 /home/janus/.ssh /home/janus/jobs /home/janus/runs /home/janus/janus-worker",
        "install -o janus -g janus -m 600 /root/.ssh/authorized_keys /home/janus/.ssh/authorized_keys",
        "loginctl enable-linger janus",
        "su - janus -c 'python3 -m venv ~/janus-worker/.venv'",
        "su - janus -c '~/janus-worker/.venv/bin/pip install --quiet --upgrade pip'",
        "su - janus -c \"~/janus-worker/.venv/bin/pip install --quiet --index-url https://download.pytorch.org/whl/cpu "
        "torch\"",
        f"printf '%s\\n' {requirements} > /home/janus/janus-worker/requirements-cpu.txt",
        "chown janus:janus /home/janus/janus-worker/requirements-cpu.txt",
        "su - janus -c '~/janus-worker/.venv/bin/pip install --quiet -r ~/janus-worker/requirements-cpu.txt'",
        "systemctl disable --now ssh.socket 2>/dev/null || true",
        "systemctl enable --now ssh",
        "echo janus-trainer-ready",
        "",
    ])
