"""The janus-trainer container: the request Janus sends Proxmox, and the first-boot script inside it."""
import pytest

from learning.trainer_guest import (TRAINER_PROFILE, bootstrap_script, container_request, nat_zone_requests,
                                    TrainerGuestError)


KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMdV8iaJ0lYvGbM3hJmW0sQyJ7dU6m8rM2r6QJ0Z1aBc janus-trainer"


def test_the_container_matches_what_the_owner_approved():
    request = container_request(vmid=991, template="local:vztmpl/debian-13-standard_13.6-1_amd64.tar.zst",
                                public_key=KEY, vnet="janusnat")
    assert request["hostname"] == "janus-trainer"
    assert request["cores"] == "4" and request["memory"] == "3584" and request["swap"] == "4096"
    assert request["rootfs"] == "local-lvm:48"
    assert request["unprivileged"] == "1"
    assert request["features"] == "nesting=1"                     # systemd inside an unprivileged CT
    assert request["net0"] == "name=eth0,bridge=janusnat,ip=dhcp,firewall=1"
    assert request["ssh-public-keys"] == KEY
    assert request["tags"] == "janus-trainer"
    assert request["onboot"] == "1"
    assert "password" not in request


def test_only_a_public_key_is_ever_sent():
    # Built from parts so the repository never holds a private-key header line.
    private_header = "-----BEGIN OPENSSH " + "PRIVATE KEY-----"
    with pytest.raises(TrainerGuestError):
        container_request(vmid=991, template="local:vztmpl/debian-13-standard_13.6-1_amd64.tar.zst",
                          public_key=private_header, vnet="janusnat")


def test_the_nat_zone_is_one_simple_zone_with_dhcp_and_snat():
    zone, vnet, subnet = nat_zone_requests(zone="janusnat", vnet="janusnat", cidr="10.77.0.0/24")
    assert zone == ("POST", "/cluster/sdn/zones", {"zone": "janusnat", "type": "simple", "dhcp": "dnsmasq",
                                                    "ipam": "pve"})
    assert vnet == ("POST", "/cluster/sdn/vnets", {"vnet": "janusnat", "zone": "janusnat"})
    assert subnet[2] == {"subnet": "10.77.0.0/24", "type": "subnet", "gateway": "10.77.0.1", "snat": "1",
                         "dhcp-range": "start-address=10.77.0.100,end-address=10.77.0.200"}


def test_first_boot_enrols_the_tailnet_installs_the_worker_and_never_logs_the_key():
    script = bootstrap_script(auth_key_file="/run/janus/ts-key", python_requirements="torch==2.13.0+cpu")
    assert "tailscale up --auth-key=file:/run/janus/ts-key --advertise-tags=tag:janus-trainer" in script
    assert "--ssh" not in script                                  # Janus uses its own key, not Tailscale SSH
    assert "rm -f /run/janus/ts-key" in script                    # the one-time key is removed after use
    assert "set -x" not in script
    assert "useradd --create-home --shell /bin/bash janus" in script
    assert "loginctl enable-linger janus" in script               # user units run without a login
    assert "--index-url https://download.pytorch.org/whl/cpu" in script   # CPU wheels: no CUDA download
    assert script.startswith("#!/bin/sh\nset -eu\n")


def test_the_profile_is_the_owner_decision():
    assert TRAINER_PROFILE == {"hostname": "janus-trainer", "cores": 4, "memory_mib": 3584, "swap_mib": 4096,
                               "disk_gib": 48, "storage": "local-lvm"}
