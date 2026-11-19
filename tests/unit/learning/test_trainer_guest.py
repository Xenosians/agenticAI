"""The janus-trainer container: the request Janus sends Proxmox, and the first-boot script inside it."""
import pytest

from learning.trainer_guest import (TRAINER_PROFILE, TRAINER_TAGS, bootstrap_script, container_arguments,
                                    cpu_requirements, nat_zone_requests, TrainerGuestError)


KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMdV8iaJ0lYvGbM3hJmW0sQyJ7dU6m8rM2r6QJ0Z1aBc janus-trainer"


def test_the_container_matches_what_the_owner_approved():
    # Arguments of the governed ProxmoxService.guest_create_lxc: one request, task followed, read back.
    arguments = container_arguments(template="local:vztmpl/debian-13-standard_13.6-1_amd64.tar.zst",
                                     public_key=KEY, vnet="janusnat", cidr="10.77.0.0/24")
    assert arguments["hostname"] == "janus-trainer"
    assert (arguments["cores"], arguments["memory_mib"], arguments["swap_mib"]) == (4, 3584, 4096)
    assert (arguments["storage"], arguments["disk_gib"]) == ("local-lvm", 48)
    assert arguments["pool"] == "janus"                            # where Janus may create its own guests
    assert arguments["nesting"] is True                            # systemd inside an unprivileged CT
    # A fixed address: SDN DHCP needs dnsmasq installed on every node, which the API cannot install.
    assert arguments["net0"] == "name=eth0,bridge=janusnat,ip=10.77.0.10/24,gw=10.77.0.1,firewall=1"
    assert arguments["nameserver"] == "1.1.1.1 9.9.9.9"
    # A root shell on the console for first boot only (guest_set_options(console_mode="tty") ends it).
    assert arguments["console_mode"] == "shell"
    assert arguments["ssh_public_keys"] == KEY
    # Tags follow creation: Proxmox checks tag rights on /vms/<id> alone, which the pool covers only afterwards.
    assert "tags" not in arguments and TRAINER_TAGS == "janus-trainer"
    assert arguments["on_boot"] is True
    assert "password" not in arguments


def test_only_a_public_key_is_ever_sent():
    # Built from parts so the repository never holds a private-key header line.
    private_header = "-----BEGIN OPENSSH " + "PRIVATE KEY-----"
    with pytest.raises(TrainerGuestError):
        container_arguments(template="local:vztmpl/debian-13-standard_13.6-1_amd64.tar.zst",
                            public_key=private_header, vnet="janusnat", cidr="10.77.0.0/24")


def test_the_nat_zone_is_one_simple_zone_with_snat_and_is_applied():
    zone, vnet, subnet, apply = nat_zone_requests(zone="janusnat", vnet="janusnat", cidr="10.77.0.0/24")
    assert zone == ("POST", "/cluster/sdn/zones", {"zone": "janusnat", "type": "simple", "ipam": "pve"})
    assert vnet == ("POST", "/cluster/sdn/vnets", {"vnet": "janusnat", "zone": "janusnat"})
    assert subnet[2] == {"subnet": "10.77.0.0/24", "type": "subnet", "gateway": "10.77.0.1", "snat": "1"}
    # Pending SDN changes take effect only when applied.
    assert apply == ("PUT", "/cluster/sdn", {})


@pytest.mark.parametrize("cidr", ["8.8.8.0/24", "10.77.0.0/16", "fd00::/64"])
def test_the_nat_subnet_is_one_private_ipv4_slash_24(cidr):
    with pytest.raises(TrainerGuestError):
        nat_zone_requests(zone="janusnat", vnet="janusnat", cidr=cidr)


REQUIREMENTS = """# The training worker
torch==2.13.0
transformers==5.16.1
peft==0.20.0
# QLoRA (CUDA workers only)
bitsandbytes==0.50.2; platform_system == "Linux" and platform_machine == "x86_64"
"""


def test_a_cpu_worker_pins_the_cpu_build_of_the_same_torch_and_skips_qlora():
    torch, rest = cpu_requirements(REQUIREMENTS)
    assert torch == "torch==2.13.0+cpu"                            # the version serving trains with, CPU build
    assert rest.splitlines() == ["transformers==5.16.1", "peft==0.20.0"]
    with pytest.raises(TrainerGuestError):
        cpu_requirements("transformers==5.16.1\n")                 # torch must be pinned


def test_first_boot_enrols_the_tailnet_installs_the_worker_and_never_logs_the_key():
    script = bootstrap_script(auth_key_file="/run/janus/ts-key", torch_requirement="torch==2.13.0+cpu",
                              python_requirements="transformers==5.16.1")
    # The pinned CPU build first, from the CPU index; then the rest from PyPI, torch already satisfied.
    assert "--index-url https://download.pytorch.org/whl/cpu torch==2.13.0+cpu" in script
    assert script.index("whl/cpu") < script.index("requirements-cpu.txt'")
    assert "tailscale up --auth-key=file:/run/janus/ts-key --advertise-tags=tag:janus-trainer" in script
    assert "--ssh" not in script                                  # Janus uses its own key, not Tailscale SSH
    # No /dev/net/tun in an unprivileged container: userspace networking, set before tailscaled restarts.
    assert script.index("--tun=userspace-networking") < script.index("systemctl restart tailscaled") \
        < script.index("tailscale up")
    assert "rm -f /run/janus/ts-key" in script                    # the one-time key is removed after use
    assert "set -x" not in script
    assert "useradd --create-home --shell /bin/bash janus" in script
    assert "loginctl enable-linger janus" in script               # user units run without a login
    assert script.startswith("#!/bin/sh\nset -eu\n")


def test_the_profile_is_the_owner_decision():
    assert TRAINER_PROFILE == {"hostname": "janus-trainer", "cores": 4, "memory_mib": 3584, "swap_mib": 4096,
                               "disk_gib": 48, "storage": "local-lvm"}
