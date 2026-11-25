"""The janus-control container: the request Janus sends Proxmox, its first boot and its unit."""
import pytest

from learning.control_guest import (CONTROL_PROFILE, CONTROL_TAGS, ControlGuestError, bootstrap_script,
                                    container_arguments, control_address, service_unit)


KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMdV8iaJ0lYvGbM3hJmW0sQyJ7dU6m8rM2r6QJ0Z1aBc janus-control"
TEMPLATE = "local:vztmpl/ubuntu-24.04-standard_24.04-2_amd64.tar.zst"


def test_the_container_matches_what_the_owner_approved():
    arguments = container_arguments(template=TEMPLATE, public_key=KEY, vnet="janusnat", cidr="10.77.0.0/24")
    assert arguments["hostname"] == "janus-control"
    assert (arguments["cores"], arguments["memory_mib"], arguments["swap_mib"]) == (2, 2048, 1024)
    assert (arguments["storage"], arguments["disk_gib"]) == ("local-lvm", 16)
    assert arguments["pool"] == "janus"
    assert arguments["nesting"] is True
    assert arguments["net0"] == "name=eth0,bridge=janusnat,ip=10.77.0.11/24,gw=10.77.0.1,firewall=1"
    assert arguments["nameserver"] == "1.1.1.1 9.9.9.9"
    assert arguments["console_mode"] == "shell"
    assert arguments["ssh_public_keys"] == KEY
    assert "tags" not in arguments and CONTROL_TAGS == "janus-control"
    assert arguments["on_boot"] is True
    assert "password" not in arguments
    assert CONTROL_PROFILE["hostname"] == "janus-control"


def test_only_an_ubuntu_template_and_a_public_key_are_accepted():
    with pytest.raises(ControlGuestError):
        container_arguments(template="local:vztmpl/debian-13-standard_13.6-1_amd64.tar.zst",
                            public_key=KEY, vnet="janusnat", cidr="10.77.0.0/24")
    with pytest.raises(ControlGuestError):
        container_arguments(template=TEMPLATE, public_key="-----BEGIN OPENSSH " + "PRIVATE KEY-----",
                            vnet="janusnat", cidr="10.77.0.0/24")
    with pytest.raises(ControlGuestError):
        container_arguments(template=TEMPLATE, public_key=KEY, vnet="not a vnet", cidr="10.77.0.0/24")


@pytest.mark.parametrize("cidr", ["8.8.8.0/24", "10.77.0.0/16", "fd00::/64"])
def test_the_nat_subnet_is_one_private_ipv4_slash_24(cidr):
    with pytest.raises(ControlGuestError):
        control_address(cidr)


def test_first_boot_enrols_the_tailnet_and_prepares_the_service_account():
    script = bootstrap_script(auth_key_file="/run/janus/ts-key")
    assert "FLAGS=\"--tun=userspace-networking\"" in script
    assert script.index("userspace-networking") < script.index("systemctl restart tailscaled")
    assert "tailscale up --auth-key=file:/run/janus/ts-key --advertise-tags=tag:janus-control " \
           "--hostname=janus-control" in script
    assert "--ssh" not in script
    assert "rm -f /run/janus/ts-key" in script
    assert "useradd --system --create-home --home-dir /var/lib/janus" in script
    assert "install -d -o janus -g janus -m 750 /opt/janus/control-plane /var/lib/janus" in script
    assert "install -d -o root -g janus -m 750 /etc/janus" in script
    assert "tskey-" not in script


def test_the_unit_binds_loopback_and_restarts_from_one_database_and_token_file():
    unit = service_unit()
    assert "User=janus" in unit
    assert "ExecStart=/opt/janus/control-plane/.venv/bin/python -m learning.control_plane " \
           "--database /var/lib/janus/jobs.sqlite3 --tokens /etc/janus/worker-tokens.json " \
           "--host 127.0.0.1 --port 8765" in unit
    assert "Restart=on-failure" in unit
    assert "ProtectSystem=strict" in unit
    assert "ReadWritePaths=/var/lib/janus" in unit
    assert "WantedBy=multi-user.target" in unit


def test_invalid_unit_paths_and_ports_are_refused(tmp_path):
    for arguments in ({"database": "relative/jobs.sqlite3"},
                      {"tokens": "/etc/janus/../janus/worker-tokens.json"},
                      {"port": 0}, {"port": 70000}):
        with pytest.raises(ControlGuestError):
            service_unit(**arguments)
