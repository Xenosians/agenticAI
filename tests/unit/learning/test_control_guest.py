"""The janus-control container: the request Janus sends Proxmox, its first boot and its unit."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from learning.control_guest import (CONTROL_PROFILE, CONTROL_TAGS, ControlGuestError, bootstrap_script,
                                    build_control_package, container_arguments, control_address, package_digest,
                                    service_unit)


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


# What `import learning.control_plane.daemon` may load outside the standard library: the two pinned
# requirements and what they install themselves.
CONTROL_THIRD_PARTY = {"pydantic", "pydantic_core", "annotated_types", "typing_extensions", "typing_inspection",
                       "websockets"}
IMPORT_PROBE = (
    "import sys\n"
    "before = set(sys.modules)\n"
    "import learning.control_plane.daemon, learning\n"
    "print(learning.__file__)\n"
    "roots = {name.split('.')[0] for name in set(sys.modules) - before}\n"
    "print(','.join(sorted(name for name in sys.modules if name.split('.')[0] in ('learning', 'services', 'config'))))\n"
    # Interpreter-private modules (_sysconfigdata_*, _cffi_backend ...) are not packages anyone installs.
    "print(','.join(sorted(name for name in roots - set(sys.stdlib_module_names) - {'learning'}\n"
    "                      if not name.startswith('_'))))\n"
)


def _import_from(package: Path) -> list[str]:
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    imported = subprocess.run([sys.executable, "-c", IMPORT_PROBE], cwd=package, capture_output=True, text=True,
                              env=environment)
    assert imported.returncode == 0, imported.stderr
    return imported.stdout.strip().splitlines()


def test_the_control_package_is_the_control_plane_alone_with_its_requirements(tmp_path):
    package = build_control_package(tmp_path / "package")
    files = sorted(path.relative_to(package).as_posix() for path in package.rglob("*") if path.is_file())
    assert {"requirements-control.txt", "learning/__init__.py", "learning/control_plane/daemon.py",
            "learning/control_plane/__main__.py"} <= set(files)
    assert all(name.startswith("learning/control_plane/") or name in ("learning/__init__.py",
                                                                       "requirements-control.txt")
               for name in files)
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in files)
    # The service's learning/__init__.py imports the whole service (95 modules); the package's imports nothing.
    assert "import" not in (package / "learning" / "__init__.py").read_text(encoding="utf-8")
    learning_file, local, third_party = _import_from(package)
    assert Path(learning_file).resolve() == (package / "learning" / "__init__.py").resolve()
    assert all(name == "learning" or name.startswith("learning.control_plane") for name in local.split(","))
    assert set(filter(None, third_party.split(","))) <= CONTROL_THIRD_PARTY


def test_the_control_requirements_pin_exactly_what_the_control_plane_imports(tmp_path):
    package = build_control_package(tmp_path / "package")
    lines = [line.strip() for line in (package / "requirements-control.txt").read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")]
    pinned = dict(line.split("==", 1) for line in lines)
    assert set(pinned) == {"pydantic", "websockets"} and all(pinned.values())
    from importlib.metadata import version

    # The guest installs what the control plane's tests pass with here.
    assert pinned == {"pydantic": version("pydantic"), "websockets": version("websockets")}


def test_the_package_digest_follows_the_code_and_ignores_caches(tmp_path):
    package = build_control_package(tmp_path / "package")
    first = package_digest(package)
    assert first == package_digest(build_control_package(tmp_path / "again"))
    cache = package / "learning" / "control_plane" / "__pycache__"
    cache.mkdir()
    (cache / "daemon.cpython-312.pyc").write_bytes(b"compiled")
    assert package_digest(package) == first
    with (package / "learning" / "control_plane" / "daemon.py").open("a", encoding="utf-8") as handle:
        handle.write("\n# changed\n")
    assert package_digest(package) != first


def test_building_the_package_twice_leaves_no_stale_files(tmp_path):
    package = build_control_package(tmp_path / "package")
    (package / "learning" / "control_plane" / "removed_module.py").write_text("x = 1\n", encoding="utf-8")
    rebuilt = build_control_package(tmp_path / "package")
    assert not (rebuilt / "learning" / "control_plane" / "removed_module.py").exists()
