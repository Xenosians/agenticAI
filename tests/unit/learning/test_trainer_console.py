"""The janus-trainer console: typing the first boot through Proxmox's terminal (owner decision 19 Nov 2026).

A real bash in a pseudo-terminal stands in for the container's root shell, so echo, ``read`` and
``stty`` behave as they do in the container. Nothing here reaches Proxmox or Tailscale.
"""
import hashlib
import os
import pty
import select
import shutil
import signal
import time
from pathlib import Path

import pytest

from learning.trainer_console import (Console, ConsoleError, ProxmoxConsoleChannel, parse_termproxy,
                                      read_host_key, type_first_boot)

# The shape of a Tailscale auth key, not a real one; built from parts so no scanner mistakes it for a key.
KEY = "tskey-" + "auth-" + "kFixture1CNTRL-" + "Q" * 30
HOST_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMdV8iaJ0lYvGbM3hJmW0sQyJ7dU6m8rM2r6QJ0Z1aBc root@janus-trainer"


class PtyChannel:
    """The container's console as Janus sees it: whatever the terminal shows, in order."""

    def __init__(self, fd):
        self.fd, self.seen, self.sent = fd, "", []

    def send(self, text):
        self.sent.append(text)
        os.write(self.fd, text.encode())

    def recv(self, timeout):
        ready, _, _ = select.select([self.fd], [], [], timeout)
        if not ready:
            return None
        try:
            data = os.read(self.fd, 65536)
        except OSError:
            return None
        text = data.decode("utf-8", "replace")
        self.seen += text
        return text

    def drain(self, seconds=0.5):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.recv(0.05)


@pytest.fixture
def shell(tmp_path):
    if shutil.which("bash") is None or shutil.which("base64") is None or shutil.which("sha256sum") is None:
        pytest.skip("needs bash, base64 and sha256sum")
    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    # systemd-run stands in for the container's systemd: it records what it was asked to start.
    record = tmp_path / "systemd-run.args"
    (bin_directory / "systemd-run").write_text(f"#!/bin/sh\nprintf '%s\\n' \"$@\" > {record}\n", encoding="utf-8")
    (bin_directory / "systemd-run").chmod(0o755)
    (tmp_path / "ssh_host_ed25519_key.pub").write_text(HOST_KEY + "\n", encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()
    environment = {"PATH": f"{bin_directory}:/usr/bin:/bin", "HOME": str(home), "TERM": "xterm",
                   "PS1": "root@janus-trainer:~# ", "HISTFILE": str(home / ".bash_history"), "LANG": "C.UTF-8"}
    pid, fd = pty.fork()
    if pid == 0:  # the child becomes the console's shell
        os.execvpe("bash", ["bash", "--norc", "--noprofile", "-i"], environment)
    channel = PtyChannel(fd)
    yield channel, tmp_path, record, home
    try:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
    except OSError:
        pass
    os.close(fd)


def test_the_key_is_typed_with_echo_off_and_never_shown(shell):
    channel, root, _record, _home = shell
    console = Console(channel)
    target = root / "run" / "ts-key"
    console.run(f"mkdir -p {target.parent}", "D")
    console.enter_secret(KEY, str(target))
    channel.drain()
    assert target.read_text() == KEY
    assert oct(target.stat().st_mode & 0o777) == "0o600"
    # The console never showed the key, and it was never part of a typed command line.
    assert KEY not in channel.seen
    assert not any(KEY in line for line in channel.sent if line != KEY + "\r")


def test_first_boot_writes_the_key_and_script_starts_it_and_reads_the_host_key(shell):
    channel, root, record, home = shell
    script = "#!/bin/sh\nset -eu\necho 'it''s the first boot'\n" + "\n".join(f"echo line {n}" for n in range(200))
    directory = root / "run" / "janus"
    console = Console(channel)
    type_first_boot(console, auth_key=KEY, script=script, directory=str(directory))
    host_key = read_host_key(console, path=str(root / "ssh_host_ed25519_key.pub"), wait_seconds=5)
    channel.drain()
    assert host_key == HOST_KEY.rsplit(" ", 1)[0]
    assert (directory / "ts-key").read_text() == KEY
    # The script arrived byte for byte (it is checked against its SHA-256 before it may start).
    assert hashlib.sha256((directory / "bootstrap.sh").read_bytes()).hexdigest() == \
        hashlib.sha256(script.encode()).hexdigest()
    assert not (directory / "bootstrap.sh.b64").exists()
    # systemd starts it as its own unit, which keeps its result after the console session ends.
    assert record.read_text().split("\n")[:4] == ["--unit=janus-bootstrap", "--remain-after-exit", "/bin/sh",
                                                  str(directory / "bootstrap.sh")]
    assert KEY not in channel.seen
    # Nothing typed is kept in the shell's history.
    channel.send("exit\r")
    channel.drain(1.0)
    assert not (home / ".bash_history").exists() or KEY not in (home / ".bash_history").read_text()


def test_a_host_key_that_is_not_a_key_is_never_pinned(shell):
    channel, root, _record, _home = shell
    (root / "bad.pub").write_text("ssh-ed25519 short root@x\n", encoding="utf-8")
    with pytest.raises(ConsoleError):
        read_host_key(Console(channel), path=str(root / "bad.pub"), wait_seconds=2)


def test_a_step_that_fails_in_the_shell_stops_with_its_name(shell):
    channel, _root, _record, _home = shell
    console = Console(channel)
    with pytest.raises(ConsoleError, match="'W'"):
        console.run("false", "W", timeout=2.0)


def test_a_typed_marker_echo_never_counts_as_the_step_having_run(shell):
    channel, _root, _record, _home = shell
    console = Console(channel)
    # Typed but never run (no Enter): its echo shows the command, not the marker.
    channel.send(console.marker_command("Z"))
    with pytest.raises(ConsoleError):
        console.wait("Z", timeout=1.5)


@pytest.mark.parametrize("secret", ["", "tskey auth", "tskey-\x03-auth", "x" * 9, "tskey-auth-" + "\x15" * 20])
def test_only_a_plain_key_is_ever_typed(secret):
    class Never:
        def send(self, text):
            raise AssertionError("nothing may be typed")

        def recv(self, timeout):
            return None

    with pytest.raises(ValueError):
        Console(Never()).enter_secret(secret, "/run/janus/ts-key")


class FakeConnection:
    def __init__(self, answers):
        self.answers, self.sent, self.closed = list(answers), [], False

    def send(self, message):
        self.sent.append(message)

    def recv(self, timeout=None):
        if not self.answers:
            raise TimeoutError
        return self.answers.pop(0)

    def close(self):
        self.closed = True


def admitted(calls):
    return lambda *args, **kwargs: calls.append((args, kwargs))


def test_the_proxmox_console_uses_the_termproxy_ticket_and_the_xterm_framing():
    seen, admissions = {}, []

    def connect(url, **options):
        seen["url"], seen["options"] = url, options
        return FakeConnection([b"OK", b"root@janus-trainer:~# "])

    channel = ProxmoxConsoleChannel.open(
        origin="https://hummingbird.example.ts.net", node="hummingbird", vmid=991, port="5900",
        ticket="PVEVNC:6501ABCD::sig+/=", user="janus@pve!dashboard",
        authorization="PVEAPIToken=janus@pve!dashboard=fixture", ssl_context=None, connect=connect,
        admit=admitted(admissions))
    # The link crossed the host's Lua admission (destination and operation only, never the ticket).
    assert admissions == [(("websocket", "https://hummingbird.example.ts.net"),
                           {"operation": "container_console", "details": {"node": "hummingbird", "vmid": 991}})]
    assert seen["url"] == ("wss://hummingbird.example.ts.net/api2/json/nodes/hummingbird/lxc/991/vncwebsocket"
                           "?port=5900&vncticket=PVEVNC%3A6501ABCD%3A%3Asig%2B%2F%3D")
    assert seen["options"]["additional_headers"] == {"Authorization": "PVEAPIToken=janus@pve!dashboard=fixture"}
    # Proxmox closes a console socket that does not ask for the "binary" subprotocol, as the web UI does.
    assert seen["options"]["subprotocols"] == ["binary"]
    assert seen["options"]["origin"] == "https://hummingbird.example.ts.net"
    connection = channel.connection
    # The ticket proves the session; then a wide terminal, so long lines do not wrap.
    assert connection.sent[:2] == ["janus@pve!dashboard:PVEVNC:6501ABCD::sig+/=\n", "1:250:50:"]
    channel.send("ls é\r")
    assert connection.sent[-1] == "0:6:ls é\r"                     # the length counts UTF-8 bytes
    assert channel.recv(1.0) == "root@janus-trainer:~# "
    assert channel.recv(0.1) is None


def test_a_refused_console_ticket_stops_before_anything_is_typed():
    connection = FakeConnection([b"NO"])
    with pytest.raises(ConsoleError, match="refused"):
        ProxmoxConsoleChannel.open(origin="https://h.example.ts.net", node="h", vmid=991, port="5900",
                                   ticket="PVEVNC:x", user="janus@pve!dashboard", authorization="PVEAPIToken=x",
                                   ssl_context=None, connect=lambda url, **_: connection, admit=admitted([]))
    assert connection.closed and len(connection.sent) == 1


def test_a_link_the_admission_refuses_is_never_opened():
    def refuse(*args, **kwargs):
        raise PermissionError("refused by the adapter")

    def connect(url, **options):
        raise AssertionError("nothing may connect")

    with pytest.raises(PermissionError):
        ProxmoxConsoleChannel.open(origin="https://h.example.ts.net", node="h", vmid=991, port="5900",
                                   ticket="PVEVNC:x", user="u@pve", authorization="x", ssl_context=None,
                                   connect=connect, admit=refuse)


def test_a_termproxy_answer_is_read_only_in_its_known_shape():
    assert parse_termproxy({"port": 5900, "ticket": "PVEVNC:abc", "user": "janus@pve!dashboard", "upid": "x"}) == (
        "5900", "PVEVNC:abc", "janus@pve!dashboard")
    for answer in (None, {"port": "59x", "ticket": "PVEVNC:a", "user": "u"},
                   {"port": 5900, "ticket": "ticket", "user": "u"}, {"port": 5900, "ticket": "PVEVNC:a", "user": ""}):
        with pytest.raises(ConsoleError):
            parse_termproxy(answer)


def test_only_an_https_origin_becomes_a_console_link():
    with pytest.raises(ValueError):
        ProxmoxConsoleChannel.open(origin="http://h.example.ts.net", node="h", vmid=991, port="5900",
                                   ticket="PVEVNC:x", user="u@pve", authorization="x", ssl_context=None,
                                   connect=lambda url, **_: FakeConnection([b"OK"]), admit=admitted([]))
