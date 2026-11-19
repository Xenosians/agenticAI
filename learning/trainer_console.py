"""Typing the janus-trainer container's first boot through its Proxmox console.

Owner decision (19 November 2026): "Through the container's console over the Proxmox API: Janus creates
it with a root-shell console, types the script (the key is never echoed), then switches the console back
to a login prompt". Proxmox's API cannot copy a file into a container (a bind mount or a raw LXC key
needs root@pam), so the container is created with ``cmode=shell`` and Janus types into that root shell.

- Every step ends with a marker printed by the shell itself, so typed but unrun text never counts as
  done, and a failing step stops everything after it.
- The one-time Tailscale key is read with echo off (``stty -echo`` then ``read -r``): it is never part
  of a command line, the console never shows it back, and history is off for the whole session.
- The first-boot script arrives as base64 in short lines, is checked against its SHA-256 and only then
  started, as its own systemd unit, so its result outlives the console session.
- The container's SSH host key is read on the console and pinned before Janus first connects over SSH.
"""
from __future__ import annotations

import base64
import hashlib
import re
import shlex
import time
import uuid
from collections.abc import Callable
from typing import Any, Protocol
from urllib.parse import quote, urlsplit

AUTH_KEY = re.compile(r"tskey-[A-Za-z0-9-]{10,200}")
PATH = re.compile(r"/[A-Za-z0-9._/-]{1,200}")
BASE64_LINE = 76
DEFAULT_TIMEOUT = 30.0
# The console's terminal size: wide enough that typed lines never wrap.
COLUMNS, ROWS = 250, 50


class ConsoleError(RuntimeError):
    pass


class Channel(Protocol):
    def send(self, text: str) -> None: ...

    def recv(self, timeout: float) -> str | None: ...


class Console:
    """One shell on the other end of a channel, driven one checked step at a time."""

    def __init__(self, channel: Channel, *, clock: Callable[[], float] = time.monotonic):
        self.channel, self.clock = channel, clock
        self.output = ""
        self.token = uuid.uuid4().hex[:8]
        self.steps = 0

    def marker_command(self, name: str) -> str:
        # The marker is assembled by printf from parts: the typed command's echo never contains it.
        return f"printf '%s%s\\n' 'JANUS-{self.token}-' '{name}-OK'"

    def _marker(self, name: str) -> str:
        return f"JANUS-{self.token}-{name}-OK"

    def wait(self, name: str, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        marker = self._marker(name)
        deadline = self.clock() + timeout
        while marker not in self.output:
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise ConsoleError(f"the console did not finish step {name!r} in time")
            chunk = self.channel.recv(min(remaining, 1.0))
            if chunk:
                self.output += chunk
        # Keep only what follows the marker, so a later step cannot match an earlier one's output.
        self.output = self.output.split(marker, 1)[1]

    def run(self, command: str, name: str, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        """Type one command; its marker is printed only if it succeeded."""
        self.channel.send(f"{command} && {self.marker_command(name)}\r")
        self.wait(name, timeout=timeout)

    def step(self, command: str, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.steps += 1
        self.run(command, f"S{self.steps}", timeout=timeout)

    def enter_secret(self, secret: str, path: str) -> None:
        """Write ``secret`` to ``path`` (mode 0600) without it ever being echoed or on a command line."""
        if not AUTH_KEY.fullmatch(secret or ""):
            raise ValueError("only a Tailscale auth key is typed into the console")
        if not PATH.fullmatch(path) or ".." in path:
            raise ValueError("the key goes to one absolute path")
        quoted = shlex.quote(path)
        # Echo goes off and the shell says it is waiting before the key is sent: a key sent while the shell
        # still edits the previous line reaches the terminal in raw mode, where its Enter never ends a line.
        self.channel.send(f"stty -echo; {self.marker_command('RD')}; IFS= read -r JANUS_KEY; janus_read=$?; "
                          f"stty echo; [ \"$janus_read\" = 0 ] && {self.marker_command('R')}\r")
        self.wait("RD")
        self.channel.send(secret + "\r")
        self.wait("R")
        # The key leaves the shell's memory whether or not the write worked; only a written key counts.
        self.run(f"(umask 077; printf '%s' \"$JANUS_KEY\" > {quoted}); janus_write=$?; unset JANUS_KEY; "
                 f"[ \"$janus_write\" = 0 ]", "K")

    def write_file(self, path: str, content: str, *, mode: str = "700") -> None:
        if not PATH.fullmatch(path) or ".." in path or not re.fullmatch(r"[0-7]{3}", mode):
            raise ValueError("a file goes to one absolute path with an octal mode")
        quoted, staged = shlex.quote(path), shlex.quote(path + ".b64")
        digest = hashlib.sha256(content.encode()).hexdigest()
        encoded = base64.b64encode(content.encode()).decode()
        self.step(f": > {staged}")
        lines = [encoded[index:index + BASE64_LINE] for index in range(0, len(encoded), BASE64_LINE)]
        for start in range(0, len(lines), 8):
            # A few short lines per step: no line is long enough for the terminal to mangle.
            chunk = "\n".join(lines[start:start + 8])
            self.step(f"printf '%s\\n' {' '.join(shlex.quote(line) for line in chunk.splitlines())} >> {staged}")
        self.step(f"base64 -d {staged} > {quoted} && rm -f {staged} && "
                  f"[ \"$(sha256sum {quoted} | cut -d' ' -f1)\" = {digest} ] && chmod {mode} {quoted}")

    def read_line(self, command: str, *, prefix: str, timeout: float = DEFAULT_TIMEOUT) -> str:
        """The first output line of ``command`` that starts with ``prefix`` (for a public value)."""
        self.steps += 1
        name = f"S{self.steps}"
        self.channel.send(f"{command} && {self.marker_command(name)}\r")
        marker = self._marker(name)
        deadline = self.clock() + timeout
        while marker not in self.output:
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise ConsoleError(f"the console did not finish step {name!r} in time")
            chunk = self.channel.recv(min(remaining, 1.0))
            if chunk:
                self.output += chunk
        before, self.output = self.output.split(marker, 1)
        for line in before.replace("\r", "\n").split("\n"):
            line = line.strip()
            if line.startswith(prefix):
                return line
        raise ConsoleError(f"step {name!r} printed no line starting with {prefix!r}")


def type_first_boot(console: Console, *, auth_key: str, script: str, directory: str = "/run/janus") -> None:
    """Type the one-time key and the first-boot script, and start the script as its own systemd unit."""
    if not PATH.fullmatch(directory):
        raise ValueError("first boot uses an absolute directory")
    # Nothing typed in this session is written to the shell's history.
    console.run("set +o history; unset HISTFILE", "H")
    console.step(f"install -d -m 700 {shlex.quote(directory)}")
    console.enter_secret(auth_key, f"{directory}/ts-key")
    console.write_file(f"{directory}/bootstrap.sh", script)
    console.step(f"systemd-run --unit=janus-bootstrap --remain-after-exit /bin/sh "
                 f"{shlex.quote(directory + '/bootstrap.sh')}")


def read_host_key(console: Console, *, path: str = "/etc/ssh/ssh_host_ed25519_key.pub",
                  wait_seconds: int = 120) -> str:
    """The container's SSH host key, read on its console, to pin before the first SSH connection."""
    if not PATH.fullmatch(path) or ".." in path or type(wait_seconds) is not int or not 1 <= wait_seconds <= 3600:
        raise ValueError("the host key is one absolute path, waited for at most an hour")
    quoted = shlex.quote(path)
    # The key appears when sshd's package configures itself; wait for the file rather than racing it.
    line = console.read_line(f"for i in $(seq {wait_seconds}); do [ -s {quoted} ] && break; sleep 1; done; "
                             f"cat {quoted}", prefix="ssh-ed25519 ", timeout=wait_seconds + 30)
    parts = line.split()
    if len(parts) < 2 or not re.fullmatch(r"[A-Za-z0-9+/=]{40,}", parts[1]):
        raise ConsoleError("the container's SSH host key could not be read")
    return f"{parts[0]} {parts[1]}"


class ProxmoxConsoleChannel:
    """A container's terminal over Proxmox's ``termproxy`` and ``vncwebsocket`` (the xterm.js protocol).

    Frames to Proxmox: ``0:<UTF-8 byte length>:<data>`` for input, ``1:<cols>:<rows>:`` for a resize; the
    first frame is ``<user>:<ticket>\\n`` and Proxmox answers ``OK``. Everything from Proxmox is terminal
    output.
    """

    def __init__(self, connection: Any):
        self.connection = connection

    @classmethod
    def open(cls, *, origin: str, node: str, vmid: int, port: str, ticket: str, user: str, authorization: str,
             ssl_context: Any, connect: Callable[..., Any], server_hostname: str | None = None,
             timeout: float = DEFAULT_TIMEOUT, admit: Callable[..., Any] | None = None) -> "ProxmoxConsoleChannel":
        """Connect after the host's Lua admission approved the link; a refusal opens nothing."""
        parts = urlsplit(origin)
        if parts.scheme != "https" or not parts.hostname or parts.path not in ("", "/"):
            raise ValueError("the console link is the Proxmox HTTPS origin")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", node) or type(vmid) is not int \
                or not re.fullmatch(r"[0-9]{1,5}", str(port)):
            raise ValueError("the console needs one node, one VMID and the port Proxmox gave")
        if admit is None:
            from services.lua_adapters.host_boundaries import inspect_link as admit
        admit("websocket", f"https://{parts.netloc}", operation="container_console",
              details={"node": node, "vmid": vmid})
        url = (f"wss://{parts.netloc}/api2/json/nodes/{node}/lxc/{vmid}/vncwebsocket"
               f"?port={port}&vncticket={quote(ticket, safe='')}")
        # Proxmox closes a console socket that does not ask for the "binary" subprotocol (the web UI's xterm.js
        # does). No websocket pings or compression: the console proxy speaks plain frames.
        options: dict[str, Any] = {"additional_headers": {"Authorization": authorization}, "ssl": ssl_context,
                                   "subprotocols": ["binary"], "origin": f"https://{parts.netloc}",
                                   "compression": None, "open_timeout": timeout, "max_size": 2**20,
                                   "ping_interval": None}
        if server_hostname:
            options["server_hostname"] = server_hostname
        connection = connect(url, **options)
        connection.send(f"{user}:{ticket}\n")
        answer = connection.recv(timeout=timeout)
        if (answer.decode("utf-8", "replace") if isinstance(answer, bytes) else str(answer))[:2] != "OK":
            connection.close()
            raise ConsoleError("Proxmox refused the console ticket")
        connection.send(f"1:{COLUMNS}:{ROWS}:")
        return cls(connection)

    def send(self, text: str) -> None:
        self.connection.send(f"0:{len(text.encode('utf-8'))}:{text}")

    def recv(self, timeout: float) -> str | None:
        try:
            data = self.connection.recv(timeout=timeout)
        except TimeoutError:
            return None
        return data.decode("utf-8", "replace") if isinstance(data, bytes) else str(data)

    def close(self) -> None:
        self.connection.close()


def parse_termproxy(data: Any) -> tuple[str, str, str]:
    """The port, ticket and user of a ``termproxy`` answer."""
    if not isinstance(data, dict):
        raise ConsoleError("Proxmox returned no console ticket")
    port, ticket, user = data.get("port"), data.get("ticket"), data.get("user")
    if not re.fullmatch(r"[0-9]{1,5}", str(port)) or not isinstance(ticket, str) or not ticket.startswith("PVEVNC:") \
            or not isinstance(user, str) or not user:
        raise ConsoleError("Proxmox returned a console ticket in an unknown shape")
    return str(port), ticket, user


__all__ = ["Console", "ConsoleError", "ProxmoxConsoleChannel", "parse_termproxy", "read_host_key",
           "type_first_boot"]
