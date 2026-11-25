"""Run the learning control plane as one long-lived process.

The process owns the durable job database and the authenticated socket. It does not own the GPU:
workers connect over the tailnet, take one lease and execute on their own machine. Configuration is
explicit so a deployment cannot silently widen the bind or read a world-readable token file.

    python -m learning.control_plane --database /var/lib/janus/jobs.sqlite3 \
        --tokens /etc/janus/worker-tokens.json --host 100.x.y.z --port 8765

``--host`` defaults to loopback. The deployed host must be the guest's Tailscale address; no code
here discovers or broadens the bind on its own.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import stat
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from learning.control_plane import canonical
from learning.control_plane.jobs import JobStore
from learning.control_plane.service import ControlPlaneService
from learning.control_plane.transport import ControlPlaneServer, WorkerTokens

DEFAULT_PORT = 8765
DEFAULT_LEASE_SECONDS = 300.0
DEFAULT_RETENTION_SECONDS = 7 * 24 * 3600.0
DEFAULT_PRUNE_SECONDS = 3600.0
LOGGER = logging.getLogger("janus.learning.control_plane")


@dataclass(frozen=True)
class Settings:
    """Everything the service needs, with every non-loopback choice explicit."""

    database: Path
    tokens: Path
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    lease_seconds: float = DEFAULT_LEASE_SECONDS
    retention_seconds: float = DEFAULT_RETENTION_SECONDS
    prune_seconds: float = DEFAULT_PRUNE_SECONDS

    def __post_init__(self) -> None:
        object.__setattr__(self, "database", Path(self.database))
        object.__setattr__(self, "tokens", Path(self.tokens))
        if not isinstance(self.host, str) or not self.host or self.host.strip() != self.host:
            raise canonical.ContractError("host must be one non-empty address")
        if type(self.port) is not int or not 0 <= self.port <= 65535:
            raise canonical.ContractError("port must be between 0 and 65535")
        for label in ("lease_seconds", "retention_seconds", "prune_seconds"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise canonical.ContractError(f"{label} must be positive")

    @classmethod
    def from_args(cls, argv: Sequence[str] | None = None) -> Settings:
        parser = argparse.ArgumentParser(
            prog="python -m learning.control_plane",
            description="Serve durable learning jobs to authenticated GPU workers.",
        )
        parser.add_argument("--database", type=Path, required=True,
                            help="path to jobs.sqlite3 on this control plane")
        parser.add_argument("--tokens", type=Path, required=True,
                            help="JSON object mapping worker ids to bearer tokens")
        parser.add_argument("--host", default="127.0.0.1",
                            help="bind address; a deployment passes the Tailscale address")
        parser.add_argument("--port", type=int, default=DEFAULT_PORT)
        parser.add_argument("--lease-seconds", type=float, default=DEFAULT_LEASE_SECONDS)
        parser.add_argument("--retention-seconds", type=float,
                            default=DEFAULT_RETENTION_SECONDS,
                            help="how long the message replay guard is kept")
        parser.add_argument("--prune-seconds", type=float, default=DEFAULT_PRUNE_SECONDS,
                            help="how often the replay guard is pruned")
        arguments = parser.parse_args(argv)
        return cls(
            database=arguments.database,
            tokens=arguments.tokens,
            host=arguments.host,
            port=arguments.port,
            lease_seconds=arguments.lease_seconds,
            retention_seconds=arguments.retention_seconds,
            prune_seconds=arguments.prune_seconds,
        )


def check_token_file(path: Path) -> None:
    """Refuse a token file that another local account can read on POSIX."""
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError as error:
        raise canonical.ContractError("the worker token file is not readable") from error
    if os.name == "posix" and mode & 0o077:
        raise canonical.ContractError(
            "the worker token file must be readable by its owner only (chmod 600)"
        )


def build(settings: Settings) -> tuple[ControlPlaneService, ControlPlaneServer]:
    """Open the store and build the socket service; no bind and no connection yet."""
    check_token_file(settings.tokens)
    tokens = WorkerTokens.from_file(settings.tokens)
    store = JobStore(settings.database)
    store.initialize()
    service = ControlPlaneService(store, lease_seconds=settings.lease_seconds)
    server = ControlPlaneServer(service, tokens, host=settings.host, port=settings.port)
    return service, server


async def serve(settings: Settings, stop: asyncio.Event | None = None, *,
                ready: Callable[[ControlPlaneServer], None] | None = None) -> None:
    """Run until ``stop`` is set; ``ready`` receives the listening server for tests/supervisors."""
    stop = stop or asyncio.Event()
    service, server = build(settings)
    await server.start()
    LOGGER.info(
        "learning control plane listening on %s:%s database=%s workers=%s",
        settings.host, server.port, settings.database, ",".join(server.tokens.workers()),
    )
    if ready is not None:
        ready(server)
    prune = asyncio.create_task(_prune(service.store, settings, stop), name="prune-messages")
    try:
        await stop.wait()
    finally:
        prune.cancel()
        await asyncio.gather(prune, return_exceptions=True)
        await server.stop()
        LOGGER.info("learning control plane stopped")


async def serve_until_signal(settings: Settings) -> None:
    """Run with SIGTERM/SIGINT as a clean stop when the platform supports it."""
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed: list[signal.Signals] = []
    for number in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(number, stop.set)
        except (NotImplementedError, RuntimeError):
            continue
        installed.append(number)
    try:
        await serve(settings, stop)
    finally:
        for number in installed:
            loop.remove_signal_handler(number)


async def _prune(store: JobStore, settings: Settings, stop: asyncio.Event) -> None:
    while not stop.is_set():
        store.prune_messages(older_than_seconds=settings.retention_seconds)
        try:
            await asyncio.wait_for(stop.wait(), timeout=settings.prune_seconds)
        except TimeoutError:
            continue


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        settings = Settings.from_args(argv)
        asyncio.run(serve_until_signal(settings))
    except canonical.ContractError as error:
        print(f"invalid configuration: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the module entry point
    raise SystemExit(main())
