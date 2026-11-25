"""The authenticated WebSocket transport for the learning control plane.

Only this module knows about ``websockets``. The service, store and protocol stay transport-free,
so the contract can be tested without a live tailnet.

The socket is a notification and trigger path. Authentication happens during the opening
handshake: the worker names itself in the path query and presents its own bearer token. A bad
worker, a bad token and a missing token are all refused with the same 401 so the handshake cannot
enumerate workers. A frame that violates the typed contract closes the connection with policy
code 1008; an oversized frame is closed by ``websockets`` with 1009. Neither close changes a job:
the store's lease fence and the service's dispatch are the only authority.
"""
from __future__ import annotations

import hmac
import json
import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import parse_qs, quote

from websockets.asyncio.client import ClientConnection, connect
from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Headers, Request, Response

from learning.control_plane import canonical
from learning.control_plane.protocol import (
    INBOUND_TYPES,
    MAX_MESSAGE_BYTES,
    MessageType,
    message_json,
    parse_outbound,
)
from learning.control_plane.service import ControlPlaneService

DEFAULT_PATH = "/v1/learning/worker"
POLICY_CLOSE = 1008
MAX_CLOSE_REASON_BYTES = 120
LOGGER = logging.getLogger("janus.learning.control_plane")


def _unauthorized() -> Response:
    return Response(
        401,
        "Unauthorized",
        Headers([
            ("Content-Type", "text/plain"),
            ("WWW-Authenticate", 'Bearer realm="janus-learning"'),
        ]),
    )


def _not_found() -> Response:
    return Response(404, "Not Found", Headers([("Content-Type", "text/plain")]))


def _reason(error: BaseException) -> str:
    raw = str(error).encode("utf-8")[:MAX_CLOSE_REASON_BYTES]
    return raw.decode("utf-8", errors="ignore")


class WorkerTokens:
    """The bearer tokens this control plane accepts, one per worker id.

    A token is checked in constant time against the token for the named worker. Unknown workers,
    non-ASCII values and short/empty tokens never authenticate. Token values are never included in
    an error message.
    """

    def __init__(self, tokens: Mapping[str, str]) -> None:
        if not isinstance(tokens, Mapping) or not tokens:
            raise canonical.ContractError("at least one worker token is required")
        self._tokens = {
            canonical.require(canonical.WORKER_ID, worker_id, "worker id"):
            canonical.require(canonical.WORKER_TOKEN, token, "worker token")
            for worker_id, token in tokens.items()
        }

    @classmethod
    def from_file(cls, path: str | Path) -> WorkerTokens:
        """Load a JSON object mapping worker ids to tokens."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise canonical.ContractError("a worker token file must be readable JSON") from error
        if not isinstance(data, dict):
            raise canonical.ContractError("a worker token file must map worker ids to tokens")
        return cls(data)

    def workers(self) -> tuple[str, ...]:
        return tuple(sorted(self._tokens))

    def verify(self, worker_id: str, token: str) -> bool:
        if not isinstance(token, str) or not token.isascii():
            return False
        expected = self._tokens.get(worker_id)
        if expected is None:
            return False
        return hmac.compare_digest(expected, token)


class ControlPlaneServer:
    """Serve one control-plane service to authenticated GPU workers.

    ``host`` defaults to loopback. A deployed service passes the Tailscale address explicitly; this
    class never widens the bind on its own.
    """

    def __init__(self, service: ControlPlaneService, tokens: WorkerTokens, *,
                 path: str = DEFAULT_PATH, host: str = "127.0.0.1", port: int = 0,
                 logger: logging.Logger | None = None) -> None:
        if not isinstance(service, ControlPlaneService):
            raise TypeError("service must be a ControlPlaneService")
        if not isinstance(tokens, WorkerTokens):
            raise TypeError("tokens must be WorkerTokens")
        if not path.startswith("/") or "?" in path or "#" in path:
            raise canonical.ContractError("the worker path must be an absolute path without a query")
        if type(port) is not int or not 0 <= port <= 65535:
            raise canonical.ContractError("port must be between 0 and 65535")
        self.service = service
        self.tokens = tokens
        self.path = path
        self.host = host
        self.requested_port = port
        self.logger = logger or LOGGER
        self._server: Server | None = None

    @property
    def port(self) -> int:
        server = self._server
        if server is None or not server.sockets:
            raise RuntimeError("the control-plane server is not listening")
        return int(server.sockets[0].getsockname()[1])

    async def start(self) -> ControlPlaneServer:
        if self._server is not None:
            raise RuntimeError("the control-plane server is already started")
        self._server = await serve(
            self._connection,
            self.host,
            self.requested_port,
            process_request=self._authenticate,
            max_size=MAX_MESSAGE_BYTES,
            max_queue=16,
            logger=self.logger,
        )
        return self

    async def stop(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.close()
            await server.wait_closed()

    async def _connection(self, connection: ServerConnection) -> None:
        request = connection.request
        worker_id = None if request is None else self._worker_for(request)
        if worker_id is None:
            await connection.close(POLICY_CLOSE, "unauthenticated")
            return
        try:
            session = await self.service.attach(worker_id, connection.send)
        except canonical.ContractError as error:
            await connection.close(POLICY_CLOSE, _reason(error))
            return
        try:
            async for frame in connection:
                try:
                    await self.service.handle(session, frame)
                except canonical.ContractError as error:
                    await connection.close(POLICY_CLOSE, _reason(error))
                    return
        except ConnectionClosed:
            pass
        finally:
            await self.service.detach(session)

    def _authenticate(self, connection: ServerConnection, request: Request) -> Response | None:
        worker_id = self._worker_for(request)
        if worker_id is None:
            if _path(request) != self.path:
                return _not_found()
            return _unauthorized()
        scheme, _, presented = request.headers.get("Authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not self.tokens.verify(worker_id, presented.strip()):
            return _unauthorized()
        return None

    def _worker_for(self, request: Request) -> str | None:
        if _path(request) != self.path:
            return None
        _, _, query = request.path.partition("?")
        values = parse_qs(query, keep_blank_values=True).get("worker", [])
        if len(values) != 1:
            return None
        try:
            return canonical.require(canonical.WORKER_ID, values[0], "worker id")
        except canonical.ContractError:
            return None


class WorkerConnection:
    """The worker side of the typed socket: send only worker messages, receive only control ones."""

    def __init__(self, connection: ClientConnection) -> None:
        self._connection = connection

    async def send(self, message: MessageType) -> None:
        if type(message) not in INBOUND_TYPES:
            raise canonical.ContractError(
                f"{message.type!r} is not a worker-to-control-plane message"
            )
        await self._connection.send(message_json(message))

    async def receive(self) -> MessageType:
        return parse_outbound(await self._connection.recv())

    async def close(self) -> None:
        await self._connection.close()


@asynccontextmanager
async def open_worker(uri: str, token: str) -> AsyncIterator[WorkerConnection]:
    """Connect one authenticated worker. The caller supplies the whole URI from ``worker_uri``."""
    if not isinstance(token, str) or not token.isascii():
        raise canonical.ContractError("a worker token must be ASCII text")
    async with connect(
        uri,
        additional_headers={"Authorization": f"Bearer {token}"},
        max_size=MAX_MESSAGE_BYTES,
        proxy=None,
    ) as connection:
        yield WorkerConnection(connection)


def worker_uri(host: str, port: int, worker_id: str, *, path: str = DEFAULT_PATH,
               secure: bool = False) -> str:
    """Build the one URI shape the server accepts; the worker id travels in the query."""
    canonical.require(canonical.WORKER_ID, worker_id, "worker id")
    if type(port) is not int or not 0 < port <= 65535:
        raise canonical.ContractError("port must be between 1 and 65535")
    if not isinstance(host, str) or not host:
        raise canonical.ContractError("host must be text")
    authority = f"[{host}]" if ":" in host else host
    scheme = "wss" if secure else "ws"
    return f"{scheme}://{authority}:{port}{path}?worker={quote(worker_id, safe='')}"


def _path(request: Request) -> str:
    return request.path.partition("?")[0]
