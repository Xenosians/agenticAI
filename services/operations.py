"""Read-only platform operations: Docker containers and service health endpoints."""

from __future__ import annotations

from services.lua_adapters.http_clients import client as lua_http_client

import re
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from urllib.parse import quote

import httpx

from config import Settings
from services.log_redaction import redact_log_text


DEFAULT_LOG_LINES = 50
MAX_LOG_LINES = 200
# Bound on the log text returned to the model, measured after demultiplexing.
MAX_LOG_BYTES = 16 * 1024
# Bound on what is read from Docker for one request, so a huge line cannot stall a job.
MAX_LOG_READ_BYTES = 4 * 1024 * 1024
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
DOCKER_TIME_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:\d{2})")
# A wildcard bind address is reachable through loopback on the same host.
WILDCARD_HOSTS = {"0.0.0.0": "127.0.0.1", "::": "::1"}


class OperationsProviderError(RuntimeError):
    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _failure(status: str, error_code: str, error: str, provider: str = "docker") -> dict:
    return {"ok": False, "status": status, "provider": provider, "error_code": error_code, "error": error}


def _observed(provider: str, data: dict) -> dict:
    return {"ok": True, "status": "success", "provider": provider, "observed_at": _now().isoformat(), "data": data}


def health_targets(settings: Settings) -> dict[str, str]:
    if settings.operations_health_urls:
        return dict(settings.operations_health_urls)
    targets: dict[str, str] = {}
    if settings.phoenix_base_url:
        targets["phoenix"] = settings.phoenix_base_url.rstrip("/") + "/api/health"
    if settings.ai_host:
        host = WILDCARD_HOSTS.get(settings.ai_host, settings.ai_host)
        targets["ai"] = f"http://{f'[{host}]' if ':' in host else host}:{settings.ai_port}/health"
    return targets


class OperationsService:
    """Fixed reads of allow-listed platform containers and configured health endpoints."""

    def __init__(self, settings: Settings, *, docker_transport: httpx.BaseTransport | None = None,
                 health_transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings
        self._docker_transport = docker_transport
        self._health_transport = health_transport

    def _docker(self) -> httpx.Client:
        transport = self._docker_transport or httpx.HTTPTransport(uds=str(self.settings.operations_docker_socket))
        return lua_http_client(transport=transport, base_url="http://docker",
                            timeout=self.settings.operations_timeout_seconds, trust_env=False)

    def _check_container(self, container: str | None) -> dict | None:
        allowed = self.settings.operations_containers
        if not allowed:
            return _failure("error", "not_configured", "OPERATIONS_CONTAINERS is not configured.")
        if container is not None and container not in allowed:
            return _failure("denied", "container_not_allowed",
                            f"Container {container!r} is not on the operations allowlist.")
        return None

    @staticmethod
    def _inspect(client: httpx.Client, name: str) -> dict | None:
        response = client.get(f"/containers/{quote(name, safe='')}/json")
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise OperationsProviderError("provider_error", f"Docker Engine refused the read (HTTP {response.status_code}).")
        body = response.json()
        if not isinstance(body, dict):
            raise OperationsProviderError("provider_error", "Docker Engine returned an invalid container record.")
        return body

    def _run(self, read):
        try:
            return read()
        except OperationsProviderError as exc:
            return _failure("error", exc.error_code, str(exc))
        except httpx.TimeoutException:
            return _failure("error", "provider_timeout", "Docker Engine did not answer in time.")
        except (httpx.HTTPError, OSError, ValueError):
            return _failure("error", "provider_unavailable", "Docker Engine is not reachable on the configured socket.")

    def container_status(self, container: str | None = None) -> dict:
        denied = self._check_container(container)
        if denied:
            return denied
        names = [container] if container is not None else list(self.settings.operations_containers)

        def read() -> dict:
            now = _now()
            with self._docker() as client:
                records = [_container_record(name, self._inspect(client, name), now) for name in names]
            return _observed("docker", {
                "containers": records,
                "count": len(records),
                "running": sum(1 for item in records if item["state"] == "running"),
                "unhealthy": sum(1 for item in records if item["health"] == "unhealthy"),
                "not_found": sum(1 for item in records if not item["found"]),
            })

        return self._run(read)

    def container_logs(self, container: str, lines: int | None = None) -> dict:
        denied = self._check_container(container)
        if denied:
            return denied
        lines = DEFAULT_LOG_LINES if lines is None else lines
        if isinstance(lines, bool) or not isinstance(lines, int) or not 1 <= lines <= MAX_LOG_LINES:
            return _failure("denied", "invalid_arguments", f"lines must be an integer from 1 to {MAX_LOG_LINES}.")

        def read() -> dict:
            path = f"/containers/{quote(container, safe='')}"
            with self._docker() as client:
                info = self._inspect(client, container)
                if info is None:
                    raise OperationsProviderError("not_found", f"Container {container!r} does not exist on this Docker host.")
                tty = isinstance(info.get("Config"), dict) and info["Config"].get("Tty") is True
                params = {"stdout": "1", "stderr": "1", "tail": str(lines)}
                with client.stream("GET", path + "/logs", params=params) as response:
                    if response.status_code != 200:
                        raise OperationsProviderError("provider_error",
                                                      f"Docker Engine refused the log read (HTTP {response.status_code}).")
                    raw, clipped = _log_payload(response.iter_bytes(), tty, MAX_LOG_BYTES)
            text = ANSI_ESCAPE_PATTERN.sub("", raw.decode("utf-8", errors="replace"))
            if clipped:
                # The byte bound cut the oldest line; a partial line could hide the key of a secret.
                text = text.split("\n", 1)[1] if "\n" in text else ""
            text, redacted = redact_log_text(text)
            # Redaction markers can lengthen lines, so the byte bound is applied again afterwards.
            kept, dropped = _newest_lines(text.splitlines(), MAX_LOG_BYTES)
            return _observed("docker", {"container": container, "requested_lines": lines, "line_count": len(kept),
                                        "logs": "\n".join(kept), "truncated": clipped or dropped,
                                        "redaction_applied": redacted})

        return self._run(read)

    def service_health(self) -> dict:
        targets = health_targets(self.settings)
        if not targets:
            return _failure("error", "not_configured",
                            "No health endpoints are configured (OPERATIONS_HEALTH_URLS or PHOENIX_BASE_URL).", "http")
        # trust_env=False: no proxy or .netrc credentials are ever attached to a probe.
        with lua_http_client(transport=self._health_transport, timeout=self.settings.operations_timeout_seconds,
                          follow_redirects=False, trust_env=False) as client:
            services = [_probe(client, name, url) for name, url in targets.items()]
        up = sum(1 for item in services if item["up"])
        return _observed("http", {"services": services, "count": len(services), "up": up, "down": len(services) - up})


def _docker_time(value: object) -> datetime | None:
    match = DOCKER_TIME_PATTERN.fullmatch(value) if isinstance(value, str) else None
    # Docker reports the zero time for a container that never started.
    if match is None or value.startswith("0001-"):
        return None
    fraction = (match.group(2) or "")[:6].ljust(6, "0")
    zone = "+00:00" if match.group(3) == "Z" else match.group(3)
    return datetime.fromisoformat(f"{match.group(1)}.{fraction}{zone}")


def _container_record(name: str, data: dict | None, now: datetime) -> dict:
    if data is None:
        return {"name": name, "found": False, "state": "not_found", "health": None, "started_at": None,
                "uptime_seconds": None, "exit_code": None, "restart_count": None, "image": None}
    state = data.get("State") if isinstance(data.get("State"), dict) else {}
    health = state.get("Health") if isinstance(state.get("Health"), dict) else {}
    config = data.get("Config") if isinstance(data.get("Config"), dict) else {}
    running = state.get("Running") is True
    started = _docker_time(state.get("StartedAt"))
    exit_code = state.get("ExitCode")
    return {
        "name": name,
        "found": True,
        "state": state.get("Status") if isinstance(state.get("Status"), str) else "unknown",
        "health": health.get("Status") if isinstance(health.get("Status"), str) else None,
        "started_at": started.isoformat() if started else None,
        "uptime_seconds": max(0, int((now - started).total_seconds())) if running and started else None,
        "exit_code": exit_code if not running and isinstance(exit_code, int) else None,
        "restart_count": data.get("RestartCount") if isinstance(data.get("RestartCount"), int) else None,
        "image": config.get("Image") if isinstance(config.get("Image"), str) else None,
    }


def _log_payload(chunks: Iterable[bytes], tty: bool, limit: int) -> tuple[bytes, bool]:
    """The newest `limit` bytes of a Docker log stream and whether older bytes were dropped.

    Without a TTY, Docker multiplexes stdout and stderr in frames with an 8-byte
    header (stream, 0, 0, 0, big-endian payload size); with a TTY the stream is raw.
    """
    kept = bytearray()
    pending = b""
    remaining = 0
    read = 0
    clipped = False
    for chunk in chunks:
        read += len(chunk)
        if tty:
            kept += chunk
        else:
            pending += chunk
            while pending:
                if remaining == 0:
                    if len(pending) < 8:
                        break
                    remaining = int.from_bytes(pending[4:8], "big")
                    pending = pending[8:]
                    continue
                payload = pending[:remaining]
                kept += payload
                remaining -= len(payload)
                pending = pending[len(payload):]
        if len(kept) > limit:
            del kept[:len(kept) - limit]
            clipped = True
        if read > MAX_LOG_READ_BYTES:
            clipped = True
            break
    return bytes(kept), clipped


def _newest_lines(lines: list[str], limit: int) -> tuple[list[str], bool]:
    total = -1
    for index in range(len(lines) - 1, -1, -1):
        total += len(lines[index].encode()) + 1
        if total > limit:
            return lines[index + 1:], True
    return lines, False


def _probe(client: httpx.Client, name: str, url: str) -> dict:
    started = time.perf_counter()
    result = {"name": name, "url": url, "up": False, "http_status": None, "latency_ms": None, "error": None}
    try:
        # Only the status line is needed; the body is never read.
        with client.stream("GET", url, headers={"accept": "application/json"}) as response:
            result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
            result["http_status"] = response.status_code
    except httpx.TimeoutException:
        result["error"] = "timeout"
        return result
    except httpx.HTTPError:
        result["error"] = "connection_failed"
        return result
    result["up"] = 200 <= result["http_status"] < 300
    if not result["up"]:
        result["error"] = "unexpected_status"
    return result
