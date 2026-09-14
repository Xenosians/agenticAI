from __future__ import annotations

import json
import re

from pathlib import Path
from typing import Any

from services.process_runner import (
    resolve_cwd,
    run_trusted_process,
)


MAX_PROCESS_RECORDS = 100
LOG_TAIL_LINES = 100

PROCESS_TIMEOUT_SECONDS = 10
SERVICE_TIMEOUT_SECONDS = 15


SIMPLE_SERVICE_NAME_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"
)


SENSITIVE_ASSIGNMENT_PATTERN = re.compile(
    (
        r"(?i)"
        r"\b("
        r"password|passwd|pwd|secret|token|"
        r"api[_-]?key|authorization"
        r")"
        r"(\s*[:=]\s*)"
        r"((?:Bearer\s+)?[^\s,;]+)"
    )
)


BEARER_PATTERN = re.compile(
    r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"
)


URL_CREDENTIAL_PATTERN = re.compile(
    (
        r"(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)"
        r"(?P<user>[^/\s:@]+)"
        r":"
        r"(?P<password>[^/\s@]+)"
        r"@"
    )
)


def _resolve_runtime_directory(
    relative_path: str,
) -> tuple[
    Path | None,
    str | None,
]:
    if not isinstance(
        relative_path,
        str,
    ):
        return (
            None,
            "relative_path must be a string.",
        )

    relative_path = (
        relative_path.strip()
    )

    if not relative_path:
        relative_path = "."

    try:
        directory = (
            resolve_cwd(
                relative_path
            )
        )

    except ValueError as exc:
        return (
            None,
            str(
                exc
            ),
        )

    return (
        directory,
        None,
    )


def _process_error(
    result: dict[
        str,
        Any,
    ],
    *,
    provider: str,
) -> dict[str, Any]:
    error = (
        result.get(
            "error"
        )
    )

    if not isinstance(
        error,
        str,
    ):
        stderr = (
            result.get(
                "stderr"
            )
        )

        if (
            isinstance(
                stderr,
                str,
            )
            and stderr.strip()
        ):
            error = (
                stderr.strip()
            )

        else:
            error = (
                "Runtime inspection command failed."
            )

    return {
        "ok": False,

        "status":
            result.get(
                "status",
                "error",
            ),

        "provider":
            provider,

        "error":
            error,
    }


def workspace_process_snapshot(
) -> dict[str, Any]:
    """
    Return a bounded host process snapshot.

    Only non-sensitive process metadata is requested:

    - PID
    - parent PID
    - process state
    - elapsed runtime
    - executable name

    Full command lines and environment variables are never
    requested.
    """

    try:
        root = (
            resolve_cwd(
                "."
            )
        )

    except ValueError as exc:
        return {
            "ok": False,
            "status": "denied",
            "provider": "host_ps",
            "error": str(
                exc
            ),
        }

    result = (
        run_trusted_process(
            executable="ps",
            args=[
                "-eo",
                (
                    "pid=,ppid=,stat=,"
                    "etimes=,comm="
                ),
            ],
            cwd=str(
                root
            ),
            timeout_seconds=(
                PROCESS_TIMEOUT_SECONDS
            ),
        )
    )

    if not result.get(
        "ok",
        False,
    ):
        return (
            _process_error(
                result,
                provider="host_ps",
            )
        )

    stdout = (
        result.get(
            "stdout"
        )
    )

    if not isinstance(
        stdout,
        str,
    ):
        stdout = ""

    processes = []
    truncated = False

    for line in stdout.splitlines():
        stripped = (
            line.strip()
        )

        if not stripped:
            continue

        parts = (
            stripped.split(
                maxsplit=4
            )
        )

        if len(parts) != 5:
            continue

        (
            pid_text,
            parent_pid_text,
            state,
            elapsed_text,
            executable,
        ) = parts

        try:
            pid = int(
                pid_text
            )

            parent_pid = int(
                parent_pid_text
            )

            elapsed_seconds = int(
                elapsed_text
            )

        except ValueError:
            continue

        if (
            len(processes)
            >= MAX_PROCESS_RECORDS
        ):
            truncated = True
            break

        processes.append(
            {
                "pid":
                    pid,

                "parent_pid":
                    parent_pid,

                "state":
                    state,

                "elapsed_seconds":
                    elapsed_seconds,

                "executable":
                    executable,
            }
        )

    return {
        "ok": True,
        "status": "success",

        "provider":
            "host_ps",

        "processes":
            processes,

        "count":
            len(
                processes
            ),

        "truncated":
            truncated,

        "error":
            None,
    }


def _compose_services(
    project_dir: Path,
) -> tuple[
    list[str] | None,
    dict[str, Any] | None,
]:
    result = (
        run_trusted_process(
            executable="docker",
            args=[
                "compose",
                "config",
                "--services",
            ],
            cwd=str(
                project_dir
            ),
            timeout_seconds=(
                SERVICE_TIMEOUT_SECONDS
            ),
        )
    )

    if not result.get(
        "ok",
        False,
    ):
        return (
            None,
            _process_error(
                result,
                provider=(
                    "docker_compose"
                ),
            ),
        )

    stdout = (
        result.get(
            "stdout"
        )
    )

    if not isinstance(
        stdout,
        str,
    ):
        stdout = ""

    services = []

    for line in stdout.splitlines():
        service_name = (
            line.strip()
        )

        if not service_name:
            continue

        if (
            SIMPLE_SERVICE_NAME_PATTERN
            .fullmatch(
                service_name
            )
            is None
        ):
            continue

        services.append(
            service_name
        )

    return (
        services,
        None,
    )


def _parse_compose_ps(
    stdout: str,
) -> list[
    dict[str, Any]
]:
    """
    Support Docker Compose versions that emit either:

    - one JSON object per line
    - one JSON array
    - one JSON object
    """

    stripped = (
        stdout.strip()
    )

    if not stripped:
        return []

    decoded_items: list[
        dict[str, Any]
    ] = []

    try:
        decoded = json.loads(
            stripped
        )

        if isinstance(
            decoded,
            list,
        ):
            decoded_items.extend(
                item
                for item
                in decoded
                if isinstance(
                    item,
                    dict,
                )
            )

            return decoded_items

        if isinstance(
            decoded,
            dict,
        ):
            return [
                decoded
            ]

    except json.JSONDecodeError:
        pass

    for line in stripped.splitlines():
        try:
            decoded = json.loads(
                line
            )

        except json.JSONDecodeError:
            continue

        if isinstance(
            decoded,
            dict,
        ):
            decoded_items.append(
                decoded
            )

    return decoded_items


def workspace_service_status(
    relative_path: str = ".",
) -> dict[str, Any]:
    """
    Inspect Docker Compose services for one workspace project.

    The model does not control Docker arguments.
    """

    (
        project_dir,
        path_error,
    ) = _resolve_runtime_directory(
        relative_path
    )

    if project_dir is None:
        return {
            "ok": False,
            "status": "denied",
            "provider": (
                "docker_compose"
            ),
            "error":
                path_error,
        }

    (
        declared_services,
        service_error,
    ) = _compose_services(
        project_dir
    )

    if (
        declared_services
        is None
    ):
        return (
            service_error
            or {
                "ok": False,
                "status": "error",
                "provider": (
                    "docker_compose"
                ),
                "error": (
                    "Could not inspect "
                    "Compose services."
                ),
            }
        )

    result = (
        run_trusted_process(
            executable="docker",
            args=[
                "compose",
                "ps",
                "--all",
                "--format",
                "json",
            ],
            cwd=str(
                project_dir
            ),
            timeout_seconds=(
                SERVICE_TIMEOUT_SECONDS
            ),
        )
    )

    if not result.get(
        "ok",
        False,
    ):
        return (
            _process_error(
                result,
                provider=(
                    "docker_compose"
                ),
            )
        )

    stdout = (
        result.get(
            "stdout"
        )
    )

    if not isinstance(
        stdout,
        str,
    ):
        stdout = ""

    compose_records = (
        _parse_compose_ps(
            stdout
        )
    )

    services = []
    represented_services = set()

    for record in (
        compose_records
    ):
        service_name = (
            record.get(
                "Service"
            )
        )

        if not isinstance(
            service_name,
            str,
        ):
            continue

        if (
            service_name
            not in declared_services
        ):
            continue

        represented_services.add(
            service_name
        )

        services.append(
            {
                "service":
                    service_name,

                "container":
                    record.get(
                        "Name"
                    ),

                "state":
                    record.get(
                        "State"
                    ),

                "status":
                    record.get(
                        "Status"
                    ),

                "health":
                    record.get(
                        "Health"
                    ),
            }
        )

    for service_name in (
        declared_services
    ):
        if (
            service_name
            in represented_services
        ):
            continue

        services.append(
            {
                "service":
                    service_name,

                "container":
                    None,

                "state":
                    "not_created",

                "status":
                    None,

                "health":
                    None,
            }
        )

    services.sort(
        key=lambda item: (
            str(
                item.get(
                    "service",
                    ""
                )
            ),
            str(
                item.get(
                    "container",
                    ""
                )
            ),
        )
    )

    return {
        "ok": True,
        "status": "success",

        "provider":
            "docker_compose",

        "services":
            services,

        "service_count":
            len(
                services
            ),

        "error":
            None,
    }


def _redact_log_text(
    value: str,
) -> tuple[
    str,
    bool,
]:
    """
    Apply deterministic best-effort redaction for common secret
    forms before service log output leaves the trusted adapter.

    This is intentionally conservative but not claimed to be an
    exhaustive secret-detection system.
    """

    redacted = False

    def replace_assignment(
        match: re.Match,
    ) -> str:
        nonlocal redacted

        redacted = True

        return (
            f"{match.group(1)}"
            f"{match.group(2)}"
            "[REDACTED]"
        )

    value = (
        SENSITIVE_ASSIGNMENT_PATTERN
        .sub(
            replace_assignment,
            value,
        )
    )

    if BEARER_PATTERN.search(
        value
    ):
        redacted = True

        value = (
            BEARER_PATTERN.sub(
                "Bearer [REDACTED]",
                value,
            )
        )

    def replace_url_credentials(
        match: re.Match,
    ) -> str:
        nonlocal redacted

        redacted = True

        return (
            f"{match.group('scheme')}"
            f"{match.group('user')}"
            ":[REDACTED]@"
        )

    value = (
        URL_CREDENTIAL_PATTERN
        .sub(
            replace_url_credentials,
            value,
        )
    )

    return (
        value,
        redacted,
    )


def workspace_service_logs(
    service_name: str,
    relative_path: str = ".",
) -> dict[str, Any]:
    """
    Read a bounded tail of one declared Docker Compose service.

    Service names must exist in the trusted Compose
    configuration before logs are requested.

    Common credential/token patterns are redacted before the
    result leaves this adapter.
    """

    if not isinstance(
        service_name,
        str,
    ):
        return {
            "ok": False,
            "status": "denied",
            "provider": (
                "docker_compose"
            ),
            "error": (
                "service_name must "
                "be a string."
            ),
        }

    service_name = (
        service_name.strip()
    )

    if (
        not service_name
        or SIMPLE_SERVICE_NAME_PATTERN
        .fullmatch(
            service_name
        )
        is None
    ):
        return {
            "ok": False,
            "status": "denied",
            "provider": (
                "docker_compose"
            ),
            "error": (
                "service_name is not "
                "a valid service identifier."
            ),
        }

    (
        project_dir,
        path_error,
    ) = _resolve_runtime_directory(
        relative_path
    )

    if project_dir is None:
        return {
            "ok": False,
            "status": "denied",
            "provider": (
                "docker_compose"
            ),
            "error":
                path_error,
        }

    (
        declared_services,
        service_error,
    ) = _compose_services(
        project_dir
    )

    if declared_services is None:
        return (
            service_error
            or {
                "ok": False,
                "status": "error",
                "provider": (
                    "docker_compose"
                ),
                "error": (
                    "Could not inspect "
                    "Compose services."
                ),
            }
        )

    if (
        service_name
        not in declared_services
    ):
        return {
            "ok": False,
            "status": "denied",

            "provider":
                "docker_compose",

            "error": (
                f"Service '{service_name}' "
                "is not declared by the "
                "workspace Compose configuration."
            ),
        }

    result = (
        run_trusted_process(
            executable="docker",
            args=[
                "compose",
                "logs",
                "--no-color",
                "--tail",
                str(
                    LOG_TAIL_LINES
                ),
                service_name,
            ],
            cwd=str(
                project_dir
            ),
            timeout_seconds=(
                SERVICE_TIMEOUT_SECONDS
            ),
        )
    )

    if not result.get(
        "ok",
        False,
    ):
        return (
            _process_error(
                result,
                provider=(
                    "docker_compose"
                ),
            )
        )

    stdout = (
        result.get(
            "stdout"
        )
    )

    if not isinstance(
        stdout,
        str,
    ):
        stdout = ""

    (
        logs,
        redaction_applied,
    ) = _redact_log_text(
        stdout
    )

    return {
        "ok": True,
        "status": "success",

        "provider":
            "docker_compose",

        "service_name":
            service_name,

        "tail_lines":
            LOG_TAIL_LINES,

        "logs":
            logs,

        "redaction_applied":
            redaction_applied,

        "error":
            None,
    }