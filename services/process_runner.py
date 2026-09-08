import os
import shutil
import subprocess
import time

from pathlib import Path
from typing import Any


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

MAX_TIMEOUT_SECONDS = 30
MAX_OUTPUT_CHARS = 16_000


# ============================================================
# MVP executable policy
#
# Step 1 intentionally allows exactly ONE executable.
#
# This is expanded later after the execution boundary itself
# has been proven.
# ============================================================

ALLOWED_EXECUTABLES = {
    "pwd": {
        "risk": "read",
        "allow_arguments": False,
    },
}


def workspace_root() -> Path:
    """
    Return the directory inside which process execution is
    allowed.

    Defaults to this repository.

    Later this may be configured to something such as:
        /mnt/c/project
    """

    configured = os.getenv(
        "PROCESS_WORKSPACE_ROOT"
    )

    if configured:
        return (
            Path(configured)
            .expanduser()
            .resolve()
        )

    return PROJECT_ROOT


def resolve_cwd(
    cwd: str | None,
) -> Path:
    """
    Resolve a requested working directory and guarantee that it
    stays inside PROCESS_WORKSPACE_ROOT.
    """

    root = workspace_root()

    if not root.is_dir():
        raise ValueError(
            f"Workspace root does not exist: {root}"
        )

    if cwd is None or not cwd.strip():
        candidate = root

    else:
        requested = (
            Path(cwd)
            .expanduser()
        )

        if requested.is_absolute():
            candidate = requested.resolve()

        else:
            candidate = (
                root
                / requested
            ).resolve()

    if (
        candidate != root
        and root not in candidate.parents
    ):
        raise ValueError(
            "Working directory is outside the "
            "approved workspace."
        )

    if not candidate.is_dir():
        raise ValueError(
            f"Working directory does not exist: "
            f"{candidate}"
        )

    return candidate


def _bounded_output(
    value: str,
) -> str:
    if len(value) <= MAX_OUTPUT_CHARS:
        return value

    return (
        value[:MAX_OUTPUT_CHARS]
        + "\n...[output truncated]"
    )


def run_process(
    executable: str,
    args: list[str] | None = None,
    cwd: str | None = None,
    timeout_seconds: int = 10,
) -> dict[str, Any]:
    """
    Execute one policy-approved native process.

    Important:
    - no shell=True
    - no raw command strings
    - executable and arguments are separate
    - cwd is restricted to an approved workspace
    - execution has a bounded timeout
    - stdout/stderr are bounded
    """

    if not isinstance(
        executable,
        str,
    ):
        return {
            "ok": False,
            "status": "denied",
            "error": (
                "executable must be a string."
            ),
        }

    executable = executable.strip()

    policy = ALLOWED_EXECUTABLES.get(
        executable
    )

    if policy is None:
        return {
            "ok": False,
            "status": "denied",
            "error": (
                f"Executable '{executable}' "
                "is not allowed."
            ),
        }

    if args is None:
        args = []

    if not isinstance(
        args,
        list,
    ):
        return {
            "ok": False,
            "status": "denied",
            "error": (
                "args must be a list."
            ),
        }

    if not all(
        isinstance(argument, str)
        for argument in args
    ):
        return {
            "ok": False,
            "status": "denied",
            "error": (
                "Every process argument must "
                "be a string."
            ),
        }

    if (
        not policy["allow_arguments"]
        and args
    ):
        return {
            "ok": False,
            "status": "denied",
            "error": (
                f"Executable '{executable}' "
                "does not accept arguments "
                "under the current policy."
            ),
        }

    if (
        not isinstance(
            timeout_seconds,
            int,
        )
        or timeout_seconds < 1
        or timeout_seconds
        > MAX_TIMEOUT_SECONDS
    ):
        return {
            "ok": False,
            "status": "denied",
            "error": (
                "timeout_seconds must be "
                f"between 1 and "
                f"{MAX_TIMEOUT_SECONDS}."
            ),
        }

    try:
        resolved_cwd = resolve_cwd(
            cwd
        )

    except ValueError as exc:
        return {
            "ok": False,
            "status": "denied",
            "error": str(exc),
        }

    #
    # Resolve the approved executable before launch.
    #
    # The model will never control this resolved path.
    #

    executable_path = shutil.which(
        executable
    )

    if executable_path is None:
        return {
            "ok": False,
            "status": "error",
            "error": (
                f"Executable '{executable}' "
                "was not found on this machine."
            ),
        }

    command = [
        executable_path,
        *args,
    ]

    started_at = time.monotonic()

    try:
        completed = subprocess.run(
            command,
            cwd=str(
                resolved_cwd
            ),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            shell=False,
            check=False,
        )

    except subprocess.TimeoutExpired as exc:
        duration_ms = int(
            (
                time.monotonic()
                - started_at
            )
            * 1000
        )

        return {
            "ok": False,
            "status": "timeout",
            "executable": executable,
            "args": args,
            "cwd": str(
                resolved_cwd
            ),
            "exit_code": None,
            "stdout": _bounded_output(
                exc.stdout or ""
            ),
            "stderr": _bounded_output(
                exc.stderr or ""
            ),
            "timed_out": True,
            "duration_ms": duration_ms,
        }

    except OSError as exc:
        return {
            "ok": False,
            "status": "error",
            "error": (
                f"Process launch failed: "
                f"{exc}"
            ),
        }

    duration_ms = int(
        (
            time.monotonic()
            - started_at
        )
        * 1000
    )

    stdout = _bounded_output(
        completed.stdout
    )

    stderr = _bounded_output(
        completed.stderr
    )

    return {
        "ok":
            completed.returncode == 0,

        "status":
            (
                "success"
                if completed.returncode == 0
                else "error"
            ),

        "executable":
            executable,

        "args":
            args,

        "cwd":
            str(
                resolved_cwd
            ),

        "exit_code":
            completed.returncode,

        "stdout":
            stdout,

        "stderr":
            stderr,

        "timed_out":
            False,

        "duration_ms":
            duration_ms,
    }