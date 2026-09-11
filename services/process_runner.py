import re
import shutil
import subprocess
import time

from pathlib import Path
from typing import Any

from config import (
    get_settings,
)


MAX_TIMEOUT_SECONDS = 30
MAX_OUTPUT_CHARS = 16_000


# ============================================================
# Native executable policy
#
# Single source of truth for:
# - allowed executables
# - action risk
# - approval requirement
# - argument validation
#
# ToolGateway may inspect this policy before execution.
#
# run_process() evaluates it again immediately before launch,
# so approval never bypasses the actual execution policy.
# ============================================================

ALLOWED_EXECUTABLES = {
    "pwd": {
        "risk": "read",
        "requires_approval": False,
        "argument_policy": "none",
    },

    "ls": {
        "risk": "read",
        "requires_approval": False,
        "argument_policy": "none",
    },

    "git": {
        "risk": "read",
        "requires_approval": False,
        "argument_policy": "git_status",
    },

    "mkdir": {
        "risk": "low",
        "requires_approval": True,
        "argument_policy": "single_directory_name",
    },
}


SIMPLE_DIRECTORY_NAME_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
)


def workspace_root() -> Path:
    """
    Return the directory inside which process execution is
    allowed.

    PROCESS_WORKSPACE_ROOT is owned by the centralized Settings
    provider.

    The default remains this repository, preserving the
    existing MVP behavior.
    """

    settings = get_settings()

    return settings.resolve_project_path(
        settings.process_workspace_root
    )


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

    if (
        cwd is None
        or not cwd.strip()
    ):
        candidate = root

    else:
        requested = (
            Path(cwd)
            .expanduser()
        )

        if requested.is_absolute():
            candidate = (
                requested.resolve()
            )

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
            "Working directory does not exist: "
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


def _validate_process_arguments(
    executable: str,
    args: list[str],
    argument_policy: str,
) -> tuple[
    bool,
    str | None,
]:
    """
    Validate executable-specific arguments.

    MVP policies intentionally remain narrow.
    """

    if argument_policy == "none":
        if args:
            return (
                False,
                (
                    f"Executable '{executable}' "
                    "does not accept arguments "
                    "under the current policy."
                ),
            )

        return True, None

    if argument_policy == "git_status":
        expected_args = [
            "status",
            "--short",
            "--branch",
        ]

        if args != expected_args:
            return (
                False,
                (
                    "Git currently permits only "
                    "'git status --short --branch'."
                ),
            )

        return True, None

    if (
        argument_policy
        == "single_directory_name"
    ):
        if len(args) != 1:
            return (
                False,
                (
                    f"Executable '{executable}' "
                    "requires exactly one "
                    "directory-name argument."
                ),
            )

        directory_name = args[0]

        if not directory_name:
            return (
                False,
                "Directory name cannot be empty.",
            )

        if directory_name in {
            ".",
            "..",
        }:
            return (
                False,
                (
                    "Directory name cannot be "
                    "'.' or '..'."
                ),
            )

        if (
            "/" in directory_name
            or "\\" in directory_name
        ):
            return (
                False,
                (
                    "mkdir currently accepts only "
                    "a direct child directory name, "
                    "not a path."
                ),
            )

        if directory_name.startswith(
            "-"
        ):
            return (
                False,
                (
                    "mkdir options and flags are "
                    "not allowed."
                ),
            )

        if (
            SIMPLE_DIRECTORY_NAME_PATTERN
            .fullmatch(
                directory_name
            )
            is None
        ):
            return (
                False,
                (
                    "Directory name contains "
                    "characters not allowed by "
                    "the current policy."
                ),
            )

        return True, None

    return (
        False,
        (
            f"Executable '{executable}' "
            "has an unknown argument policy."
        ),
    )


def evaluate_process_policy(
    executable: str,
    args: list[str] | None = None,
    cwd: str | None = None,
    timeout_seconds: int = 10,
) -> dict[str, Any]:
    """
    Validate and classify a proposed native process without
    executing it.

    This function is deterministic application policy.

    It answers:
    - is the executable allowed?
    - are the arguments allowed?
    - is cwd inside the workspace?
    - is the timeout allowed?
    - what risk level applies?
    - does the action require approval?

    Actual execution must call run_process(), which evaluates
    this policy again before launching the process.
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

    if not executable:
        return {
            "ok": False,
            "status": "denied",
            "error": (
                "executable cannot be empty."
            ),
        }

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

    (
        arguments_valid,
        argument_error,
    ) = _validate_process_arguments(
        executable=executable,
        args=args,
        argument_policy=policy[
            "argument_policy"
        ],
    )

    if not arguments_valid:
        return {
            "ok": False,
            "status": "denied",
            "error": argument_error,
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

    return {
        "ok": True,
        "status": "allowed",

        "risk": policy[
            "risk"
        ],

        "requires_approval": policy[
            "requires_approval"
        ],

        "executable": executable,

        "executable_path": (
            executable_path
        ),

        "args": args,

        "cwd": str(
            resolved_cwd
        ),

        "timeout_seconds": (
            timeout_seconds
        ),
    }


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
    - policy is re-evaluated immediately before execution
    """

    decision = (
        evaluate_process_policy(
            executable=executable,
            args=args,
            cwd=cwd,
            timeout_seconds=(
                timeout_seconds
            ),
        )
    )

    if not decision.get(
        "ok",
        False,
    ):
        return decision

    executable = decision[
        "executable"
    ]

    executable_path = decision[
        "executable_path"
    ]

    args = decision[
        "args"
    ]

    resolved_cwd = Path(
        decision[
            "cwd"
        ]
    )

    timeout_seconds = decision[
        "timeout_seconds"
    ]

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

        stdout = (
            exc.stdout
            if isinstance(
                exc.stdout,
                str,
            )
            else ""
        )

        stderr = (
            exc.stderr
            if isinstance(
                exc.stderr,
                str,
            )
            else ""
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

            "stdout": (
                _bounded_output(
                    stdout
                )
            ),

            "stderr": (
                _bounded_output(
                    stderr
                )
            ),

            "timed_out": True,
            "duration_ms": duration_ms,
        }

    except OSError as exc:
        return {
            "ok": False,
            "status": "error",
            "error": (
                "Process launch failed: "
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
        "ok": (
            completed.returncode
            == 0
        ),

        "status": (
            "success"
            if completed.returncode
            == 0
            else "error"
        ),

        "executable": executable,
        "args": args,

        "cwd": str(
            resolved_cwd
        ),

        "exit_code": (
            completed.returncode
        ),

        "stdout": stdout,
        "stderr": stderr,

        "timed_out": False,
        "duration_ms": duration_ms,
    }