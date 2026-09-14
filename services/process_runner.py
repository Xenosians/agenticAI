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
MAX_TRUSTED_TIMEOUT_SECONDS = 120
MAX_OUTPUT_CHARS = 16_000


# ============================================================
# GENERIC MODEL-FACING PROCESS POLICY
#
# Higher-level capability adapters may use run_trusted_process()
# with fixed command shapes.
#
# process_exec remains intentionally much narrower.
# ============================================================


GIT_READ_ARGUMENTS = {
    (
        "status",
        "--short",
        "--branch",
    ),
    (
        "branch",
        "--list",
        "--no-color",
    ),
    (
        "log",
        "--oneline",
        "--no-decorate",
        "-n",
        "20",
    ),
    (
        "diff",
        "--no-ext-diff",
        "--no-color",
        "--unified=3",
    ),
    (
        "diff",
        "--no-ext-diff",
        "--name-only",
    ),
}


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
        "argument_policy": "git_read",
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
    Return the configured developer workspace root.
    """

    settings = get_settings()

    return settings.resolve_project_path(
        settings.process_workspace_root
    )


def resolve_cwd(
    cwd: str | None,
) -> Path:
    """
    Resolve a working directory and guarantee that it remains
    inside PROCESS_WORKSPACE_ROOT.
    """

    root = (
        workspace_root()
        .resolve()
    )

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
        value[
            :MAX_OUTPUT_CHARS
        ]
        + "\n...[output truncated]"
    )


def _validate_git_arguments(
    args: list[str],
) -> tuple[
    bool,
    str | None,
]:
    normalized = tuple(
        args
    )

    if normalized in GIT_READ_ARGUMENTS:
        return True, None

    return (
        False,
        (
            "Git arguments are not permitted by "
            "the current read-only Git policy."
        ),
    )


def _validate_process_arguments(
    executable: str,
    args: list[str],
    argument_policy: str,
) -> tuple[
    bool,
    str | None,
]:
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

    if argument_policy == "git_read":
        return _validate_git_arguments(
            args
        )

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

        directory_name = args[
            0
        ]

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


def _validate_common_process_request(
    *,
    executable: str,
    args: list[str] | None,
    cwd: str | None,
    timeout_seconds: int,
    max_timeout_seconds: int,
) -> dict[str, Any]:
    """
    Validate properties shared by all trusted native execution.

    This does NOT decide whether a model is authorized to choose
    a particular executable or argument shape.
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

    executable = (
        executable.strip()
    )

    if not executable:
        return {
            "ok": False,
            "status": "denied",
            "error": (
                "executable cannot be empty."
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
        isinstance(
            argument,
            str,
        )
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
        not isinstance(
            timeout_seconds,
            int,
        )
        or timeout_seconds < 1
        or timeout_seconds
        > max_timeout_seconds
    ):
        return {
            "ok": False,
            "status": "denied",
            "error": (
                "timeout_seconds must be "
                f"between 1 and "
                f"{max_timeout_seconds}."
            ),
        }

    try:
        resolved_cwd = (
            resolve_cwd(
                cwd
            )
        )

    except ValueError as exc:
        return {
            "ok": False,
            "status": "denied",
            "error": str(
                exc
            ),
        }

    executable_path = (
        shutil.which(
            executable
        )
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

        "executable":
            executable,

        "executable_path":
            executable_path,

        "args":
            args,

        "cwd": str(
            resolved_cwd
        ),

        "timeout_seconds":
            timeout_seconds,
    }


def _execute_validated_process(
    decision: dict[
        str,
        Any,
    ],
) -> dict[str, Any]:
    executable = (
        decision[
            "executable"
        ]
    )

    executable_path = (
        decision[
            "executable_path"
        ]
    )

    args = (
        decision[
            "args"
        ]
    )

    resolved_cwd = Path(
        decision[
            "cwd"
        ]
    )

    timeout_seconds = (
        decision[
            "timeout_seconds"
        ]
    )

    command = [
        executable_path,
        *args,
    ]

    started_at = (
        time.monotonic()
    )

    try:
        completed = (
            subprocess.run(
                command,
                cwd=str(
                    resolved_cwd
                ),
                capture_output=True,
                text=True,
                timeout=(
                    timeout_seconds
                ),
                shell=False,
                check=False,
            )
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

            "executable":
                executable,

            "args":
                args,

            "cwd": str(
                resolved_cwd
            ),

            "exit_code":
                None,

            "stdout":
                _bounded_output(
                    stdout
                ),

            "stderr":
                _bounded_output(
                    stderr
                ),

            "timed_out":
                True,

            "duration_ms":
                duration_ms,
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

        "executable":
            executable,

        "args":
            args,

        "cwd": str(
            resolved_cwd
        ),

        "exit_code":
            completed.returncode,

        "stdout":
            _bounded_output(
                completed.stdout
            ),

        "stderr":
            _bounded_output(
                completed.stderr
            ),

        "timed_out":
            False,

        "duration_ms":
            duration_ms,
    }


def run_trusted_process(
    executable: str,
    args: list[str] | None = None,
    cwd: str | None = None,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """
    Execute a native process selected by trusted application code.

    IMPORTANT:

    This function is NOT a model-facing arbitrary command
    capability.

    Capability adapters using this function must own the
    executable and argument shape themselves.

    Workspace confinement, executable discovery, timeout,
    shell=False, and bounded output still apply.
    """

    decision = (
        _validate_common_process_request(
            executable=executable,
            args=args,
            cwd=cwd,
            timeout_seconds=(
                timeout_seconds
            ),
            max_timeout_seconds=(
                MAX_TRUSTED_TIMEOUT_SECONDS
            ),
        )
    )

    if not decision.get(
        "ok",
        False,
    ):
        return decision

    return (
        _execute_validated_process(
            decision
        )
    )


def evaluate_process_policy(
    executable: str,
    args: list[str] | None = None,
    cwd: str | None = None,
    timeout_seconds: int = 10,
) -> dict[str, Any]:
    """
    Validate the narrow generic process_exec capability.
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

    executable = (
        executable.strip()
    )

    policy = (
        ALLOWED_EXECUTABLES.get(
            executable
        )
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

    normalized_args = (
        []
        if args is None
        else args
    )

    if not isinstance(
        normalized_args,
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
        isinstance(
            argument,
            str,
        )
        for argument
        in normalized_args
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
        args=normalized_args,
        argument_policy=policy[
            "argument_policy"
        ],
    )

    if not arguments_valid:
        return {
            "ok": False,
            "status": "denied",
            "error":
                argument_error,
        }

    decision = (
        _validate_common_process_request(
            executable=executable,
            args=normalized_args,
            cwd=cwd,
            timeout_seconds=(
                timeout_seconds
            ),
            max_timeout_seconds=(
                MAX_TIMEOUT_SECONDS
            ),
        )
    )

    if not decision.get(
        "ok",
        False,
    ):
        return decision

    decision[
        "risk"
    ] = policy[
        "risk"
    ]

    decision[
        "requires_approval"
    ] = policy[
        "requires_approval"
    ]

    return decision


def run_process(
    executable: str,
    args: list[str] | None = None,
    cwd: str | None = None,
    timeout_seconds: int = 10,
) -> dict[str, Any]:
    """
    Execute one generic model-facing policy-approved process.
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

    return (
        _execute_validated_process(
            decision
        )
    )