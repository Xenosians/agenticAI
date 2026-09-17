from __future__ import annotations

import sys

from dataclasses import (
    dataclass,
)

from pathlib import Path
from typing import (
    Any,
    Protocol,
)

from services.process_runner import (
    MAX_TRUSTED_TIMEOUT_SECONDS,
    resolve_cwd,
    run_trusted_process,
)


DEFAULT_EXECUTION_TIMEOUT_SECONDS = 60


@dataclass(
    frozen=True
)
class CommandSpec:
    executable: str
    args: tuple[
        str,
        ...,
    ]


class ProjectStrategy(
    Protocol
):
    name: str

    def detect(
        self,
        project_dir: Path,
    ) -> list[str]:
        ...

    def test_command(
        self,
        project_dir: Path,
    ) -> CommandSpec:
        ...

    def build_command(
        self,
        project_dir: Path,
    ) -> CommandSpec:
        ...


class PythonProjectStrategy:
    name = "python"

    MARKERS = (
        "pyproject.toml",
        "pytest.ini",
        "setup.cfg",
        "setup.py",
        "requirements.txt",
        "requirements-dev.txt",
    )

    def detect(
        self,
        project_dir: Path,
    ) -> list[str]:
        markers = [
            marker

            for marker
            in self.MARKERS

            if (
                project_dir
                / marker
            ).is_file()
        ]

        if not markers:
            python_files = list(
                project_dir.glob(
                    "*.py"
                )
            )

            if python_files:
                markers.append(
                    "*.py"
                )

        return markers

    def test_command(
        self,
        project_dir: Path,
    ) -> CommandSpec:
        return CommandSpec(
            executable=(
                sys.executable
            ),
            args=(
                "-m",
                "pytest",
                "-q",
            ),
        )

    def build_command(
        self,
        project_dir: Path,
    ) -> CommandSpec:
        return CommandSpec(
            executable=(
                sys.executable
            ),
            args=(
                "-m",
                "compileall",
                "-q",
                ".",
            ),
        )


class ElixirProjectStrategy:
    name = "elixir"

    def detect(
        self,
        project_dir: Path,
    ) -> list[str]:
        if (
            project_dir
            / "mix.exs"
        ).is_file():
            return [
                "mix.exs"
            ]

        return []

    def test_command(
        self,
        project_dir: Path,
    ) -> CommandSpec:
        return CommandSpec(
            executable="mix",
            args=(
                "test",
            ),
        )

    def build_command(
        self,
        project_dir: Path,
    ) -> CommandSpec:
        return CommandSpec(
            executable="mix",
            args=(
                "compile",
            ),
        )


class NimProjectStrategy:
    name = "nim"

    def detect(
        self,
        project_dir: Path,
    ) -> list[str]:
        nimble_files = sorted(
            project_dir.glob(
                "*.nimble"
            )
        )

        return [
            path.name
            for path
            in nimble_files
        ]

    def test_command(
        self,
        project_dir: Path,
    ) -> CommandSpec:
        return CommandSpec(
            executable="nimble",
            args=(
                "test",
            ),
        )

    def build_command(
        self,
        project_dir: Path,
    ) -> CommandSpec:
        return CommandSpec(
            executable="nimble",
            args=(
                "build",
            ),
        )


PROJECT_STRATEGIES: tuple[
    ProjectStrategy,
    ...,
] = (
    PythonProjectStrategy(),
    ElixirProjectStrategy(),
    NimProjectStrategy(),
)


def _resolve_project_directory(
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
        project_dir = (
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
        project_dir,
        None,
    )


def _detect_project(
    project_dir: Path,
) -> tuple[
    ProjectStrategy | None,
    list[str],
    str | None,
]:
    matches: list[
        tuple[
            ProjectStrategy,
            list[str],
        ]
    ] = []

    for strategy in (
        PROJECT_STRATEGIES
    ):
        markers = (
            strategy.detect(
                project_dir
            )
        )

        if markers:
            matches.append(
                (
                    strategy,
                    markers,
                )
            )

    if not matches:
        return (
            None,
            [],
            (
                "No supported Python, Elixir, "
                "or Nim project was detected "
                "at this workspace path."
            ),
        )

    if len(
        matches
    ) > 1:
        project_types = ", ".join(
            strategy.name
            for (
                strategy,
                _markers,
            )
            in matches
        )

        return (
            None,
            [],
            (
                "Multiple project types were detected "
                f"at this path: {project_types}. "
                "Use a narrower project directory."
            ),
        )

    (
        strategy,
        markers,
    ) = matches[
        0
    ]

    return (
        strategy,
        markers,
        None,
    )


def workspace_project_info(
    relative_path: str = ".",
) -> dict[str, Any]:
    """
    Detect the supported project/runtime type at one workspace
    directory without executing project code.
    """

    (
        project_dir,
        path_error,
    ) = _resolve_project_directory(
        relative_path
    )

    if (
        project_dir is None
    ):
        return {
            "ok": False,
            "status": "denied",
            "error":
                path_error,
        }

    (
        strategy,
        markers,
        detection_error,
    ) = _detect_project(
        project_dir
    )

    if strategy is None:
        return {
            "ok": False,
            "status": "error",
            "error":
                detection_error,
        }

    root = (
        resolve_cwd(
            "."
        )
    )

    normalized_path = (
        "."
        if project_dir == root
        else (
            project_dir
            .relative_to(
                root
            )
            .as_posix()
        )
    )

    return {
        "ok": True,
        "status": "success",

        "project_type":
            strategy.name,

        "path":
            normalized_path,

        "markers":
            markers,

        "supported_operations": [
            "tests",
            "build",
        ],

        "error":
            None,
    }


def _validate_execution_timeout(
    timeout_seconds: int,
) -> str | None:
    if (
        not isinstance(
            timeout_seconds,
            int,
        )
        or timeout_seconds < 1
        or timeout_seconds
        > MAX_TRUSTED_TIMEOUT_SECONDS
    ):
        return (
            "timeout_seconds must be between "
            f"1 and {MAX_TRUSTED_TIMEOUT_SECONDS}."
        )

    return None


def _run_project_operation(
    *,
    operation: str,
    relative_path: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    timeout_error = (
        _validate_execution_timeout(
            timeout_seconds
        )
    )

    if timeout_error is not None:
        return {
            "ok": False,
            "status": "denied",
            "error":
                timeout_error,
        }

    (
        project_dir,
        path_error,
    ) = _resolve_project_directory(
        relative_path
    )

    if project_dir is None:
        return {
            "ok": False,
            "status": "denied",
            "error":
                path_error,
        }

    (
        strategy,
        _markers,
        detection_error,
    ) = _detect_project(
        project_dir
    )

    if strategy is None:
        return {
            "ok": False,
            "status": "error",
            "error":
                detection_error,
        }

    if operation == "tests":
        command = (
            strategy.test_command(
                project_dir
            )
        )

    elif operation == "build":
        command = (
            strategy.build_command(
                project_dir
            )
        )

    else:
        return {
            "ok": False,
            "status": "error",
            "error": (
                "Unsupported developer "
                f"operation '{operation}'."
            ),
        }

    result = (
        run_trusted_process(
            executable=(
                command.executable
            ),
            args=list(
                command.args
            ),
            cwd=str(
                project_dir
            ),
            timeout_seconds=(
                timeout_seconds
            ),
        )
    )

    root = (
        resolve_cwd(
            "."
        )
    )

    project_path = (
        "."
        if project_dir == root
        else (
            project_dir
            .relative_to(
                root
            )
            .as_posix()
        )
    )

    return {
        **result,

        "operation":
            operation,

        "project_type":
            strategy.name,

        "project_path":
            project_path,
    }


def workspace_run_tests(
    relative_path: str = ".",
    timeout_seconds: int = (
        DEFAULT_EXECUTION_TIMEOUT_SECONDS
    ),
) -> dict[str, Any]:
    """
    Execute the trusted test command for one detected project.

    ToolGateway approval must occur before this MCP function is
    called because project tests execute repository code.
    """

    return (
        _run_project_operation(
            operation="tests",
            relative_path=(
                relative_path
            ),
            timeout_seconds=(
                timeout_seconds
            ),
        )
    )


def workspace_run_build(
    relative_path: str = ".",
    timeout_seconds: int = (
        DEFAULT_EXECUTION_TIMEOUT_SECONDS
    ),
) -> dict[str, Any]:
    """
    Execute the trusted build/check command for one detected
    project.

    ToolGateway approval must occur before this MCP function is
    called because build systems may execute repository code.
    """

    return (
        _run_project_operation(
            operation="build",
            relative_path=(
                relative_path
            ),
            timeout_seconds=(
                timeout_seconds
            ),
        )
    )