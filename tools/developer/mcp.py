from typing import Any

from pydantic import (
    BaseModel,
)

from mcp.server import (
    MCPServer,
)

from tools.developer.execution import (
    workspace_project_info
    as run_workspace_project_info,

    workspace_run_build
    as run_workspace_run_build,

    workspace_run_tests
    as run_workspace_run_tests,
)

from tools.developer.runtime import (
    workspace_process_snapshot
    as run_workspace_process_snapshot,

    workspace_service_logs
    as run_workspace_service_logs,

    workspace_service_status
    as run_workspace_service_status,
)


DEFAULT_RELATIVE_PATH = "."
DEFAULT_EXECUTION_TIMEOUT_SECONDS = 60


class ProjectInfoResult(
    BaseModel
):
    ok: bool
    status: str

    project_type: (
        str | None
    ) = None

    path: (
        str | None
    ) = None

    markers: (
        list[str] | None
    ) = None

    supported_operations: (
        list[str] | None
    ) = None

    error: (
        str | None
    ) = None


class DeveloperExecutionResult(
    BaseModel
):
    ok: bool
    status: str

    operation: (
        str | None
    ) = None

    project_type: (
        str | None
    ) = None

    project_path: (
        str | None
    ) = None

    executable: (
        str | None
    ) = None

    args: (
        list[str] | None
    ) = None

    cwd: (
        str | None
    ) = None

    exit_code: (
        int | None
    ) = None

    stdout: (
        str | None
    ) = None

    stderr: (
        str | None
    ) = None

    timed_out: (
        bool | None
    ) = None

    duration_ms: (
        int | None
    ) = None

    error: (
        str | None
    ) = None


class ProcessSnapshotResult(
    BaseModel
):
    ok: bool
    status: str

    provider: (
        str | None
    ) = None

    processes: (
        list[
            dict[
                str,
                Any,
            ]
        ]
        | None
    ) = None

    count: (
        int | None
    ) = None

    truncated: (
        bool | None
    ) = None

    error: (
        str | None
    ) = None


class ServiceStatusResult(
    BaseModel
):
    ok: bool
    status: str

    provider: (
        str | None
    ) = None

    services: (
        list[
            dict[
                str,
                Any,
            ]
        ]
        | None
    ) = None

    service_count: (
        int | None
    ) = None

    error: (
        str | None
    ) = None


class ServiceLogsResult(
    BaseModel
):
    ok: bool
    status: str

    provider: (
        str | None
    ) = None

    service_name: (
        str | None
    ) = None

    tail_lines: (
        int | None
    ) = None

    logs: (
        str | None
    ) = None

    redaction_applied: (
        bool | None
    ) = None

    error: (
        str | None
    ) = None


def _relative_path_or_default(
    relative_path: str | None,
) -> str:
    """
    Canonicalize an omitted/null optional workspace path.

    The model may omit the field or emit JSON null for an
    optional path. Neither gives the model additional authority:
    both mean the trusted workspace root.
    """

    if relative_path is None:
        return DEFAULT_RELATIVE_PATH

    normalized = (
        relative_path.strip()
    )

    if not normalized:
        return DEFAULT_RELATIVE_PATH

    return normalized


def _timeout_or_default(
    timeout_seconds: int | None,
) -> int:
    """
    Canonicalize an omitted/null optional execution timeout.

    Range enforcement still belongs to the trusted execution
    layer.
    """

    if timeout_seconds is None:
        return (
            DEFAULT_EXECUTION_TIMEOUT_SECONDS
        )

    return timeout_seconds


def register_developer_tools(
    server: MCPServer,
) -> None:
    """
    Register governed developer capabilities with MCP.

    This is the developer-tool composition boundary. The main MCP
    server only needs to register this feature group once.

    Optional model-facing arguments are canonicalized here before
    entering trusted adapters. A JSON null for an optional field
    therefore behaves like an omitted field instead of failing
    MCP/Pydantic validation before deterministic policy can run.
    """

    # ============================================================
    # PROJECT DETECTION / EXECUTION
    # ============================================================

    @server.tool()
    def workspace_project_info(
        relative_path: (
            str | None
        ) = None,
    ) -> ProjectInfoResult:
        return ProjectInfoResult(
            **run_workspace_project_info(
                relative_path=(
                    _relative_path_or_default(
                        relative_path
                    )
                )
            )
        )

    @server.tool()
    def workspace_run_tests(
        relative_path: (
            str | None
        ) = None,
        timeout_seconds: (
            int | None
        ) = None,
    ) -> DeveloperExecutionResult:
        return DeveloperExecutionResult(
            **run_workspace_run_tests(
                relative_path=(
                    _relative_path_or_default(
                        relative_path
                    )
                ),
                timeout_seconds=(
                    _timeout_or_default(
                        timeout_seconds
                    )
                ),
            )
        )

    @server.tool()
    def workspace_run_build(
        relative_path: (
            str | None
        ) = None,
        timeout_seconds: (
            int | None
        ) = None,
    ) -> DeveloperExecutionResult:
        return DeveloperExecutionResult(
            **run_workspace_run_build(
                relative_path=(
                    _relative_path_or_default(
                        relative_path
                    )
                ),
                timeout_seconds=(
                    _timeout_or_default(
                        timeout_seconds
                    )
                ),
            )
        )

    # ============================================================
    # RUNTIME / SERVICE INSPECTION
    # ============================================================

    @server.tool()
    def workspace_process_snapshot(
    ) -> ProcessSnapshotResult:
        return ProcessSnapshotResult(
            **run_workspace_process_snapshot()
        )

    @server.tool()
    def workspace_service_status(
        relative_path: (
            str | None
        ) = None,
    ) -> ServiceStatusResult:
        return ServiceStatusResult(
            **run_workspace_service_status(
                relative_path=(
                    _relative_path_or_default(
                        relative_path
                    )
                )
            )
        )

    @server.tool()
    def workspace_service_logs(
        service_name: str,
        relative_path: (
            str | None
        ) = None,
    ) -> ServiceLogsResult:
        return ServiceLogsResult(
            **run_workspace_service_logs(
                service_name=(
                    service_name
                ),
                relative_path=(
                    _relative_path_or_default(
                        relative_path
                    )
                ),
            )
        )