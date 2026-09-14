from pydantic import (
    BaseModel,
)

from mcp.server import (
    MCPServer,
)

from tools.developer_execution import (
    workspace_project_info
    as run_workspace_project_info,

    workspace_run_build
    as run_workspace_run_build,

    workspace_run_tests
    as run_workspace_run_tests,
)


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


def register_developer_tools(
    server: MCPServer,
) -> None:
    """
    Register governed developer execution capabilities with MCP.
    """

    @server.tool()
    def workspace_project_info(
        relative_path: str = ".",
    ) -> ProjectInfoResult:
        return ProjectInfoResult(
            **run_workspace_project_info(
                relative_path=(
                    relative_path
                )
            )
        )

    @server.tool()
    def workspace_run_tests(
        relative_path: str = ".",
        timeout_seconds: int = 60,
    ) -> DeveloperExecutionResult:
        return DeveloperExecutionResult(
            **run_workspace_run_tests(
                relative_path=(
                    relative_path
                ),
                timeout_seconds=(
                    timeout_seconds
                ),
            )
        )

    @server.tool()
    def workspace_run_build(
        relative_path: str = ".",
        timeout_seconds: int = 60,
    ) -> DeveloperExecutionResult:
        return DeveloperExecutionResult(
            **run_workspace_run_build(
                relative_path=(
                    relative_path
                ),
                timeout_seconds=(
                    timeout_seconds
                ),
            )
        )