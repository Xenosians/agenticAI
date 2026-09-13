from pydantic import (
    BaseModel,
)

from mcp.server import (
    MCPServer,
)

from config import (
    Settings,
)

from services.directory import (
    DirectoryService,
    build_directory_service,
)

from services.process_runner import (
    run_process,
)

from tools.workspace import (
    workspace_git_status
    as run_workspace_git_status,

    workspace_mkdir
    as run_workspace_mkdir,

    workspace_read_text
    as run_workspace_read_text,
)


class AccountStatusResult(
    BaseModel
):
    ok: bool

    user_id: (
        str | None
    ) = None

    enabled: (
        bool | None
    ) = None

    locked: (
        bool | None
    ) = None

    error: (
        str | None
    ) = None


class AccessCheckResult(
    BaseModel
):
    ok: bool

    user_id: (
        str | None
    ) = None

    resource: (
        str | None
    ) = None

    has_access: (
        bool | None
    ) = None

    error: (
        str | None
    ) = None


class MutationResult(
    BaseModel
):
    ok: bool

    status: str

    changed: (
        bool | None
    ) = None

    user_id: (
        str | None
    ) = None

    password_reset_count: (
        int | None
    ) = None

    message: (
        str | None
    ) = None

    error: (
        str | None
    ) = None


class ProcessExecResult(
    BaseModel
):
    ok: bool

    status: str

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


class WorkspaceReadTextResult(
    BaseModel
):
    ok: bool

    status: str

    path: (
        str | None
    ) = None

    content: (
        str | None
    ) = None

    truncated: (
        bool | None
    ) = None

    error: (
        str | None
    ) = None


def create_mcp_server(
    *,
    settings: Settings | None = None,
    directory: (
        DirectoryService | None
    ) = None,
) -> MCPServer:
    """
    Build one MCP server runtime.

    The MCP subprocess owns one DirectoryService instance.

    Tests may inject an isolated DirectoryService instead.

    There is no hidden directory-service cache or service
    locator.
    """

    runtime_settings = (
        settings
        if settings is not None
        else Settings()
    )

    directory_service = (
        directory
        if directory is not None
        else build_directory_service(
            runtime_settings
        )
    )

    server = (
        MCPServer(
            "ITSM Tools"
        )
    )

    @server.tool()
    def account_status(
        user_id: str,
    ) -> AccountStatusResult:
        result = (
            directory_service
            .account_status(
                user_id
            )
        )

        return (
            AccountStatusResult(
                **result
            )
        )

    @server.tool()
    def check_access(
        user_id: str,
        resource: str,
    ) -> AccessCheckResult:
        result = (
            directory_service
            .check_access(
                user_id,
                resource,
            )
        )

        return (
            AccessCheckResult(
                **result
            )
        )

    @server.tool()
    def unlock_user(
        user_id: str,
    ) -> MutationResult:
        result = (
            directory_service
            .unlock_user(
                user_id
            )
        )

        return (
            MutationResult(
                **result
            )
        )

    @server.tool()
    def reset_password(
        user_id: str,
    ) -> MutationResult:
        result = (
            directory_service
            .reset_password(
                user_id
            )
        )

        return (
            MutationResult(
                **result
            )
        )

    @server.tool()
    def process_exec(
        executable: str,
        args: (
            list[str] | None
        ) = None,
        cwd: (
            str | None
        ) = None,
        timeout_seconds: int = 10,
    ) -> ProcessExecResult:
        result = (
            run_process(
                executable=(
                    executable
                ),
                args=(
                    args
                ),
                cwd=(
                    cwd
                ),
                timeout_seconds=(
                    timeout_seconds
                ),
            )
        )

        return (
            ProcessExecResult(
                **result
            )
        )

    @server.tool()
    def workspace_mkdir(
        directory_name: str,
        cwd: (
            str | None
        ) = None,
        timeout_seconds: int = 10,
    ) -> ProcessExecResult:
        result = (
            run_workspace_mkdir(
                directory_name=(
                    directory_name
                ),
                cwd=(
                    cwd
                ),
                timeout_seconds=(
                    timeout_seconds
                ),
            )
        )

        return (
            ProcessExecResult(
                **result
            )
        )

    @server.tool()
    def workspace_read_text(
        relative_path: str,
    ) -> WorkspaceReadTextResult:
        result = (
            run_workspace_read_text(
                relative_path=(
                    relative_path
                ),
            )
        )

        return (
            WorkspaceReadTextResult(
                **result
            )
        )

    @server.tool()
    def workspace_git_status(
    ) -> ProcessExecResult:
        result = (
            run_workspace_git_status()
        )

        return (
            ProcessExecResult(
                **result
            )
        )

    return server


# MCP subprocess composition root.
#
# This is one explicit server instance owned by this process,
# not a hidden service locator.
mcp = (
    create_mcp_server()
)


if __name__ == "__main__":
    mcp.run()