from pydantic import BaseModel
from mcp.server import MCPServer

from services.directory import (
    get_directory_service,
)

from services.process_runner import (
    run_process,
)

from tools.workspace import (
    workspace_mkdir as run_workspace_mkdir,
    workspace_read_text as run_workspace_read_text,
)


mcp = MCPServer(
    "ITSM Tools"
)


class AccountStatusResult(
    BaseModel
):
    ok: bool

    user_id: str | None = None

    enabled: bool | None = None
    locked: bool | None = None

    error: str | None = None


class AccessCheckResult(
    BaseModel
):
    ok: bool

    user_id: str | None = None
    resource: str | None = None

    has_access: bool | None = None

    error: str | None = None


class MutationResult(
    BaseModel
):
    ok: bool
    status: str

    changed: bool | None = None
    user_id: str | None = None

    password_reset_count: (
        int | None
    ) = None

    message: str | None = None
    error: str | None = None


class ProcessExecResult(
    BaseModel
):
    ok: bool
    status: str

    executable: str | None = None

    args: (
        list[str] | None
    ) = None

    cwd: str | None = None

    exit_code: int | None = None

    stdout: str | None = None
    stderr: str | None = None

    timed_out: bool | None = None

    duration_ms: int | None = None

    error: str | None = None


class WorkspaceReadTextResult(
    BaseModel
):
    ok: bool
    status: str

    path: str | None = None
    content: str | None = None
    truncated: bool | None = None

    error: str | None = None


@mcp.tool()
def account_status(
    user_id: str,
) -> AccountStatusResult:
    directory = (
        get_directory_service()
    )

    result = (
        directory.account_status(
            user_id
        )
    )

    return AccountStatusResult(
        **result
    )


@mcp.tool()
def check_access(
    user_id: str,
    resource: str,
) -> AccessCheckResult:
    directory = (
        get_directory_service()
    )

    result = (
        directory.check_access(
            user_id,
            resource,
        )
    )

    return AccessCheckResult(
        **result
    )


@mcp.tool()
def unlock_user(
    user_id: str,
) -> MutationResult:
    directory = (
        get_directory_service()
    )

    result = (
        directory.unlock_user(
            user_id
        )
    )

    return MutationResult(
        **result
    )


@mcp.tool()
def reset_password(
    user_id: str,
) -> MutationResult:
    directory = (
        get_directory_service()
    )

    result = (
        directory.reset_password(
            user_id
        )
    )

    return MutationResult(
        **result
    )


@mcp.tool()
def process_exec(
    executable: str,
    args: list[str] | None = None,
    cwd: str | None = None,
    timeout_seconds: int = 10,
) -> ProcessExecResult:
    """
    Execute a structured native process through the trusted
    local process runner.
    """

    result = run_process(
        executable=executable,
        args=args,
        cwd=cwd,
        timeout_seconds=(
            timeout_seconds
        ),
    )

    return ProcessExecResult(
        **result
    )


@mcp.tool()
def workspace_mkdir(
    directory_name: str,
    cwd: str | None = None,
    timeout_seconds: int = 10,
) -> ProcessExecResult:
    """
    Create one direct-child workspace directory through the
    trusted process runner.
    """

    result = run_workspace_mkdir(
        directory_name=(
            directory_name
        ),
        cwd=cwd,
        timeout_seconds=(
            timeout_seconds
        ),
    )

    return ProcessExecResult(
        **result
    )


@mcp.tool()
def workspace_read_text(
    relative_path: str,
) -> WorkspaceReadTextResult:
    """
    Read one approved UTF-8 text/source file from the
    configured developer workspace.
    """

    result = run_workspace_read_text(
        relative_path=relative_path,
    )

    return WorkspaceReadTextResult(
        **result
    )


if __name__ == "__main__":
    mcp.run()