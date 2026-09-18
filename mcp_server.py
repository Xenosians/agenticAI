from typing import (
    Any,
)

from pydantic import (
    BaseModel,
)

from mcp.server import (
    MCPServer,
)

from config import (
    Settings,
)

from services.assets import (
    AssetMutationService,
    AssetService,
    build_asset_mutation_service,
    build_asset_service,
)

from services.directory import (
    AccessMutationService,
    AccountLifecycleService,
    DirectoryService,
    build_access_mutation_service,
    build_account_lifecycle_service,
    build_directory_service,
)

from services.process_runner import (
    run_process,
)

from services.ticketing import (
    TicketMutationService,
    TicketService,
    build_ticket_mutation_service,
    build_ticket_service,
)

from tools.access.mcp import (
    register_access_mutation_tools,
)

from tools.account.mcp import (
    register_account_lifecycle_tools,
)

from tools.assets.mcp import (
    register_asset_tools,
)

from tools.developer.mcp import (
    register_developer_tools,
)

from tools.git.mcp import (
    register_git_tools,
)

from tools.ticketing.mcp import (
    register_ticketing_tools,
)

from tools.workspace import (
    workspace_mkdir
    as run_workspace_mkdir,

    workspace_read_text
    as run_workspace_read_text,
)

from tools.workspace.discovery import (
    workspace_file_info
    as run_workspace_file_info,

    workspace_list
    as run_workspace_list,

    workspace_search
    as run_workspace_search,
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
    password_reset_count: int | None = None
    message: str | None = None
    error: str | None = None


class ProcessExecResult(
    BaseModel
):
    ok: bool
    status: str
    executable: str | None = None
    args: list[str] | None = None
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


class WorkspaceListResult(
    BaseModel
):
    ok: bool
    status: str
    path: str | None = None

    entries: (
        list[
            dict[
                str,
                Any,
            ]
        ]
        | None
    ) = None

    truncated: bool | None = None
    error: str | None = None


class WorkspaceSearchResult(
    BaseModel
):
    ok: bool
    status: str
    query: str | None = None
    path: str | None = None

    matches: (
        list[
            dict[
                str,
                Any,
            ]
        ]
        | None
    ) = None

    files_scanned: int | None = None
    truncated: bool | None = None
    error: str | None = None


class WorkspaceFileInfoResult(
    BaseModel
):
    ok: bool
    status: str
    path: str | None = None
    name: str | None = None
    type: str | None = None
    size_bytes: int | None = None
    suffix: str | None = None
    readable_text: bool | None = None
    error: str | None = None


def create_mcp_server(
    *,
    settings: Settings | None = None,

    directory: (
        DirectoryService | None
    ) = None,

    account_lifecycle: (
        AccountLifecycleService | None
    ) = None,

    access_mutations: (
        AccessMutationService | None
    ) = None,

    assets: (
        AssetService | None
    ) = None,

    asset_mutations: (
        AssetMutationService | None
    ) = None,

    ticketing: (
        TicketService | None
    ) = None,

    ticket_mutations: (
        TicketMutationService | None
    ) = None,
) -> MCPServer:

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

    account_lifecycle_service = (
        account_lifecycle
        if account_lifecycle
        is not None
        else build_account_lifecycle_service(
            directory_service
        )
    )

    access_mutation_service = (
        access_mutations
        if access_mutations
        is not None
        else build_access_mutation_service(
            directory_service
        )
    )

    asset_service = (
        assets
        if assets is not None
        else build_asset_service()
    )

    asset_mutation_service = (
        asset_mutations
        if asset_mutations
        is not None
        else build_asset_mutation_service(
            asset_service
        )
    )

    ticket_service = (
        ticketing
        if ticketing is not None
        else build_ticket_service(
            runtime_settings
        )
    )

    ticket_mutation_service = (
        ticket_mutations
        if ticket_mutations
        is not None
        else build_ticket_mutation_service(
            ticket_service
        )
    )

    server = (
        MCPServer(
            "ITSM Tools"
        )
    )

    # ============================================================
    # DIRECTORY / IDENTITY
    # ============================================================

    @server.tool()
    def account_status(
        user_id: str,
    ) -> AccountStatusResult:

        return (
            AccountStatusResult(
                **directory_service
                .account_status(
                    user_id
                )
            )
        )

    register_account_lifecycle_tools(
        server,
        account_lifecycle_service,
    )

    @server.tool()
    def check_access(
        user_id: str,
        resource: str,
    ) -> AccessCheckResult:

        return (
            AccessCheckResult(
                **directory_service
                .check_access(
                    user_id,
                    resource,
                )
            )
        )

    register_access_mutation_tools(
        server,
        access_mutation_service,
    )

    @server.tool()
    def unlock_user(
        user_id: str,
    ) -> MutationResult:

        return (
            MutationResult(
                **directory_service
                .unlock_user(
                    user_id
                )
            )
        )

    @server.tool()
    def reset_password(
        user_id: str,
    ) -> MutationResult:

        return (
            MutationResult(
                **directory_service
                .reset_password(
                    user_id
                )
            )
        )

    # ============================================================
    # ASSET / DEVICE INVENTORY
    # ============================================================

    register_asset_tools(
        server,
        asset_service,
        asset_mutation_service,
    )

    # ============================================================
    # GENERIC PROCESS
    # ============================================================

    @server.tool()
    def process_exec(
        executable: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        timeout_seconds: int = 10,
    ) -> ProcessExecResult:

        return (
            ProcessExecResult(
                **run_process(
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
        )

    # ============================================================
    # WORKSPACE
    # ============================================================

    @server.tool()
    def workspace_mkdir(
        directory_name: str,
        cwd: str | None = None,
        timeout_seconds: int = 10,
    ) -> ProcessExecResult:

        return (
            ProcessExecResult(
                **run_workspace_mkdir(
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
        )

    @server.tool()
    def workspace_read_text(
        relative_path: str,
    ) -> WorkspaceReadTextResult:

        return (
            WorkspaceReadTextResult(
                **run_workspace_read_text(
                    relative_path=(
                        relative_path
                    )
                )
            )
        )

    @server.tool()
    def workspace_list(
        relative_path: str = ".",
    ) -> WorkspaceListResult:

        return (
            WorkspaceListResult(
                **run_workspace_list(
                    relative_path=(
                        relative_path
                    )
                )
            )
        )

    @server.tool()
    def workspace_search(
        query: str,
        relative_path: str = ".",
    ) -> WorkspaceSearchResult:

        return (
            WorkspaceSearchResult(
                **run_workspace_search(
                    query=(
                        query
                    ),

                    relative_path=(
                        relative_path
                    ),
                )
            )
        )

    @server.tool()
    def workspace_file_info(
        relative_path: str,
    ) -> WorkspaceFileInfoResult:

        return (
            WorkspaceFileInfoResult(
                **run_workspace_file_info(
                    relative_path=(
                        relative_path
                    )
                )
            )
        )

    # ============================================================
    # GIT — READ ONLY
    # ============================================================

    register_git_tools(
        server
    )

    # ============================================================
    # TICKETING
    # ============================================================

    register_ticketing_tools(
        server,
        ticket_service,
        ticket_mutation_service,
    )

    # ============================================================
    # GOVERNED DEVELOPER CAPABILITIES
    # ============================================================

    register_developer_tools(
        server
    )

    return (
        server
    )


mcp = (
    create_mcp_server()
)


if __name__ == "__main__":

    mcp.run()
