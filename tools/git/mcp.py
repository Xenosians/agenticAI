from pydantic import (
    BaseModel,
    Field,
)

from mcp.server import (
    MCPServer,
)

from tools.git import (
    workspace_git_branches as run_git_branches,
    workspace_git_changed_files as run_git_changed_files,
    workspace_git_diff as run_git_diff,
    workspace_git_log as run_git_log,
    workspace_git_status as run_git_status,
)


class GitStatusChange(
    BaseModel
):
    code: str
    path: str


class GitStatusMCPResult(
    BaseModel
):
    ok: bool
    status: str

    repository: (
        str | None
    ) = None

    branch: (
        str | None
    ) = None

    upstream: (
        str | None
    ) = None

    ahead: int = 0
    behind: int = 0

    clean: (
        bool | None
    ) = None

    changes: list[
        GitStatusChange
    ] = Field(
        default_factory=list
    )

    change_count: int = 0

    error: (
        str | None
    ) = None


class GitBranchItem(
    BaseModel
):
    name: str
    current: bool


class GitBranchesMCPResult(
    BaseModel
):
    ok: bool
    status: str

    repository: (
        str | None
    ) = None

    current_branch: (
        str | None
    ) = None

    branches: list[
        GitBranchItem
    ] = Field(
        default_factory=list
    )

    count: int = 0

    error: (
        str | None
    ) = None


class GitCommitItem(
    BaseModel
):
    hash: str
    message: str


class GitLogMCPResult(
    BaseModel
):
    ok: bool
    status: str

    repository: (
        str | None
    ) = None

    commits: list[
        GitCommitItem
    ] = Field(
        default_factory=list
    )

    count: int = 0

    error: (
        str | None
    ) = None


class GitDiffMCPResult(
    BaseModel
):
    ok: bool
    status: str

    repository: (
        str | None
    ) = None

    diff: str = ""

    has_changes: bool = False
    truncated: bool = False

    error: (
        str | None
    ) = None


class GitChangedFilesMCPResult(
    BaseModel
):
    ok: bool
    status: str

    repository: (
        str | None
    ) = None

    files: list[
        str
    ] = Field(
        default_factory=list
    )

    count: int = 0

    error: (
        str | None
    ) = None


def register_git_tools(
    server: MCPServer,
) -> None:

    @server.tool()
    def workspace_git_status(
        repository: (
            str | None
        ) = None,
    ) -> GitStatusMCPResult:

        return (
            GitStatusMCPResult(
                **run_git_status(
                    repository=repository
                )
            )
        )

    @server.tool()
    def workspace_git_branches(
        repository: (
            str | None
        ) = None,
    ) -> GitBranchesMCPResult:

        return (
            GitBranchesMCPResult(
                **run_git_branches(
                    repository=repository
                )
            )
        )

    @server.tool()
    def workspace_git_log(
        repository: (
            str | None
        ) = None,
    ) -> GitLogMCPResult:

        return (
            GitLogMCPResult(
                **run_git_log(
                    repository=repository
                )
            )
        )

    @server.tool()
    def workspace_git_diff(
        repository: (
            str | None
        ) = None,
    ) -> GitDiffMCPResult:

        return (
            GitDiffMCPResult(
                **run_git_diff(
                    repository=repository
                )
            )
        )

    @server.tool()
    def workspace_git_changed_files(
        repository: (
            str | None
        ) = None,
    ) -> GitChangedFilesMCPResult:

        return (
            GitChangedFilesMCPResult(
                **run_git_changed_files(
                    repository=repository
                )
            )
        )