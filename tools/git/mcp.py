from pydantic import (
    BaseModel,
    Field,
)

from mcp.server import (
    MCPServer,
)

from tools.git import (
    workspace_git_branches
    as run_git_branches,

    workspace_git_changed_files
    as run_git_changed_files,

    workspace_git_create_branch
    as run_git_create_branch,

    workspace_git_diff
    as run_git_diff,

    workspace_git_log
    as run_git_log,

    workspace_git_stage_files
    as run_git_stage_files,

    workspace_git_staged_diff
    as run_git_staged_diff,

    workspace_git_status
    as run_git_status,

    workspace_git_switch_branch
    as run_git_switch_branch,

    workspace_git_unstage_files
    as run_git_unstage_files,
)


class GitStatusChange(
    BaseModel
):
    code: str
    path: str

    staged: bool = False
    unstaged: bool = False
    untracked: bool = False
    conflicted: bool = False


class GitWorkingTreeResult(
    BaseModel
):
    ok: bool
    status: str

    repository: (
        str | None
    ) = None

    scope: (
        str | None
    ) = None

    changes: list[
        GitStatusChange
    ] = Field(
        default_factory=list
    )

    files: list[
        str
    ] = Field(
        default_factory=list
    )

    count: int = 0

    staged_files: list[
        str
    ] = Field(
        default_factory=list
    )

    staged_count: int = 0

    unstaged_files: list[
        str
    ] = Field(
        default_factory=list
    )

    unstaged_count: int = 0

    untracked_files: list[
        str
    ] = Field(
        default_factory=list
    )

    untracked_count: int = 0

    conflicted_files: list[
        str
    ] = Field(
        default_factory=list
    )

    conflicted_count: int = 0

    truncated: bool = False

    error: (
        str | None
    ) = None


class GitStatusMCPResult(
    GitWorkingTreeResult
):
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

    change_count: int = 0


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
    truncated: bool = False

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
    truncated: bool = False

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

    scope: (
        str | None
    ) = None

    diff: str = ""

    has_changes: bool = False
    truncated: bool = False

    error: (
        str | None
    ) = None


class GitChangedFilesMCPResult(
    GitWorkingTreeResult
):
    pass


class GitStageFilesMCPResult(
    BaseModel
):
    ok: bool
    status: str

    repository: (
        str | None
    ) = None

    requested_paths: list[
        str
    ] = Field(
        default_factory=list
    )

    files: list[
        str
    ] = Field(
        default_factory=list
    )

    staged_paths: list[
        str
    ] = Field(
        default_factory=list
    )

    staged_count: int = 0

    verification_ok: (
        bool | None
    ) = None

    verification_error: (
        str | None
    ) = None

    error: (
        str | None
    ) = None


class GitCreateBranchMCPResult(
    BaseModel
):
    ok: bool
    status: str

    repository: (
        str | None
    ) = None

    requested_branch: (
        str | None
    ) = None

    created_branch: (
        str | None
    ) = None

    created_from: (
        str | None
    ) = None

    created_commit: (
        str | None
    ) = None

    files: list[
        str
    ] = Field(
        default_factory=list
    )

    verification_ok: (
        bool | None
    ) = None

    verification_error: (
        str | None
    ) = None

    error: (
        str | None
    ) = None


class GitUnstageFilesMCPResult(
    BaseModel
):
    ok: bool
    status: str

    repository: (
        str | None
    ) = None

    requested_paths: list[
        str
    ] = Field(
        default_factory=list
    )

    files: list[
        str
    ] = Field(
        default_factory=list
    )

    staged_before_paths: list[
        str
    ] = Field(
        default_factory=list
    )

    unstaged_paths: list[
        str
    ] = Field(
        default_factory=list
    )

    unstaged_count: int = 0

    remaining_staged_paths: list[
        str
    ] = Field(
        default_factory=list
    )

    mutation_performed: bool = False

    verification_ok: (
        bool | None
    ) = None

    verification_error: (
        str | None
    ) = None

    error: (
        str | None
    ) = None


class GitSwitchBranchMCPResult(
    BaseModel
):
    ok: bool
    status: str

    repository: (
        str | None
    ) = None

    requested_branch: (
        str | None
    ) = None

    previous_branch: (
        str | None
    ) = None

    current_branch: (
        str | None
    ) = None

    switched_branch: (
        str | None
    ) = None

    switched_to_commit: (
        str | None
    ) = None

    mutation_performed: bool = False

    files: list[
        str
    ] = Field(
        default_factory=list
    )

    verification_ok: (
        bool | None
    ) = None

    verification_error: (
        str | None
    ) = None

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
    def workspace_git_staged_diff(
        repository: (
            str | None
        ) = None,
    ) -> GitDiffMCPResult:

        return (
            GitDiffMCPResult(
                **run_git_staged_diff(
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

    @server.tool()
    def workspace_git_stage_files(
        repository: str,
        paths: list[str],
    ) -> GitStageFilesMCPResult:

        return (
            GitStageFilesMCPResult(
                **run_git_stage_files(
                    repository=repository,
                    paths=paths,
                )
            )
        )

    @server.tool()
    def workspace_git_unstage_files(
        repository: str,
        paths: list[str],
    ) -> GitUnstageFilesMCPResult:

        return (
            GitUnstageFilesMCPResult(
                **run_git_unstage_files(
                    repository=repository,
                    paths=paths,
                )
            )
        )

    @server.tool()
    def workspace_git_create_branch(
        repository: str,
        branch_name: str,
    ) -> GitCreateBranchMCPResult:

        return (
            GitCreateBranchMCPResult(
                **run_git_create_branch(
                    repository=repository,
                    branch_name=branch_name,
                )
            )
        )

    @server.tool()
    def workspace_git_switch_branch(
        repository: str,
        branch_name: str,
    ) -> GitSwitchBranchMCPResult:

        return (
            GitSwitchBranchMCPResult(
                **run_git_switch_branch(
                    repository=repository,
                    branch_name=branch_name,
                )
            )
        )
