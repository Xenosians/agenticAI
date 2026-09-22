import tools.git.mcp as git_mcp

from tools.git.catalog import (
    GIT_TOOLS,
)

from tools.git.presentation import (
    build_git_branches_card,
    build_git_changed_files_card,
    build_git_diff_card,
    build_git_log_card,
    build_git_staged_diff_card,
    build_git_status_card,
)


class FakeMCPServer:

    def __init__(
        self,
    ) -> None:

        self.functions = {}

    def tool(
        self,
    ):

        def decorator(
            function,
        ):

            self.functions[
                function.__name__
            ] = (
                function
            )

            return (
                function
            )

        return decorator


def test_git_catalog_exposes_repository_argument():

    for tool_name in (
        "workspace_git_status",
        "workspace_git_branches",
        "workspace_git_log",
        "workspace_git_diff",
        "workspace_git_staged_diff",
        "workspace_git_changed_files",
        "workspace_git_stage_files",
        "workspace_git_unstage_files",
        "workspace_git_create_branch",
        "workspace_git_switch_branch",
    ):

        tool = (
            GIT_TOOLS[
                tool_name
            ]
        )

        assert (
            "repository"
            in tool[
                "parameters"
            ]
        )

        assert (
            "repository"
            in tool[
                "grounded_arguments"
            ]
        )


def test_git_read_capabilities_are_read_only():

    for tool_name in (
        "workspace_git_status",
        "workspace_git_branches",
        "workspace_git_log",
        "workspace_git_diff",
        "workspace_git_staged_diff",
        "workspace_git_changed_files",
    ):

        tool = (
            GIT_TOOLS[
                tool_name
            ]
        )

        assert (
            tool[
                "risk"
            ]
            == "read"
        )

        assert (
            tool[
                "requires_approval"
            ]
            is False
        )


def test_git_stage_files_is_approval_gated():

    tool = (
        GIT_TOOLS[
            "workspace_git_stage_files"
        ]
    )

    assert (
        tool[
            "risk"
        ]
        == "low"
    )

    assert (
        tool[
            "requires_approval"
        ]
        is True
    )

    assert (
        tool[
            "grounded_arguments"
        ]
        == [
            "repository",
            "paths",
        ]
    )


def test_all_git_capabilities_register_with_mcp():

    server = (
        FakeMCPServer()
    )

    git_mcp.register_git_tools(
        server
    )

    assert set(
        server.functions
    ) == set(
        GIT_TOOLS
    )


def test_git_mcp_preserves_rich_working_tree_metadata(
    monkeypatch,
):

    server = (
        FakeMCPServer()
    )

    monkeypatch.setattr(
        git_mcp,
        "run_git_changed_files",

        lambda repository=None: {
            "ok":
                True,

            "status":
                "success",

            "repository":
                repository,

            "scope":
                "working_tree",

            "changes": [
                {
                    "code":
                        "M ",

                    "path":
                        "staged.py",

                    "staged":
                        True,

                    "unstaged":
                        False,

                    "untracked":
                        False,

                    "conflicted":
                        False,
                },

                {
                    "code":
                        "??",

                    "path":
                        "new.py",

                    "staged":
                        False,

                    "unstaged":
                        False,

                    "untracked":
                        True,

                    "conflicted":
                        False,
                },
            ],

            "files": [
                "staged.py",
                "new.py",
            ],

            "count":
                2,

            "staged_files": [
                "staged.py",
            ],

            "staged_count":
                1,

            "unstaged_files":
                [],

            "unstaged_count":
                0,

            "untracked_files": [
                "new.py",
            ],

            "untracked_count":
                1,

            "conflicted_files":
                [],

            "conflicted_count":
                0,

            "truncated":
                False,
        },
    )

    git_mcp.register_git_tools(
        server
    )

    result = (
        server.functions[
            "workspace_git_changed_files"
        ](
            repository="ai"
        )
    )

    assert (
        result.repository
        == "ai"
    )

    assert (
        result.scope
        == "working_tree"
    )

    assert (
        result.staged_files
        == [
            "staged.py",
        ]
    )

    assert (
        result.untracked_files
        == [
            "new.py",
        ]
    )

    assert (
        result.changes[
            0
        ].staged
        is True
    )


def test_git_staged_diff_registers_and_preserves_scope(
    monkeypatch,
):

    server = (
        FakeMCPServer()
    )

    monkeypatch.setattr(
        git_mcp,
        "run_git_staged_diff",

        lambda repository=None: {
            "ok":
                True,

            "status":
                "success",

            "repository":
                repository,

            "scope":
                "staged",

            "diff":
                "diff --git a/a.py b/a.py",

            "has_changes":
                True,

            "truncated":
                False,
        },
    )

    git_mcp.register_git_tools(
        server
    )

    result = (
        server.functions[
            "workspace_git_staged_diff"
        ](
            repository="ai"
        )
    )

    assert (
        result.ok
        is True
    )

    assert (
        result.repository
        == "ai"
    )

    assert (
        result.scope
        == "staged"
    )

    assert (
        result.has_changes
        is True
    )


def test_git_stage_files_registers_and_preserves_paths(
    monkeypatch,
):

    server = (
        FakeMCPServer()
    )

    monkeypatch.setattr(
        git_mcp,
        "run_git_stage_files",

        lambda repository, paths: {
            "ok":
                True,

            "status":
                "success",

            "repository":
                repository,

            "requested_paths":
                paths,

            "files":
                paths,

            "staged_paths":
                paths,

            "staged_count":
                len(
                    paths
                ),

            "verification_ok":
                True,

            "verification_error":
                None,
        },
    )

    git_mcp.register_git_tools(
        server
    )

    result = (
        server.functions[
            "workspace_git_stage_files"
        ](
            repository="ai",

            paths=[
                "src/app.py",
                "tests/test_app.py",
            ],
        )
    )

    assert (
        result.ok
        is True
    )

    assert (
        result.repository
        == "ai"
    )

    assert (
        result.requested_paths
        == [
            "src/app.py",
            "tests/test_app.py",
        ]
    )

    assert (
        result.staged_paths
        == [
            "src/app.py",
            "tests/test_app.py",
        ]
    )


def test_git_status_card():

    card = (
        build_git_status_card(
            {
                "ok":
                    True,

                "status":
                    "success",

                "repository":
                    "frontend",

                "branch":
                    "main",

                "upstream":
                    "origin/main",

                "ahead":
                    0,

                "behind":
                    0,

                "clean":
                    False,

                "count":
                    1,

                "staged_count":
                    0,

                "unstaged_count":
                    1,

                "untracked_count":
                    0,

                "conflicted_count":
                    0,

                "changes": [
                    {
                        "code":
                            " M",

                        "path":
                            "src/app.nim",
                    }
                ],
            }
        )
    )

    assert (
        card[
            "schema"
        ]
        == "result-card.v1"
    )

    assert (
        card[
            "kind"
        ]
        == "git-status"
    )


def test_git_branches_card():

    card = (
        build_git_branches_card(
            {
                "ok":
                    True,

                "status":
                    "success",

                "repository":
                    "backend",

                "current_branch":
                    "main",

                "count":
                    2,

                "branches": [
                    {
                        "name":
                            "main",

                        "current":
                            True,
                    },

                    {
                        "name":
                            "feature/test",

                        "current":
                            False,
                    },
                ],
            }
        )
    )

    assert (
        card[
            "kind"
        ]
        == "git-branches"
    )


def test_git_log_card():

    card = (
        build_git_log_card(
            {
                "ok":
                    True,

                "status":
                    "success",

                "repository":
                    "ai",

                "count":
                    1,

                "commits": [
                    {
                        "hash":
                            "abc123",

                        "message":
                            "example commit",
                    }
                ],
            }
        )
    )

    assert (
        card[
            "kind"
        ]
        == "git-log"
    )


def test_git_diff_card():

    card = (
        build_git_diff_card(
            {
                "ok":
                    True,

                "status":
                    "success",

                "repository":
                    "backend",

                "scope":
                    "unstaged",

                "diff":
                    "diff --git a/a b/a",

                "has_changes":
                    True,

                "truncated":
                    False,
            }
        )
    )

    assert (
        card[
            "kind"
        ]
        == "git-diff"
    )


def test_git_staged_diff_card():

    card = (
        build_git_staged_diff_card(
            {
                "ok":
                    True,

                "status":
                    "success",

                "repository":
                    "ai",

                "scope":
                    "staged",

                "diff":
                    "diff --git a/a b/a",

                "has_changes":
                    True,

                "truncated":
                    False,
            }
        )
    )

    assert (
        card[
            "kind"
        ]
        == "git-staged-diff"
    )


def test_git_changed_files_card():

    card = (
        build_git_changed_files_card(
            {
                "ok":
                    True,

                "status":
                    "success",

                "repository":
                    "frontend",

                "scope":
                    "working_tree",

                "files": [
                    "src/app.nim",
                ],

                "count":
                    1,

                "staged_count":
                    0,

                "unstaged_count":
                    1,

                "untracked_count":
                    0,

                "conflicted_count":
                    0,
            }
        )
    )

    assert (
        card[
            "kind"
        ]
        == "git-files"
    )
