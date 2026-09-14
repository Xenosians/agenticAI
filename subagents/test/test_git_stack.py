from pathlib import (
    Path,
)

import tools.git_mcp as git_mcp

from tools.git_catalog import (
    GIT_TOOLS,
)

from tools.git_presentation import (
    build_git_branches_card,
    build_git_changed_files_card,
    build_git_diff_card,
    build_git_log_card,
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
            ] = function

            return function

        return decorator


def test_git_catalog_exposes_repository_argument():
    for tool_name in (
        "workspace_git_status",
        "workspace_git_branches",
        "workspace_git_log",
        "workspace_git_diff",
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

        assert (
            "repository"
            in tool[
                "parameters"
            ]
        )

        assert (
            tool[
                "grounded_arguments"
            ]
            == [
                "repository"
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
    ) == {
        "workspace_git_status",
        "workspace_git_branches",
        "workspace_git_log",
        "workspace_git_diff",
        "workspace_git_changed_files",
    }


def test_git_mcp_preserves_repository(
    monkeypatch,
):
    server = (
        FakeMCPServer()
    )

    monkeypatch.setattr(
        git_mcp,
        "run_git_status",

        lambda repository=None: {
            "ok":
                True,

            "status":
                "success",

            "repository":
                repository,

            "branch":
                "main",

            "upstream":
                "origin/main",

            "ahead":
                0,

            "behind":
                0,

            "clean":
                True,

            "changes":
                [],

            "change_count":
                0,
        },
    )

    git_mcp.register_git_tools(
        server
    )

    result = (
        server.functions[
            "workspace_git_status"
        ](
            repository="frontend"
        )
    )

    assert (
        result.ok
        is True
    )

    assert (
        result.repository
        == "frontend"
    )

    assert (
        result.branch
        == "main"
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

                "change_count":
                    1,

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

    assert (
        card[
            "title"
        ]
        == "frontend Git status"
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

                "files": [
                    "src/app.nim",
                ],

                "count":
                    1,
            }
        )
    )

    assert (
        card[
            "kind"
        ]
        == "git-files"
    )