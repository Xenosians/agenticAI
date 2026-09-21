from types import (
    SimpleNamespace,
)

import tools.git as git_tools
import tools.git.mcp as git_mcp

from tools.git.catalog import (
    GIT_TOOLS,
)


def _target():

    return (
        SimpleNamespace(
            name="ai"
        )
    )


def _valid_request(
    *,
    repository,
    paths,
):

    return (
        _target(),
        list(
            paths
        ),
        None,
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


def test_unstage_catalog_is_approval_gated():

    tool = (
        GIT_TOOLS[
            "workspace_git_unstage_files"
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


def test_unstage_policy_requires_approval(
    monkeypatch,
):

    monkeypatch.setattr(
        git_tools,
        "_validate_unstage_request",
        _valid_request,
    )

    result = (
        git_tools
        .evaluate_git_unstage_policy(
            repository="ai",

            paths=[
                "tools/git/catalog.py",
            ],
        )
    )

    assert (
        result
        == {
            "ok":
                True,

            "status":
                "allowed",

            "risk":
                "low",

            "requires_approval":
                True,
        }
    )


def test_unstage_uses_exact_index_only_reset_and_verifies(
    monkeypatch,
):

    monkeypatch.setattr(
        git_tools,
        "_validate_unstage_request",
        _valid_request,
    )

    calls = []

    responses = [
        {
            "ok":
                True,

            "lines": [
                "tools/git/catalog.py",
            ],
        },

        {
            "ok":
                True,
        },

        {
            "ok":
                True,

            "lines":
                [],
        },
    ]

    def fake_run_git(
        *,
        target,
        args,
        timeout_seconds,
    ):

        calls.append(
            list(
                args
            )
        )

        return (
            responses.pop(
                0
            )
        )

    monkeypatch.setattr(
        git_tools,
        "_run_git",
        fake_run_git,
    )

    monkeypatch.setattr(
        git_tools,
        "_stdout_lines",

        lambda result: (
            list(
                result.get(
                    "lines",
                    [],
                )
            )
        ),
    )

    result = (
        git_tools
        .workspace_git_unstage_files(
            repository="ai",

            paths=[
                "tools/git/catalog.py",
            ],
        )
    )

    literal = (
        ":(literal)tools/git/catalog.py"
    )

    assert (
        calls
        == [
            [
                "diff",
                "--cached",
                "--name-only",
                "--",
                literal,
            ],

            [
                "reset",
                "--",
                literal,
            ],

            [
                "diff",
                "--cached",
                "--name-only",
                "--",
                literal,
            ],
        ]
    )

    assert (
        result[
            "ok"
        ]
        is True
    )

    assert (
        result[
            "mutation_performed"
        ]
        is True
    )

    assert (
        result[
            "verification_ok"
        ]
        is True
    )

    assert (
        result[
            "unstaged_paths"
        ]
        == [
            "tools/git/catalog.py",
        ]
    )

    assert (
        result[
            "remaining_staged_paths"
        ]
        == []
    )


def test_unstage_does_not_mutate_when_requested_path_is_not_staged(
    monkeypatch,
):

    monkeypatch.setattr(
        git_tools,
        "_validate_unstage_request",
        _valid_request,
    )

    calls = []

    def fake_run_git(
        *,
        target,
        args,
        timeout_seconds,
    ):

        calls.append(
            list(
                args
            )
        )

        return {
            "ok":
                True,

            "lines":
                [],
        }

    monkeypatch.setattr(
        git_tools,
        "_run_git",
        fake_run_git,
    )

    monkeypatch.setattr(
        git_tools,
        "_stdout_lines",

        lambda result: (
            list(
                result.get(
                    "lines",
                    [],
                )
            )
        ),
    )

    result = (
        git_tools
        .workspace_git_unstage_files(
            repository="ai",

            paths=[
                "already-unstaged.py",
            ],
        )
    )

    assert (
        len(
            calls
        )
        == 1
    )

    assert (
        result[
            "ok"
        ]
        is True
    )

    assert (
        result[
            "mutation_performed"
        ]
        is False
    )

    assert (
        result[
            "unstaged_count"
        ]
        == 0
    )


def test_unstage_revalidates_after_approval(
    monkeypatch,
):

    monkeypatch.setattr(
        git_tools,
        "_validate_unstage_request",

        lambda **kwargs: (
            None,
            None,
            "unsafe path",
        ),
    )

    result = (
        git_tools
        .workspace_git_unstage_files(
            repository="ai",

            paths=[
                "../outside.py",
            ],
        )
    )

    assert (
        result[
            "ok"
        ]
        is False
    )

    assert (
        result[
            "status"
        ]
        == "denied"
    )


def test_unstage_mcp_preserves_exact_paths(
    monkeypatch,
):

    server = (
        FakeMCPServer()
    )

    monkeypatch.setattr(
        git_mcp,
        "run_git_unstage_files",

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

            "staged_before_paths":
                paths,

            "unstaged_paths":
                paths,

            "unstaged_count":
                len(
                    paths
                ),

            "remaining_staged_paths":
                [],

            "mutation_performed":
                True,

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
            "workspace_git_unstage_files"
        ](
            repository="ai",

            paths=[
                "tools/git/catalog.py",
            ],
        )
    )

    assert (
        result.repository
        == "ai"
    )

    assert (
        result.unstaged_paths
        == [
            "tools/git/catalog.py",
        ]
    )
