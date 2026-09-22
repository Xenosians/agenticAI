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


class FakeMCPServer:

    def __init__(
        self,
    ):

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


def test_stage_all_catalog_is_approval_gated():

    tool = (
        GIT_TOOLS[
            "workspace_git_stage_all"
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
        ]
    )

    assert (
        tool[
            "policy_owns_preconditions"
        ]
        is True
    )


def test_stage_all_policy_requires_approval_for_dirty_tree(
    monkeypatch,
):

    monkeypatch.setattr(
        git_tools,
        "_resolve_target",

        lambda repository: (
            _target(),
            None,
        ),
    )

    monkeypatch.setattr(
        git_tools,
        "_run_git",

        lambda **kwargs: {
            "ok":
                True,

            "stdout":
                (
                    " M src/app.py\n"
                    "?? tests/new_test.py\n"
                ),
        },
    )

    result = (
        git_tools
        .evaluate_git_stage_all_policy(
            repository="ai"
        )
    )

    assert (
        result[
            "ok"
        ]
        is True
    )

    assert (
        result[
            "requires_approval"
        ]
        is True
    )


def test_stage_all_policy_denies_conflicts(
    monkeypatch,
):

    monkeypatch.setattr(
        git_tools,
        "_resolve_target",

        lambda repository: (
            _target(),
            None,
        ),
    )

    monkeypatch.setattr(
        git_tools,
        "_run_git",

        lambda **kwargs: {
            "ok":
                True,

            "stdout":
                "UU src/app.py\n",
        },
    )

    result = (
        git_tools
        .evaluate_git_stage_all_policy(
            repository="ai"
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

    assert (
        "conflicted"
        in result[
            "error"
        ]
    )


def test_stage_all_uses_fixed_git_add_a_and_verifies(
    monkeypatch,
):

    monkeypatch.setattr(
        git_tools,
        "_resolve_target",

        lambda repository: (
            _target(),
            None,
        ),
    )

    calls = []

    responses = [
        {
            "ok":
                True,

            "stdout":
                (
                    " M src/app.py\n"
                    "?? tests/new_test.py\n"
                ),
        },

        {
            "ok":
                True,

            "status":
                "success",

            "stdout":
                "",
        },

        {
            "ok":
                True,

            "stdout":
                (
                    "M  src/app.py\n"
                    "A  tests/new_test.py\n"
                ),
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

    result = (
        git_tools
        .workspace_git_stage_all(
            repository="ai"
        )
    )

    assert (
        calls
        == [
            [
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],

            [
                "add",
                "-A",
            ],

            [
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
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
            "staged_paths"
        ]
        == [
            "src/app.py",
            "tests/new_test.py",
        ]
    )


def test_stage_all_already_staged_is_truthful_noop(
    monkeypatch,
):

    monkeypatch.setattr(
        git_tools,
        "_resolve_target",

        lambda repository: (
            _target(),
            None,
        ),
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

            "stdout":
                "M  src/app.py\n",
        }

    monkeypatch.setattr(
        git_tools,
        "_run_git",
        fake_run_git,
    )

    result = (
        git_tools
        .workspace_git_stage_all(
            repository="ai"
        )
    )

    assert (
        calls
        == [
            [
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
        ]
    )

    assert (
        result[
            "mutation_performed"
        ]
        is False
    )

    assert (
        result[
            "verification_ok"
        ]
        is True
    )


def test_stage_all_registers_with_mcp(
    monkeypatch,
):

    server = (
        FakeMCPServer()
    )

    monkeypatch.setattr(
        git_mcp,
        "run_git_stage_all",

        lambda repository: {
            "ok":
                True,

            "status":
                "success",

            "repository":
                repository,

            "files": [
                "src/app.py",
            ],

            "staged_paths": [
                "src/app.py",
            ],

            "staged_count":
                1,

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
            "workspace_git_stage_all"
        ](
            repository="ai"
        )
    )

    assert (
        result.repository
        == "ai"
    )

    assert (
        result.staged_count
        == 1
    )
