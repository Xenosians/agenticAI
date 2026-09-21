from pathlib import (
    Path,
)

import tools.git as git_tools
import tools.git.mcp as git_mcp

from services.git_repositories import (
    GitRepositoryTarget,
)

from services.process_runner import (
    evaluate_process_policy,
)

from tools.git.catalog import (
    GIT_TOOLS,
)


class FakeTrustedProcessRunner:

    def __init__(
        self,
    ) -> None:

        self.calls = []

    def __call__(
        self,
        executable,
        args,
        cwd=None,
        timeout_seconds=60,
    ):

        self.calls.append(
            {
                "executable":
                    executable,

                "args":
                    args,

                "cwd":
                    cwd,

                "timeout_seconds":
                    timeout_seconds,
            }
        )

        if (
            args[
                :2
            ]
            == [
                "add",
                "--",
            ]
        ):

            return {
                "ok":
                    True,

                "status":
                    "success",

                "stdout":
                    "",

                "stderr":
                    "",

                "exit_code":
                    0,
            }

        return {
            "ok":
                True,

            "status":
                "success",

            "stdout":
                (
                    "src/app.py\n"
                    "tests/test_app.py\n"
                ),

            "stderr":
                "",

            "exit_code":
                0,
        }


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

        return (
            decorator
        )


def install_target(
    monkeypatch,
    tmp_path: Path,
):

    root = (
        tmp_path
        / "ai"
    )

    root.mkdir()

    target = (
        GitRepositoryTarget(
            name="ai",
            path=root,
        )
    )

    monkeypatch.setattr(
        git_tools,
        "resolve_git_repository",

        lambda repository=None: (
            target
        ),
    )

    return (
        target
    )


def test_stage_policy_accepts_safe_exact_files(
    monkeypatch,
    tmp_path,
):

    install_target(
        monkeypatch,
        tmp_path,
    )

    result = (
        git_tools
        .evaluate_git_stage_policy(
            repository="ai",

            paths=[
                "src/app.py",
                "tests/test_app.py",
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


def test_stage_policy_rejects_parent_escape(
    monkeypatch,
    tmp_path,
):

    install_target(
        monkeypatch,
        tmp_path,
    )

    result = (
        git_tools
        .evaluate_git_stage_policy(
            repository="ai",

            paths=[
                "../secret.txt",
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


def test_stage_policy_rejects_absolute_path(
    monkeypatch,
    tmp_path,
):

    install_target(
        monkeypatch,
        tmp_path,
    )

    result = (
        git_tools
        .evaluate_git_stage_policy(
            repository="ai",

            paths=[
                "/etc/passwd",
            ],
        )
    )

    assert (
        result[
            "ok"
        ]
        is False
    )


def test_stage_policy_rejects_existing_directory(
    monkeypatch,
    tmp_path,
):

    target = (
        install_target(
            monkeypatch,
            tmp_path,
        )
    )

    (
        target.path
        / "src"
    ).mkdir()

    result = (
        git_tools
        .evaluate_git_stage_policy(
            repository="ai",

            paths=[
                "src",
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
        "directory"
        in result[
            "error"
        ]
    )


def test_stage_files_uses_literal_pathspecs_and_verifies(
    monkeypatch,
    tmp_path,
):

    target = (
        install_target(
            monkeypatch,
            tmp_path,
        )
    )

    runner = (
        FakeTrustedProcessRunner()
    )

    monkeypatch.setattr(
        git_tools,
        "run_trusted_process",
        runner,
    )

    result = (
        git_tools
        .workspace_git_stage_files(
            repository="ai",

            paths=[
                "src/app.py",
                "tests/test_app.py",
            ],
        )
    )

    assert (
        runner.calls[
            0
        ]
        == {
            "executable":
                "git",

            "args": [
                "add",
                "--",
                ":(literal)src/app.py",
                ":(literal)tests/test_app.py",
            ],

            "cwd":
                str(
                    target.path
                ),

            "timeout_seconds":
                10,
        }
    )

    assert (
        runner.calls[
            1
        ][
            "args"
        ]
        == [
            "diff",
            "--cached",
            "--name-only",
            "--",
            ":(literal)src/app.py",
            ":(literal)tests/test_app.py",
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
            "staged_paths"
        ]
        == [
            "src/app.py",
            "tests/test_app.py",
        ]
    )

    assert (
        result[
            "verification_ok"
        ]
        is True
    )


def test_generic_process_policy_still_cannot_git_add():

    result = (
        evaluate_process_policy(
            executable="git",

            args=[
                "add",
                "--",
                "src/app.py",
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


def test_stage_capability_is_approval_gated():

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


def test_stage_files_registers_with_mcp(
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
        result.staged_paths
        == [
            "src/app.py",
        ]
    )