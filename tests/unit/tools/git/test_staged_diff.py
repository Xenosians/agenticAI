from pathlib import (
    Path,
)

import tools.git as git_tools

from services.git_repositories import (
    GitRepositoryTarget,
)

from subagents.core.tooling.capabilities import (
    build_capability_spec,
)


class FakeTrustedProcessRunner:

    def __init__(
        self,
        *,
        stdout: str = "",
    ) -> None:

        self.stdout = (
            stdout
        )

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

        return {
            "ok":
                True,

            "status":
                "success",

            "executable":
                executable,

            "args":
                args,

            "cwd":
                cwd,

            "exit_code":
                0,

            "stdout":
                self.stdout,

            "stderr":
                "",

            "timed_out":
                False,

            "duration_ms":
                1,
        }


def install_target(
    monkeypatch,
):

    target = (
        GitRepositoryTarget(
            name="ai",

            path=(
                Path(
                    "/approved/ai"
                )
            ),
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


def test_staged_diff_uses_fixed_cached_command(
    monkeypatch,
):

    target = (
        install_target(
            monkeypatch
        )
    )

    runner = (
        FakeTrustedProcessRunner(
            stdout=(
                "diff --git a/example.py b/example.py\n"
                "--- a/example.py\n"
                "+++ b/example.py\n"
            )
        )
    )

    monkeypatch.setattr(
        git_tools,
        "run_trusted_process",
        runner,
    )

    result = (
        git_tools
        .workspace_git_staged_diff(
            repository="ai"
        )
    )

    assert (
        runner.calls
        == [
            {
                "executable":
                    "git",

                "args": [
                    "diff",
                    "--cached",
                    "--no-ext-diff",
                    "--no-color",
                    "--unified=3",
                ],

                "cwd":
                    str(
                        target.path
                    ),

                "timeout_seconds":
                    10,
            }
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
            "scope"
        ]
        == "staged"
    )

    assert (
        result[
            "has_changes"
        ]
        is True
    )

    assert (
        "example.py"
        in result[
            "diff"
        ]
    )


def test_staged_diff_reports_empty_index(
    monkeypatch,
):

    install_target(
        monkeypatch
    )

    monkeypatch.setattr(
        git_tools,
        "run_trusted_process",

        FakeTrustedProcessRunner(
            stdout=""
        ),
    )

    result = (
        git_tools
        .workspace_git_staged_diff(
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
            "scope"
        ]
        == "staged"
    )

    assert (
        result[
            "has_changes"
        ]
        is False
    )

    assert (
        result[
            "diff"
        ]
        == ""
    )


def test_unstaged_and_staged_commands_are_distinct(
    monkeypatch,
):

    install_target(
        monkeypatch
    )

    runner = (
        FakeTrustedProcessRunner(
            stdout=""
        )
    )

    monkeypatch.setattr(
        git_tools,
        "run_trusted_process",
        runner,
    )

    git_tools.workspace_git_diff(
        repository="ai"
    )

    git_tools.workspace_git_staged_diff(
        repository="ai"
    )

    assert (
        runner.calls[
            0
        ][
            "args"
        ]
        == [
            "diff",
            "--no-ext-diff",
            "--no-color",
            "--unified=3",
        ]
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
            "--no-ext-diff",
            "--no-color",
            "--unified=3",
        ]
    )


def test_staged_diff_exposes_bounded_repository_values():

    spec = (
        build_capability_spec(
            "workspace_git_staged_diff"
        )
    )

    repository_schema = (
        spec[
            "argument_schema"
        ][
            "repository"
        ]
    )

    assert (
        repository_schema[
            "enum"
        ]
        == [
            "ai",
            "backend",
            "frontend",
        ]
    )