from services.process_runner import (
    evaluate_process_policy,
)

import tools.git as git_tools


class FakeProcessRunner:
    def __init__(
        self,
    ) -> None:
        self.calls = []

    def __call__(
        self,
        *,
        executable,
        args,
        cwd=None,
        timeout_seconds=10,
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
            "ok": True,
            "status": "success",

            "executable":
                executable,

            "args":
                args,

            "cwd":
                cwd,

            "exit_code":
                0,

            "stdout":
                "",

            "stderr":
                "",

            "timed_out":
                False,

            "duration_ms":
                1,
        }


def test_git_status_uses_fixed_command(
    monkeypatch,
):
    runner = (
        FakeProcessRunner()
    )

    monkeypatch.setattr(
        git_tools,
        "run_process",
        runner,
    )

    result = (
        git_tools
        .workspace_git_status()
    )

    assert result[
        "ok"
    ] is True

    assert runner.calls == [
        {
            "executable":
                "git",

            "args": [
                "status",
                "--short",
                "--branch",
            ],

            "cwd":
                None,

            "timeout_seconds":
                10,
        }
    ]


def test_git_branches_uses_fixed_command(
    monkeypatch,
):
    runner = (
        FakeProcessRunner()
    )

    monkeypatch.setattr(
        git_tools,
        "run_process",
        runner,
    )

    git_tools.workspace_git_branches()

    assert runner.calls[
        0
    ][
        "args"
    ] == [
        "branch",
        "--list",
        "--no-color",
    ]


def test_git_log_uses_bounded_history(
    monkeypatch,
):
    runner = (
        FakeProcessRunner()
    )

    monkeypatch.setattr(
        git_tools,
        "run_process",
        runner,
    )

    git_tools.workspace_git_log()

    assert runner.calls[
        0
    ][
        "args"
    ] == [
        "log",
        "--oneline",
        "--no-decorate",
        "-n",
        "20",
    ]


def test_git_diff_disables_external_diff(
    monkeypatch,
):
    runner = (
        FakeProcessRunner()
    )

    monkeypatch.setattr(
        git_tools,
        "run_process",
        runner,
    )

    git_tools.workspace_git_diff()

    assert runner.calls[
        0
    ][
        "args"
    ] == [
        "diff",
        "--no-ext-diff",
        "--no-color",
        "--unified=3",
    ]


def test_git_changed_files_uses_name_only(
    monkeypatch,
):
    runner = (
        FakeProcessRunner()
    )

    monkeypatch.setattr(
        git_tools,
        "run_process",
        runner,
    )

    git_tools.workspace_git_changed_files()

    assert runner.calls[
        0
    ][
        "args"
    ] == [
        "diff",
        "--no-ext-diff",
        "--name-only",
    ]


def test_process_policy_rejects_git_push():
    result = (
        evaluate_process_policy(
            executable="git",
            args=[
                "push",
            ],
        )
    )

    assert result[
        "ok"
    ] is False

    assert result[
        "status"
    ] == "denied"


def test_process_policy_rejects_arbitrary_git_config():
    result = (
        evaluate_process_policy(
            executable="git",
            args=[
                "-c",
                "core.pager=cat",
                "status",
            ],
        )
    )

    assert result[
        "ok"
    ] is False

    assert result[
        "status"
    ] == "denied"


def test_process_policy_rejects_git_shell_alias():
    result = (
        evaluate_process_policy(
            executable="git",
            args=[
                "-c",
                "alias.escape=!bash",
                "escape",
            ],
        )
    )

    assert result[
        "ok"
    ] is False

    assert result[
        "status"
    ] == "denied"


def test_process_policy_rejects_unapproved_revision():
    result = (
        evaluate_process_policy(
            executable="git",
            args=[
                "show",
                "HEAD",
            ],
        )
    )

    assert result[
        "ok"
    ] is False

    assert result[
        "status"
    ] == "denied"