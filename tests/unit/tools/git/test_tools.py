from pathlib import (
    Path,
)

from services.git_repositories import (
    GitRepositoryRegistry,
    GitRepositoryTarget,
)

from services.process_runner import (
    evaluate_process_policy,
)

import tools.git as git_tools


class FakeProcessRunner:

    def __init__(
        self,
        *,
        stdout: str = "",
    ) -> None:

        self.calls = []

        self.stdout = (
            stdout
        )

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
    *,
    name: str = "frontend",
):
    target = (
        GitRepositoryTarget(
            name=name,

            path=(
                Path(
                    "/approved"
                )
                / name
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

    return target


def test_git_status_uses_fixed_command(
    monkeypatch,
):
    target = (
        install_target(
            monkeypatch
        )
    )

    runner = (
        FakeProcessRunner(
            stdout=(
                "## main...origin/main\n"
                " M src/app.nim\n"
            )
        )
    )

    monkeypatch.setattr(
        git_tools,
        "run_process",
        runner,
    )

    result = (
        git_tools
        .workspace_git_status(
            repository="frontend"
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
            "repository"
        ]
        == "frontend"
    )

    assert (
        result[
            "branch"
        ]
        == "main"
    )

    assert (
        result[
            "clean"
        ]
        is False
    )

    assert (
        result[
            "change_count"
        ]
        == 1
    )

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
                str(
                    target.path
                ),

            "timeout_seconds":
                10,
        }
    ]


def test_git_status_parses_ahead_and_behind(
    monkeypatch,
):
    install_target(
        monkeypatch
    )

    runner = (
        FakeProcessRunner(
            stdout=(
                "## main...origin/main "
                "[ahead 2, behind 1]\n"
            )
        )
    )

    monkeypatch.setattr(
        git_tools,
        "run_process",
        runner,
    )

    result = (
        git_tools
        .workspace_git_status(
            repository="frontend"
        )
    )

    assert (
        result[
            "ahead"
        ]
        == 2
    )

    assert (
        result[
            "behind"
        ]
        == 1
    )

    assert (
        result[
            "clean"
        ]
        is True
    )


def test_git_branches_uses_fixed_command(
    monkeypatch,
):
    install_target(
        monkeypatch,
        name="backend",
    )

    runner = (
        FakeProcessRunner(
            stdout=(
                "* main\n"
                "  feature/test\n"
            )
        )
    )

    monkeypatch.setattr(
        git_tools,
        "run_process",
        runner,
    )

    result = (
        git_tools
        .workspace_git_branches(
            repository="backend"
        )
    )

    assert (
        runner.calls[
            0
        ][
            "args"
        ]
        == [
            "branch",
            "--list",
            "--no-color",
        ]
    )

    assert (
        result[
            "current_branch"
        ]
        == "main"
    )

    assert (
        result[
            "count"
        ]
        == 2
    )


def test_git_log_uses_bounded_history(
    monkeypatch,
):
    install_target(
        monkeypatch,
        name="ai",
    )

    runner = (
        FakeProcessRunner(
            stdout=(
                "abc123 first commit\n"
                "def456 second commit\n"
            )
        )
    )

    monkeypatch.setattr(
        git_tools,
        "run_process",
        runner,
    )

    result = (
        git_tools
        .workspace_git_log(
            repository="ai"
        )
    )

    assert (
        runner.calls[
            0
        ][
            "args"
        ]
        == [
            "log",
            "--oneline",
            "--no-decorate",
            "-n",
            "20",
        ]
    )

    assert (
        result[
            "commits"
        ][
            0
        ][
            "hash"
        ]
        == "abc123"
    )

    assert (
        result[
            "count"
        ]
        == 2
    )


def test_git_diff_disables_external_diff(
    monkeypatch,
):
    install_target(
        monkeypatch
    )

    runner = (
        FakeProcessRunner(
            stdout=(
                "diff --git a/a b/a\n"
                "--- a/a\n"
                "+++ b/a\n"
            )
        )
    )

    monkeypatch.setattr(
        git_tools,
        "run_process",
        runner,
    )

    result = (
        git_tools
        .workspace_git_diff(
            repository="frontend"
        )
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
        result[
            "has_changes"
        ]
        is True
    )


def test_git_changed_files_uses_name_only(
    monkeypatch,
):
    install_target(
        monkeypatch
    )

    runner = (
        FakeProcessRunner(
            stdout=(
                "src/app.nim\n"
                "public/index.html\n"
            )
        )
    )

    monkeypatch.setattr(
        git_tools,
        "run_process",
        runner,
    )

    result = (
        git_tools
        .workspace_git_changed_files(
            repository="frontend"
        )
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
            "--name-only",
        ]
    )

    assert (
        result[
            "files"
        ]
        == [
            "src/app.nim",
            "public/index.html",
        ]
    )


def test_repository_registry_resolves_logical_alias(
    tmp_path,
):
    frontend = (
        tmp_path
        / "frontend"
    )

    frontend.mkdir()

    (
        frontend
        / ".git"
    ).mkdir()

    registry = (
        GitRepositoryRegistry(
            root=tmp_path,

            repositories={
                "frontend":
                    Path(
                        "frontend"
                    ),
            },

            default_repository=(
                "frontend"
            ),
        )
    )

    target = (
        registry.resolve(
            "frontend"
        )
    )

    assert (
        target.name
        == "frontend"
    )

    assert (
        target.path
        == frontend.resolve()
    )


def test_repository_registry_uses_default(
    tmp_path,
):
    ai = (
        tmp_path
        / "ai"
    )

    ai.mkdir()

    (
        ai
        / ".git"
    ).mkdir()

    registry = (
        GitRepositoryRegistry(
            root=tmp_path,

            repositories={
                "ai":
                    Path(
                        "ai"
                    ),
            },

            default_repository="ai",
        )
    )

    assert (
        registry.resolve().name
        == "ai"
    )


def test_repository_registry_rejects_unknown_alias(
    tmp_path,
):
    ai = (
        tmp_path
        / "ai"
    )

    ai.mkdir()

    (
        ai
        / ".git"
    ).mkdir()

    registry = (
        GitRepositoryRegistry(
            root=tmp_path,

            repositories={
                "ai":
                    Path(
                        "ai"
                    ),
            },

            default_repository="ai",
        )
    )

    try:
        registry.resolve(
            "production"
        )

    except ValueError as exc:
        assert (
            "Unknown Git repository"
            in str(
                exc
            )
        )

    else:
        raise AssertionError(
            "Unknown repository "
            "should have been denied."
        )


def test_repository_registry_rejects_escape(
    tmp_path,
):
    root = (
        tmp_path
        / "workspace"
    )

    root.mkdir()

    outside = (
        tmp_path
        / "outside"
    )

    outside.mkdir()

    (
        outside
        / ".git"
    ).mkdir()

    registry = (
        GitRepositoryRegistry(
            root=root,

            repositories={
                "escape":
                    Path(
                        "../outside"
                    ),
            },

            default_repository="escape",
        )
    )

    try:
        registry.resolve(
            "escape"
        )

    except ValueError as exc:
        assert (
            "escapes the approved workspace"
            in str(
                exc
            )
        )

    else:
        raise AssertionError(
            "Escaping repository path "
            "should have been denied."
        )


def test_process_policy_rejects_git_push():
    result = (
        evaluate_process_policy(
            executable="git",

            args=[
                "push",
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