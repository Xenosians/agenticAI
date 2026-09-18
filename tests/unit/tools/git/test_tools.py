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


class FakeTrustedProcessRunner:

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
    *,
    name: str = "frontend",
):

    target = (
        GitRepositoryTarget(
            name=(
                name
            ),

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


def install_runner(
    monkeypatch,
    *,
    stdout: str,
) -> FakeTrustedProcessRunner:

    runner = (
        FakeTrustedProcessRunner(
            stdout=(
                stdout
            )
        )
    )

    monkeypatch.setattr(
        git_tools,
        "run_trusted_process",
        runner,
    )

    return runner


def test_git_status_uses_fixed_trusted_command(
    monkeypatch,
):

    target = (
        install_target(
            monkeypatch
        )
    )

    runner = (
        install_runner(
            monkeypatch,

            stdout=(
                "## main...origin/main\n"
                " M src/app.nim\n"
            ),
        )
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

    assert (
        result[
            "unstaged_files"
        ]
        == [
            "src/app.nim",
        ]
    )

    assert runner.calls == [
        {
            "executable":
                "git",

            "args": [
                "status",
                "--short",
                "--branch",
                "--untracked-files=all",
            ],

            "cwd":
                str(
                    target.path
                ),

            "timeout_seconds":
                10,
        }
    ]


def test_git_status_classifies_working_tree_state(
    monkeypatch,
):

    install_target(
        monkeypatch
    )

    install_runner(
        monkeypatch,

        stdout=(
            "## main...origin/main "
            "[ahead 2, behind 1]\n"
            "M  staged.py\n"
            " M unstaged.py\n"
            "MM both.py\n"
            "?? new.py\n"
            "UU conflict.py\n"
        ),
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
        is False
    )

    assert (
        result[
            "files"
        ]
        == [
            "staged.py",
            "unstaged.py",
            "both.py",
            "new.py",
            "conflict.py",
        ]
    )

    assert (
        result[
            "staged_files"
        ]
        == [
            "staged.py",
            "both.py",
        ]
    )

    assert (
        result[
            "unstaged_files"
        ]
        == [
            "unstaged.py",
            "both.py",
        ]
    )

    assert (
        result[
            "untracked_files"
        ]
        == [
            "new.py",
        ]
    )

    assert (
        result[
            "conflicted_files"
        ]
        == [
            "conflict.py",
        ]
    )

    assert (
        result[
            "staged_count"
        ]
        == 2
    )

    assert (
        result[
            "unstaged_count"
        ]
        == 2
    )

    assert (
        result[
            "untracked_count"
        ]
        == 1
    )

    assert (
        result[
            "conflicted_count"
        ]
        == 1
    )


def test_git_status_parses_new_repository_branch(
    monkeypatch,
):

    install_target(
        monkeypatch
    )

    install_runner(
        monkeypatch,

        stdout=(
            "## No commits yet on main\n"
        ),
    )

    result = (
        git_tools
        .workspace_git_status(
            repository="frontend"
        )
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
        is True
    )


def test_git_branches_uses_fixed_command(
    monkeypatch,
):

    target = (
        install_target(
            monkeypatch,
            name="backend",
        )
    )

    runner = (
        install_runner(
            monkeypatch,

            stdout=(
                "* main\n"
                "  feature/test\n"
            ),
        )
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
        runner.calls[
            0
        ][
            "cwd"
        ]
        == str(
            target.path
        )
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
        install_runner(
            monkeypatch,

            stdout=(
                "abc123 first commit\n"
                "def456 second commit\n"
            ),
        )
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


def test_git_diff_is_explicitly_unstaged(
    monkeypatch,
):

    install_target(
        monkeypatch
    )

    runner = (
        install_runner(
            monkeypatch,

            stdout=(
                "diff --git a/a b/a\n"
                "--- a/a\n"
                "+++ b/a\n"
            ),
        )
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
            "scope"
        ]
        == "unstaged"
    )

    assert (
        result[
            "has_changes"
        ]
        is True
    )


def test_git_changed_files_includes_all_working_tree_classes(
    monkeypatch,
):

    install_target(
        monkeypatch
    )

    runner = (
        install_runner(
            monkeypatch,

            stdout=(
                "M  staged.py\n"
                " M unstaged.py\n"
                "?? untracked.py\n"
                "UU conflicted.py\n"
            ),
        )
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
            "status",
            "--short",
            "--untracked-files=all",
        ]
    )

    assert (
        result[
            "scope"
        ]
        == "working_tree"
    )

    assert (
        result[
            "files"
        ]
        == [
            "staged.py",
            "unstaged.py",
            "untracked.py",
            "conflicted.py",
        ]
    )

    assert (
        result[
            "staged_files"
        ]
        == [
            "staged.py",
        ]
    )

    assert (
        result[
            "unstaged_files"
        ]
        == [
            "unstaged.py",
        ]
    )

    assert (
        result[
            "untracked_files"
        ]
        == [
            "untracked.py",
        ]
    )

    assert (
        result[
            "conflicted_files"
        ]
        == [
            "conflicted.py",
        ]
    )


def test_truncation_marker_is_not_treated_as_file(
    monkeypatch,
):

    install_target(
        monkeypatch
    )

    install_runner(
        monkeypatch,

        stdout=(
            " M src/app.py\n"
            "...[output truncated]\n"
        ),
    )

    result = (
        git_tools
        .workspace_git_changed_files(
            repository="frontend"
        )
    )

    assert (
        result[
            "files"
        ]
        == [
            "src/app.py",
        ]
    )

    assert (
        result[
            "truncated"
        ]
        is True
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
            root=(
                tmp_path
            ),

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
            root=(
                tmp_path
            ),

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
            root=(
                tmp_path
            ),

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
            root=(
                root
            ),

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


# ============================================================
# GENERIC PROCESS POLICY MUST REMAIN NARROW
# ============================================================


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