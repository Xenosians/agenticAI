from types import (
    SimpleNamespace,
)

import tools.git as git_tools
import tools.git.mcp as git_mcp

from services.process_runner import (
    evaluate_process_policy,
)

from tools.git.catalog import (
    GIT_TOOLS,
)


OLD_COMMIT = (
    "0123456789abcdef"
    "0123456789abcdef"
    "01234567"
)

NEW_COMMIT = (
    "fedcba9876543210"
    "fedcba9876543210"
    "fedcba98"
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


def test_commit_catalog_is_medium_risk_and_approval_gated():

    tool = (
        GIT_TOOLS[
            "workspace_git_commit"
        ]
    )

    assert (
        tool[
            "risk"
        ]
        == "medium"
    )

    assert (
        tool[
            "requires_approval"
        ]
        is True
    )

    assert (
        tool[
            "policy_owns_preconditions"
        ]
        is True
    )

    assert (
        tool[
            "grounded_arguments"
        ]
        == [
            "repository",
            "commit_message",
        ]
    )


def test_commit_message_is_exact_and_single_line():

    valid, error = (
        git_tools
        ._validate_git_commit_message(
            "feat: governed commit"
        )
    )

    assert (
        valid
        == "feat: governed commit"
    )

    assert (
        error
        is None
    )

    invalid_messages = [
        "",
        " leading",
        "trailing ",
        "line one\nline two",
        "line one\rline two",
        "contains\x00null",
    ]

    for invalid in invalid_messages:

        valid, error = (
            git_tools
            ._validate_git_commit_message(
                invalid
            )
        )

        assert (
            valid
            is None
        )

        assert (
            error
            is not None
        )


def test_commit_policy_denies_when_no_staged_changes(
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
        "_git_mutation_worktree_state",

        lambda **kwargs: (
            [],
            None,
        ),
    )

    monkeypatch.setattr(
        git_tools,
        "_git_operation_in_progress",

        lambda **kwargs: (
            None,
            None,
        ),
    )

    monkeypatch.setattr(
        git_tools,
        "_git_staged_paths",

        lambda **kwargs: (
            [],
            None,
        ),
    )

    def fake_run_git(
        *,
        target,
        args,
        timeout_seconds,
    ):

        if (
            args
            == [
                "rev-parse",
                "--verify",
                "HEAD",
            ]
        ):

            return {
                "ok":
                    True,

                "stdout":
                    f"{OLD_COMMIT}\n",
            }

        raise AssertionError(
            f"Unexpected Git call: {args}"
        )

    monkeypatch.setattr(
        git_tools,
        "_run_git",
        fake_run_git,
    )

    result = (
        git_tools
        .evaluate_git_commit_policy(
            repository="ai",
            commit_message="feat: test",
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
        "staged"
        in result[
            "error"
        ].lower()
    )


def test_commit_policy_denies_conflicted_repository(
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
        "_git_mutation_worktree_state",

        lambda **kwargs: (
            [
                {
                    "code":
                        "UU",

                    "path":
                        "src/app.py",

                    "staged":
                        False,

                    "unstaged":
                        False,

                    "untracked":
                        False,

                    "conflicted":
                        True,
                }
            ],
            None,
        ),
    )

    result = (
        git_tools
        .evaluate_git_commit_policy(
            repository="ai",
            commit_message="feat: test",
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
        ].lower()
    )


def test_commit_policy_denies_operation_in_progress(
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
        "_git_mutation_worktree_state",

        lambda **kwargs: (
            [],
            None,
        ),
    )

    monkeypatch.setattr(
        git_tools,
        "_git_operation_in_progress",

        lambda **kwargs: (
            "rebase",
            None,
        ),
    )

    result = (
        git_tools
        .evaluate_git_commit_policy(
            repository="ai",
            commit_message="feat: test",
        )
    )

    assert (
        result[
            "ok"
        ]
        is False
    )

    assert (
        "rebase"
        in result[
            "error"
        ].lower()
    )


def test_commit_uses_fixed_trusted_command_and_verifies(
    monkeypatch,
):

    monkeypatch.setattr(
        git_tools,
        "_validate_git_commit_request",

        lambda **kwargs: (
            _target(),
            "feat: governed commit",
            OLD_COMMIT,
            [
                "src/app.py",
                "tests/test_app.py",
            ],
            None,
        ),
    )

    calls = []

    responses = [
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
                f"{NEW_COMMIT}\n",
        },

        {
            "ok":
                True,

            "stdout":
                (
                    f"{NEW_COMMIT} "
                    f"{OLD_COMMIT}\n"
                ),
        },

        {
            "ok":
                True,

            "stdout":
                "feat: governed commit\n\n",
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

        if responses:

            return (
                responses.pop(
                    0
                )
            )

        raise AssertionError(
            f"Unexpected Git call: {args}"
        )

    monkeypatch.setattr(
        git_tools,
        "_run_git",
        fake_run_git,
    )

    monkeypatch.setattr(
        git_tools,
        "_git_staged_paths",

        lambda **kwargs: (
            [],
            None,
        ),
    )

    result = (
        git_tools
        .workspace_git_commit(
            repository="ai",
            commit_message="feat: governed commit",
        )
    )

    assert (
        calls[
            0
        ]
        == [
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "--no-gpg-sign",
            "-m",
            "feat: governed commit",
        ]
    )

    assert (
        calls[
            1:
        ]
        == [
            [
                "rev-parse",
                "--verify",
                "HEAD",
            ],

            [
                "rev-list",
                "--parents",
                "-n",
                "1",
                "HEAD",
            ],

            [
                "log",
                "-1",
                "--format=%B",
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
            "status"
        ]
        == "success"
    )

    assert (
        result[
            "previous_commit"
        ]
        == OLD_COMMIT
    )

    assert (
        result[
            "commit"
        ]
        == NEW_COMMIT
    )

    assert (
        result[
            "commit_message"
        ]
        == "feat: governed commit"
    )

    assert (
        result[
            "committed_paths"
        ]
        == [
            "src/app.py",
            "tests/test_app.py",
        ]
    )

    assert (
        result[
            "committed_count"
        ]
        == 2
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


def test_commit_revalidates_after_approval(
    monkeypatch,
):

    monkeypatch.setattr(
        git_tools,
        "_validate_git_commit_request",

        lambda **kwargs: (
            None,
            None,
            None,
            None,
            "staged state changed",
        ),
    )

    result = (
        git_tools
        .workspace_git_commit(
            repository="ai",
            commit_message="feat: governed commit",
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
        result[
            "mutation_performed"
        ]
        is not True
    )


def test_commit_verification_failure_preserves_side_effect_truth(
    monkeypatch,
):

    monkeypatch.setattr(
        git_tools,
        "_validate_git_commit_request",

        lambda **kwargs: (
            _target(),
            "feat: governed commit",
            OLD_COMMIT,
            [
                "src/app.py",
            ],
            None,
        ),
    )

    responses = [
        {
            "ok":
                True,

            "status":
                "success",
        },

        {
            "ok":
                False,

            "status":
                "error",

            "stderr":
                "HEAD verification failed",
        },

        {
            "ok":
                False,

            "status":
                "error",

            "stderr":
                "parent verification failed",
        },

        {
            "ok":
                False,

            "status":
                "error",

            "stderr":
                "message verification failed",
        },
    ]

    monkeypatch.setattr(
        git_tools,
        "_run_git",

        lambda **kwargs: (
            responses.pop(
                0
            )
        ),
    )

    monkeypatch.setattr(
        git_tools,
        "_git_staged_paths",

        lambda **kwargs: (
            None,
            "staged verification failed",
        ),
    )

    result = (
        git_tools
        .workspace_git_commit(
            repository="ai",
            commit_message="feat: governed commit",
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
            "mutation_performed"
        ]
        is True
    )

    assert (
        result[
            "verification_ok"
        ]
        is False
    )

    assert (
        result[
            "verification_error"
        ]
        is not None
    )


def test_generic_process_policy_cannot_git_commit():

    result = (
        evaluate_process_policy(
            executable="git",

            args=[
                "commit",
                "-m",
                "bypass",
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


def test_commit_registers_with_mcp(
    monkeypatch,
):

    server = (
        FakeMCPServer()
    )

    monkeypatch.setattr(
        git_mcp,
        "run_git_commit",

        lambda repository, commit_message: {
            "ok":
                True,

            "status":
                "success",

            "repository":
                repository,

            "previous_commit":
                OLD_COMMIT,

            "commit":
                NEW_COMMIT,

            "commit_message":
                commit_message,

            "committed_paths": [
                "src/app.py",
            ],

            "committed_count":
                1,

            "files": [
                "src/app.py",
            ],

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

    assert (
        "workspace_git_commit"
        in server.functions
    )

    result = (
        server.functions[
            "workspace_git_commit"
        ](
            repository="ai",
            commit_message="feat: governed commit",
        )
    )

    assert (
        result.repository
        == "ai"
    )

    assert (
        result.commit
        == NEW_COMMIT
    )

    assert (
        result.commit_message
        == "feat: governed commit"
    )
