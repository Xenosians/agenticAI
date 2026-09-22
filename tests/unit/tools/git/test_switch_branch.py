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


COMMIT = (
    "0123456789abcdef"
    "0123456789abcdef"
    "01234567"
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


def _valid_switch(
    *,
    repository,
    branch_name,
    timeout_seconds=10,
):

    return (
        _target(),
        branch_name,
        "main",
        None,
    )


def test_switch_branch_catalog_is_approval_gated():

    tool = (
        GIT_TOOLS[
            "workspace_git_switch_branch"
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
            "branch_name",
        ]
    )


def test_switch_branch_preflight_requires_existing_local_branch_and_clean_tree(
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
                "test/agentic-git-smoke\n",
        },

        {
            "ok":
                True,

            "exit_code":
                0,
        },

        {
            "ok":
                True,

            "stdout":
                "main\n",
        },

        {
            "ok":
                True,

            "stdout":
                "",
        },
    ]

    def fake_run_git(
        *,
        target,
        args,
        timeout_seconds,
    ):

        calls.append(
            list(args)
        )

        return (
            responses.pop(0)
        )

    monkeypatch.setattr(
        git_tools,
        "_run_git",
        fake_run_git,
    )

    result = (
        git_tools
        .evaluate_git_switch_branch_policy(
            repository="ai",
            branch_name="test/agentic-git-smoke",
        )
    )

    assert result["ok"] is True
    assert result["requires_approval"] is True

    assert (
        calls
        == [
            [
                "check-ref-format",
                "--branch",
                "test/agentic-git-smoke",
            ],

            [
                "show-ref",
                "--verify",
                "--quiet",
                "refs/heads/test/agentic-git-smoke",
            ],

            [
                "symbolic-ref",
                "--quiet",
                "--short",
                "HEAD",
            ],

            [
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
        ]
    )


def test_switch_branch_rejects_dirty_worktree(
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

    responses = [
        {
            "ok":
                True,

            "stdout":
                "test/agentic-git-smoke\n",
        },

        {
            "ok":
                True,

            "exit_code":
                0,
        },

        {
            "ok":
                True,

            "stdout":
                "main\n",
        },

        {
            "ok":
                True,

            "stdout":
                " M tools/git/catalog.py\n",
        },
    ]

    monkeypatch.setattr(
        git_tools,
        "_run_git",

        lambda **kwargs: (
            responses.pop(0)
        ),
    )

    result = (
        git_tools
        .evaluate_git_switch_branch_policy(
            repository="ai",
            branch_name="test/agentic-git-smoke",
        )
    )

    assert result["ok"] is False
    assert "completely clean" in result["error"]


def test_switch_branch_rejects_missing_local_branch_without_remote_guessing(
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

    responses = [
        {
            "ok":
                True,

            "stdout":
                "feature/missing\n",
        },

        {
            "ok":
                False,

            "exit_code":
                1,
        },
    ]

    monkeypatch.setattr(
        git_tools,
        "_run_git",

        lambda **kwargs: (
            responses.pop(0)
        ),
    )

    result = (
        git_tools
        .evaluate_git_switch_branch_policy(
            repository="ai",
            branch_name="feature/missing",
        )
    )

    assert result["ok"] is False
    assert "does not exist" in result["error"]


def test_switch_branch_uses_no_guess_and_verifies(
    monkeypatch,
):

    monkeypatch.setattr(
        git_tools,
        "_validate_switch_branch_request",
        _valid_switch,
    )

    calls = []

    responses = [
        {
            "ok":
                True,

            "status":
                "success",
        },

        {
            "ok":
                True,

            "stdout":
                "test/agentic-git-smoke\n",
        },

        {
            "ok":
                True,

            "stdout":
                f"{COMMIT}\n",
        },

        {
            "ok":
                True,

            "stdout":
                f"{COMMIT}\n",
        },

        {
            "ok":
                True,

            "stdout":
                "",
        },
    ]

    def fake_run_git(
        *,
        target,
        args,
        timeout_seconds,
    ):

        calls.append(
            list(args)
        )

        return (
            responses.pop(0)
        )

    monkeypatch.setattr(
        git_tools,
        "_run_git",
        fake_run_git,
    )

    result = (
        git_tools
        .workspace_git_switch_branch(
            repository="ai",
            branch_name="test/agentic-git-smoke",
        )
    )

    assert (
        calls[0]
        == [
            "switch",
            "--no-guess",
            "test/agentic-git-smoke",
        ]
    )

    assert result["ok"] is True

    assert (
        result[
            "current_branch"
        ]
        == "test/agentic-git-smoke"
    )

    assert (
        result[
            "previous_branch"
        ]
        == "main"
    )

    assert (
        result[
            "switched_to_commit"
        ]
        == COMMIT
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


def test_switch_branch_same_branch_is_truthful_noop(
    monkeypatch,
):

    monkeypatch.setattr(
        git_tools,
        "_validate_switch_branch_request",

        lambda **kwargs: (
            _target(),
            "main",
            "main",
            None,
        ),
    )

    called = False

    def should_not_run(
        **kwargs,
    ):

        nonlocal called

        called = True

        raise AssertionError(
            "No Git mutation should run."
        )

    monkeypatch.setattr(
        git_tools,
        "_run_git",
        should_not_run,
    )

    result = (
        git_tools
        .workspace_git_switch_branch(
            repository="ai",
            branch_name="main",
        )
    )

    assert called is False

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


def test_switch_branch_revalidates_after_approval(
    monkeypatch,
):

    monkeypatch.setattr(
        git_tools,
        "_validate_switch_branch_request",

        lambda **kwargs: (
            None,
            None,
            None,
            "working tree is dirty",
        ),
    )

    result = (
        git_tools
        .workspace_git_switch_branch(
            repository="ai",
            branch_name="test/agentic-git-smoke",
        )
    )

    assert result["ok"] is False
    assert result["status"] == "denied"


def test_generic_process_exec_cannot_switch_git_branches():

    result = (
        evaluate_process_policy(
            executable="git",

            args=[
                "switch",
                "main",
            ],
        )
    )

    assert result["ok"] is False
    assert result["status"] == "denied"


def test_switch_branch_mcp_preserves_exact_name(
    monkeypatch,
):

    server = FakeMCPServer()

    monkeypatch.setattr(
        git_mcp,
        "run_git_switch_branch",

        lambda repository, branch_name: {
            "ok":
                True,

            "status":
                "success",

            "repository":
                repository,

            "requested_branch":
                branch_name,

            "previous_branch":
                "main",

            "current_branch":
                branch_name,

            "switched_branch":
                branch_name,

            "switched_to_commit":
                COMMIT,

            "mutation_performed":
                True,

            "files":
                [],

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
            "workspace_git_switch_branch"
        ](
            repository="ai",

            branch_name="test/agentic-git-smoke",
        )
    )

    assert result.repository == "ai"

    assert (
        result.current_branch
        == "test/agentic-git-smoke"
    )
