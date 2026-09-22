from types import (
    SimpleNamespace,
)

import tools.git as git_tools
import tools.git.mcp as git_mcp

from tools.git.catalog import (
    GIT_TOOLS,
)


BASE_COMMIT = (
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


def test_create_branch_catalog_is_approval_gated():

    tool = (
        GIT_TOOLS[
            "workspace_git_create_branch"
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


def test_create_branch_policy_uses_git_ref_validation(
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
                "feature/safe\n",
        },

        {
            "ok":
                True,

            "stdout":
                f"{BASE_COMMIT}\n",
        },

        {
            "ok":
                False,

            "status":
                "error",

            "exit_code":
                1,

            "stdout":
                "",

            "stderr":
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
        .evaluate_git_create_branch_policy(
            repository="ai",
            branch_name="feature/safe",
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

    assert (
        calls
        == [
            [
                "check-ref-format",
                "--branch",
                "feature/safe",
            ],

            [
                "rev-parse",
                "--verify",
                "HEAD",
            ],

            [
                "show-ref",
                "--verify",
                "--quiet",
                "refs/heads/feature/safe",
            ],
        ]
    )


def test_create_branch_policy_rejects_existing_branch(
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
                "feature/existing\n",
        },

        {
            "ok":
                True,

            "stdout":
                f"{BASE_COMMIT}\n",
        },

        {
            "ok":
                True,

            "status":
                "success",

            "exit_code":
                0,
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

    result = (
        git_tools
        .evaluate_git_create_branch_policy(
            repository="ai",
            branch_name="feature/existing",
        )
    )

    assert (
        result[
            "ok"
        ]
        is False
    )

    assert (
        "already exists"
        in result[
            "error"
        ]
    )


def test_create_branch_does_not_normalize_name(
    monkeypatch,
):

    called = False

    def fail_resolve(
        repository,
    ):

        nonlocal called

        called = True

        return (
            _target(),
            None,
        )

    monkeypatch.setattr(
        git_tools,
        "_resolve_target",
        fail_resolve,
    )

    result = (
        git_tools
        .evaluate_git_create_branch_policy(
            repository="ai",
            branch_name=" feature/safe ",
        )
    )

    assert (
        called
        is True
    )

    assert (
        result[
            "ok"
        ]
        is False
    )


def test_create_branch_uses_trusted_head_and_verifies(
    monkeypatch,
):

    monkeypatch.setattr(
        git_tools,
        "_validate_create_branch_request",

        lambda **kwargs: (
            _target(),
            "feature/safe",
            BASE_COMMIT,
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
        },

        {
            "ok":
                True,

            "status":
                "success",

            "stdout":
                f"{BASE_COMMIT}\n",
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
        .workspace_git_create_branch(
            repository="ai",
            branch_name="feature/safe",
        )
    )

    assert (
        calls
        == [
            [
                "branch",
                "feature/safe",
                BASE_COMMIT,
            ],

            [
                "rev-parse",
                "--verify",
                "refs/heads/feature/safe",
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
            "created_branch"
        ]
        == "feature/safe"
    )

    assert (
        result[
            "created_from"
        ]
        == BASE_COMMIT
    )

    assert (
        result[
            "created_commit"
        ]
        == BASE_COMMIT
    )

    assert (
        result[
            "verification_ok"
        ]
        is True
    )


def test_create_branch_revalidates_after_approval(
    monkeypatch,
):

    monkeypatch.setattr(
        git_tools,
        "_validate_create_branch_request",

        lambda **kwargs: (
            None,
            None,
            None,
            "invalid branch",
        ),
    )

    result = (
        git_tools
        .workspace_git_create_branch(
            repository="ai",
            branch_name="../bad",
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


def test_create_branch_verification_mismatch_preserves_side_effect_truth(
    monkeypatch,
):

    monkeypatch.setattr(
        git_tools,
        "_validate_create_branch_request",

        lambda **kwargs: (
            _target(),
            "feature/safe",
            BASE_COMMIT,
            None,
        ),
    )

    other_commit = (
        "fedcba9876543210"
        "fedcba9876543210"
        "fedcba98"
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
                True,

            "status":
                "success",

            "stdout":
                f"{other_commit}\n",
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

    result = (
        git_tools
        .workspace_git_create_branch(
            repository="ai",
            branch_name="feature/safe",
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
            "verification_ok"
        ]
        is False
    )

    assert (
        result[
            "created_commit"
        ]
        == other_commit
    )


def test_create_branch_mcp_preserves_exact_name(
    monkeypatch,
):

    server = (
        FakeMCPServer()
    )

    monkeypatch.setattr(
        git_mcp,
        "run_git_create_branch",

        lambda repository, branch_name: {
            "ok":
                True,

            "status":
                "success",

            "repository":
                repository,

            "requested_branch":
                branch_name,

            "created_branch":
                branch_name,

            "created_from":
                BASE_COMMIT,

            "created_commit":
                BASE_COMMIT,

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
            "workspace_git_create_branch"
        ](
            repository="ai",
            branch_name="feature/safe",
        )
    )

    assert (
        result.repository
        == "ai"
    )

    assert (
        result.created_branch
        == "feature/safe"
    )
