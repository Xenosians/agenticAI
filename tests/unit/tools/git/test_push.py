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


LOCAL_HEAD = (
    "aaaaaaaaaaaaaaaa"
    "aaaaaaaaaaaaaaaa"
    "aaaaaaaa"
)

REMOTE_HEAD = (
    "bbbbbbbbbbbbbbbb"
    "bbbbbbbbbbbbbbbb"
    "bbbbbbbb"
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


def _snapshot(
    *,
    head=LOCAL_HEAD,
    remote_head=REMOTE_HEAD,
):

    return {
        "branch":
            "main",

        "head":
            head,

        "remote":
            "origin",

        "remote_ref":
            "refs/heads/main",

        "remote_url":
            "git@github.com:Xenosians/agenticAI.git",

        "remote_head":
            remote_head,
    }


def test_push_catalog_is_high_risk_and_approval_gated():

    tool = (
        GIT_TOOLS[
            "workspace_git_push"
        ]
    )

    assert (
        tool[
            "risk"
        ]
        == "high"
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
        ]
    )

    assert (
        set(
            tool[
                "trusted_policy_arguments"
            ]
        )
        == {
            "expected_branch",
            "expected_head",
            "expected_remote",
            "expected_remote_ref",
            "expected_remote_url",
            "expected_remote_head",
        }
    )


def test_push_policy_binds_exact_snapshot(
    monkeypatch,
):

    monkeypatch.setattr(
        git_tools,
        "_git_push_snapshot",

        lambda **kwargs: (
            _target(),
            _snapshot(),
            None,
        ),
    )

    result = (
        git_tools
        .evaluate_git_push_policy(
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
            "risk"
        ]
        == "high"
    )

    arguments = (
        result[
            "execution_arguments"
        ]
    )

    assert (
        arguments[
            "repository"
        ]
        == "ai"
    )

    assert (
        arguments[
            "expected_branch"
        ]
        == "main"
    )

    assert (
        arguments[
            "expected_head"
        ]
        == LOCAL_HEAD
    )

    assert (
        arguments[
            "expected_remote"
        ]
        == "origin"
    )

    assert (
        arguments[
            "expected_remote_head"
        ]
        == REMOTE_HEAD
    )


def test_push_denies_if_approval_snapshot_changes(
    monkeypatch,
):

    changed_head = (
        "cccccccccccccccc"
        "cccccccccccccccc"
        "cccccccc"
    )

    monkeypatch.setattr(
        git_tools,
        "_git_push_snapshot",

        lambda **kwargs: (
            _target(),

            _snapshot(
                head=changed_head,
            ),

            None,
        ),
    )

    result = (
        git_tools
        .workspace_git_push(
            repository="ai",
            expected_branch="main",
            expected_head=LOCAL_HEAD,
            expected_remote="origin",
            expected_remote_ref="refs/heads/main",
            expected_remote_url=(
                "git@github.com:Xenosians/agenticAI.git"
            ),
            expected_remote_head=REMOTE_HEAD,
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
        is False
    )


def test_push_uses_exact_non_force_refspec_and_verifies(
    monkeypatch,
):

    monkeypatch.setattr(
        git_tools,
        "_git_push_snapshot",

        lambda **kwargs: (
            _target(),
            _snapshot(),
            None,
        ),
    )

    calls = []

    def fake_network(
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

            "status":
                "success",

            "stdout":
                "",
        }

    monkeypatch.setattr(
        git_tools,
        "_run_git_network",
        fake_network,
    )

    monkeypatch.setattr(
        git_tools,
        "_git_push_remote_head",

        lambda **kwargs: (
            LOCAL_HEAD,
            None,
        ),
    )

    result = (
        git_tools
        .workspace_git_push(
            repository="ai",
            expected_branch="main",
            expected_head=LOCAL_HEAD,
            expected_remote="origin",
            expected_remote_ref="refs/heads/main",
            expected_remote_url=(
                "git@github.com:Xenosians/agenticAI.git"
            ),
            expected_remote_head=REMOTE_HEAD,
        )
    )

    assert len(
        calls
    ) == 1

    command = (
        calls[
            0
        ]
    )

    assert (
        "push"
        in command
    )

    assert (
        "--no-verify"
        in command
    )

    assert (
        "origin"
        in command
    )

    assert (
        f"{LOCAL_HEAD}:refs/heads/main"
        in command
    )

    assert (
        "--force"
        not in command
    )

    assert all(
        not value.startswith(
            "+"
        )

        for value
        in command
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
        is True
    )

    assert (
        result[
            "remote_commit"
        ]
        == LOCAL_HEAD
    )


def test_push_is_truthful_noop_when_already_synchronized(
    monkeypatch,
):

    monkeypatch.setattr(
        git_tools,
        "_git_push_snapshot",

        lambda **kwargs: (
            _target(),

            _snapshot(
                head=LOCAL_HEAD,
                remote_head=LOCAL_HEAD,
            ),

            None,
        ),
    )

    def must_not_execute(
        **kwargs,
    ):

        raise AssertionError(
            "Network push must not execute for synchronized ref."
        )

    monkeypatch.setattr(
        git_tools,
        "_run_git_network",
        must_not_execute,
    )

    result = (
        git_tools
        .workspace_git_push(
            repository="ai",
            expected_branch="main",
            expected_head=LOCAL_HEAD,
            expected_remote="origin",
            expected_remote_ref="refs/heads/main",
            expected_remote_url=(
                "git@github.com:Xenosians/agenticAI.git"
            ),
            expected_remote_head=LOCAL_HEAD,
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
        is False
    )

    assert (
        result[
            "verification_ok"
        ]
        is True
    )


def test_generic_process_policy_cannot_git_push():

    result = (
        evaluate_process_policy(
            executable="git",

            args=[
                "push",
                "origin",
                "main",
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


def test_push_registers_with_mcp(
    monkeypatch,
):

    server = (
        FakeMCPServer()
    )

    monkeypatch.setattr(
        git_mcp,
        "run_git_push",

        lambda **kwargs: {
            "ok":
                True,

            "status":
                "success",

            "repository":
                kwargs[
                    "repository"
                ],

            "branch":
                kwargs[
                    "expected_branch"
                ],

            "commit":
                kwargs[
                    "expected_head"
                ],

            "remote":
                kwargs[
                    "expected_remote"
                ],

            "remote_ref":
                kwargs[
                    "expected_remote_ref"
                ],

            "remote_url":
                kwargs[
                    "expected_remote_url"
                ],

            "remote_commit":
                kwargs[
                    "expected_head"
                ],

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

    assert (
        "workspace_git_push"
        in server.functions
    )

    result = (
        server.functions[
            "workspace_git_push"
        ](
            repository="ai",
            expected_branch="main",
            expected_head=LOCAL_HEAD,
            expected_remote="origin",
            expected_remote_ref="refs/heads/main",
            expected_remote_url=(
                "git@github.com:Xenosians/agenticAI.git"
            ),
            expected_remote_head=REMOTE_HEAD,
        )
    )

    assert (
        result.repository
        == "ai"
    )

    assert (
        result.branch
        == "main"
    )

    assert (
        result.remote_commit
        == LOCAL_HEAD
    )
