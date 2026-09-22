from __future__ import annotations

from typing import Any

from tools.git.workflow import (
    evaluate_git_governed_switch_policy,
    evaluate_git_push_policy,
)


PUSH_TRUSTED_ARGUMENTS = [
    "expected_branch",
    "expected_head",
    "expected_remote",
    "expected_remote_ref",
    "expected_remote_url",
    "expected_remote_head",
]

SWITCH_TRUSTED_ARGUMENTS = [
    "expected_source_branch",
    "expected_source_head",
    "expected_target_head",
    "expected_worktree_digest",
    "checkpoint_required",
    "expected_remote",
    "expected_remote_ref",
    "expected_remote_url",
    "expected_remote_head",
    "checkpoint_message",
]


def format_git_push_approval(arguments: dict[str, Any]) -> str:
    repository = arguments.get("repository") or "repository"
    branch = arguments.get("expected_branch") or "current branch"
    remote = arguments.get("expected_remote") or "configured remote"
    remote_ref = arguments.get("expected_remote_ref") or "configured upstream"
    head = arguments.get("expected_head")
    head_text = head[:12] if isinstance(head, str) and head else "unknown"
    return (
        f"Pushing {repository}:{branch} at {head_text} to its configured "
        f"upstream {remote}:{remote_ref} requires approval. No force push, "
        "tags, arbitrary refspec, or model-selected remote is permitted."
    )


def format_git_push_result(result: dict[str, Any]) -> str:
    repository = result.get("repository") or "repository"
    branch = result.get("branch") or "branch"
    remote = result.get("remote") or "configured remote"
    commit = result.get("commit")
    commit_text = commit[:12] if isinstance(commit, str) and commit else "unknown"
    if result.get("mutation_performed") is False:
        return f"{repository}:{branch} was already synchronized with {remote} at {commit_text}."
    if result.get("verification_ok") is False:
        return f"Git push ran for {repository}:{branch}, but remote verification did not complete successfully."
    return f"Pushed {repository}:{branch} to {remote} and verified remote commit {commit_text}."


def format_git_governed_switch_approval(arguments: dict[str, Any]) -> str:
    repository = arguments.get("repository") or "repository"
    target = arguments.get("branch_name") or "requested branch"
    if arguments.get("checkpoint_required") is True:
        source = arguments.get("expected_source_branch") or "current branch"
        remote = arguments.get("expected_remote") or "configured remote"
        remote_ref = arguments.get("expected_remote_ref") or "configured upstream"
        message = arguments.get("checkpoint_message") or "trusted checkpoint message"
        return (
            f"Switching {repository} from {source} to {target} requires a HIGH-risk "
            "checkpoint workflow: stage ALL current non-ignored changes, create one "
            f"commit with message {message!r}, push the exact new commit to "
            f"{remote}:{remote_ref}, verify the remote commit, then switch to {target}."
        )
    return (
        f"Switching {repository} to existing local branch {target} requires approval. "
        "The working tree is clean, so no checkpoint commit or push will be performed."
    )


def format_git_governed_switch_result(result: dict[str, Any]) -> str:
    repository = result.get("repository") or "repository"
    target = result.get("current_branch") or result.get("requested_branch") or "requested branch"
    checkpoint_commit = result.get("checkpoint_commit")
    if result.get("checkpoint_performed") is True and isinstance(checkpoint_commit, str):
        return (
            f"Checkpointed and pushed the previous branch in {repository}, then switched "
            f"to {target}. Checkpoint commit: {checkpoint_commit[:12]}."
        )
    return f"Switched {repository} to {target}."


def install_git_workflow_catalog(
    git_tools: dict[str, dict[str, Any]],
    repository_parameter: dict[str, Any],
    argument_values_resolver,
) -> None:
    git_tools["workspace_git_push"] = {
        "description": (
            "PUSH the exact current commit of the current attached Git branch to that "
            "branch's already-configured upstream remote and branch. Trusted policy "
            "binds the current branch, HEAD commit, remote, remote URL, upstream ref, "
            "and remote HEAD into approval. Execution fails closed if bound state "
            "changes. This never force-pushes, pushes tags, selects an arbitrary "
            "remote/refspec, stages, commits, pulls, fetches, or switches branches."
        ),
        "risk": "high",
        "requires_approval": True,
        "policy_owns_preconditions": True,
        "policy_resolver": evaluate_git_push_policy,
        "grounded_arguments": ["repository"],
        "trusted_policy_arguments": PUSH_TRUSTED_ARGUMENTS,
        "parameters": {"repository": repository_parameter},
        "argument_values_resolver": argument_values_resolver,
        "result_formatter": format_git_push_result,
        "approval_formatter": format_git_push_approval,
    }

    switch_tool = git_tools["workspace_git_switch_branch"]
    switch_tool.update(
        {
            "description": (
                "SWITCH to one exact existing LOCAL Git branch. Trusted policy chooses "
                "the safe execution path. If the working tree is clean, only the branch "
                "switch is performed. If the working tree is dirty, the governed workflow "
                "first stages ALL current non-ignored changes with git add -A, creates a "
                "deterministic checkpoint commit, pushes that exact commit to the current "
                "branch's configured upstream, verifies the remote commit, and only then "
                "switches branches. Dirty checkpoint switching requires HIGH-risk approval "
                "and fails closed before mutation if conflicts, special Git operations, "
                "missing commit identity, missing upstream, unsupported remote, or an "
                "unsynchronized source HEAD are detected. It never stashes, discards, "
                "force-pushes, creates a branch, pulls, or fetches."
            ),
            "policy_resolver": evaluate_git_governed_switch_policy,
            "trusted_policy_arguments": SWITCH_TRUSTED_ARGUMENTS,
            "approval_formatter": format_git_governed_switch_approval,
            "result_formatter": format_git_governed_switch_result,
        }
    )
