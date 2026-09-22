from __future__ import annotations

import hashlib
import os
import re

from pathlib import Path
from typing import Any

from services.git_repositories import GitRepositoryTarget
from tools.git import (
    DEFAULT_TIMEOUT_SECONDS,
    _git_mutation_worktree_state,
    _git_operation_in_progress,
    _process_failure,
    _resolve_target,
    _run_git,
    _stdout_lines,
    _working_tree_summary,
    workspace_git_commit,
    workspace_git_stage_all,
    workspace_git_switch_branch as _clean_switch,
)

NETWORK_TIMEOUT_SECONDS = 30
MAX_CHECKPOINT_PATHS = 200
MAX_BRANCH_NAME_CHARS = 255
SCP_REMOTE_PATTERN = re.compile(r"^[^@\s:/]+@[^:\s/]+:.+$")


def _single_stdout(result: dict[str, Any], *, error: str) -> tuple[str | None, str | None]:
    if not result.get("ok", False):
        return None, error
    lines = _stdout_lines(result)
    if len(lines) != 1:
        return None, error
    value = lines[0].strip()
    if not value:
        return None, error
    return value, None


def _current_branch_and_head(*, target: GitRepositoryTarget, timeout_seconds: int) -> tuple[str | None, str | None, str | None]:
    branch, error = _single_stdout(
        _run_git(
            target=target,
            args=["symbolic-ref", "--quiet", "--short", "HEAD"],
            timeout_seconds=timeout_seconds,
        ),
        error="Governed Git mutation requires an attached current branch.",
    )
    if branch is None:
        return None, None, error

    head, error = _single_stdout(
        _run_git(
            target=target,
            args=["rev-parse", "--verify", "HEAD"],
            timeout_seconds=timeout_seconds,
        ),
        error="Could not safely resolve the current Git HEAD.",
    )
    if head is None:
        return None, None, error
    return branch, head, None


def _validate_local_target_branch(*, target: GitRepositoryTarget, branch_name: Any, timeout_seconds: int) -> tuple[str | None, str | None, str | None]:
    if not isinstance(branch_name, str):
        return None, None, "Git branch name must be a string."
    if not branch_name:
        return None, None, "Git branch name cannot be empty."
    if branch_name != branch_name.strip():
        return None, None, "Git branch name must not contain leading or trailing whitespace."
    if len(branch_name) > MAX_BRANCH_NAME_CHARS:
        return None, None, f"Git branch name exceeds the {MAX_BRANCH_NAME_CHARS}-character limit."
    if branch_name.startswith("-"):
        return None, None, "Git branch name must not begin with '-'."
    if "@{" in branch_name:
        return None, None, "Git branch name must not contain reflog-selection syntax."
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in branch_name):
        return None, None, "Git branch name contains control characters."

    validation = _run_git(
        target=target,
        args=["check-ref-format", "--branch", branch_name],
        timeout_seconds=timeout_seconds,
    )
    lines = _stdout_lines(validation) if validation.get("ok", False) else []
    if len(lines) != 1 or lines[0] != branch_name:
        return None, None, "Git branch name is not valid."

    ref = f"refs/heads/{branch_name}"
    exists = _run_git(
        target=target,
        args=["show-ref", "--verify", "--quiet", ref],
        timeout_seconds=timeout_seconds,
    )
    if not exists.get("ok", False):
        if exists.get("exit_code") == 1:
            return None, None, f"Local Git branch '{branch_name}' does not exist."
        return None, None, "Could not safely verify the requested local Git branch."

    ref_head, error = _single_stdout(
        _run_git(
            target=target,
            args=["rev-parse", "--verify", ref],
            timeout_seconds=timeout_seconds,
        ),
        error="Could not safely resolve the requested local branch.",
    )
    if ref_head is None:
        return None, None, error
    return branch_name, ref_head, None


def _configured_upstream(*, target: GitRepositoryTarget, branch: str, timeout_seconds: int) -> tuple[str | None, str | None, str | None]:
    remote, error = _single_stdout(
        _run_git(
            target=target,
            args=["config", "--get", f"branch.{branch}.remote"],
            timeout_seconds=timeout_seconds,
        ),
        error=f"Current branch '{branch}' does not have a configured upstream remote.",
    )
    if remote is None:
        return None, None, error
    if remote == "." or remote.startswith("-"):
        return None, None, "Configured Git remote is not permitted for governed push."

    remote_ref, error = _single_stdout(
        _run_git(
            target=target,
            args=["config", "--get", f"branch.{branch}.merge"],
            timeout_seconds=timeout_seconds,
        ),
        error=f"Current branch '{branch}' does not have a configured upstream branch.",
    )
    if remote_ref is None:
        return None, None, error
    if not remote_ref.startswith("refs/heads/"):
        return None, None, "Governed push only supports configured branch upstream refs."
    return remote, remote_ref, None


def _configured_push_url(*, target: GitRepositoryTarget, remote: str, timeout_seconds: int) -> tuple[str | None, str | None]:
    url, error = _single_stdout(
        _run_git(
            target=target,
            args=["remote", "get-url", "--push", remote],
            timeout_seconds=timeout_seconds,
        ),
        error=f"Could not safely resolve push URL for remote '{remote}'.",
    )
    if url is None:
        return None, error
    allowed = url.startswith("https://") or url.startswith("ssh://") or SCP_REMOTE_PATTERN.fullmatch(url) is not None
    if not allowed:
        return None, "Governed Git push only permits configured HTTPS or SSH remotes."
    return url, None


def _remote_branch_head(*, target: GitRepositoryTarget, remote: str, remote_ref: str, timeout_seconds: int) -> tuple[str | None, str | None]:
    result = _run_git(
        target=target,
        args=["ls-remote", "--heads", remote, remote_ref],
        timeout_seconds=timeout_seconds,
    )
    if not result.get("ok", False):
        failure = _process_failure(repository=target.name, result=result)
        return None, failure.get("error") or "Could not inspect the configured remote branch."
    lines = _stdout_lines(result)
    if len(lines) != 1:
        return None, "Configured upstream branch does not resolve to exactly one remote branch."
    pieces = lines[0].split()
    if len(pieces) != 2 or pieces[1] != remote_ref:
        return None, "Configured upstream branch returned an unexpected remote reference."
    return pieces[0], None


def _worktree_snapshot(*, target: GitRepositoryTarget, timeout_seconds: int) -> tuple[str | None, dict[str, Any] | None, str | None]:
    changes, error = _git_mutation_worktree_state(target=target, timeout_seconds=timeout_seconds)
    if changes is None:
        return None, None, error or "Could not inspect the Git working tree."

    summary = _working_tree_summary(changes)
    paths = summary["files"]
    if len(paths) > MAX_CHECKPOINT_PATHS:
        return None, None, f"Dirty checkpoint exceeds the {MAX_CHECKPOINT_PATHS}-path limit."

    digest = hashlib.sha256()
    for change in changes:
        digest.update(repr(sorted(change.items())).encode("utf-8"))
        digest.update(b"\0")

    for relative in sorted(paths):
        # Porcelain v1 quote/rename parsing is intentionally not expanded here.
        # Fail closed instead of hashing the wrong path.
        if relative.startswith('"') or " -> " in relative:
            return None, None, "Dirty checkpoint currently rejects quoted or renamed porcelain paths."
        if os.path.isabs(relative) or relative == ".." or relative.startswith("../"):
            return None, None, "Unsafe changed path encountered while binding checkpoint approval."

        path = target.path / relative
        digest.update(relative.encode("utf-8", errors="surrogateescape"))
        try:
            st = path.lstat()
        except FileNotFoundError:
            digest.update(b"<missing>")
        else:
            digest.update(f"mode:{st.st_mode:o}".encode("ascii"))
            if path.is_symlink():
                digest.update(b"<symlink>")
                digest.update(os.readlink(path).encode("utf-8", errors="surrogateescape"))
            elif path.is_file():
                digest.update(b"<file>")
                with path.open("rb") as handle:
                    while True:
                        chunk = handle.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
            else:
                digest.update(b"<other>")

        index_state = _run_git(
            target=target,
            args=["ls-files", "--stage", "--", relative],
            timeout_seconds=timeout_seconds,
        )
        if not index_state.get("ok", False):
            return None, None, f"Could not bind index state for '{relative}'."
        digest.update("\n".join(_stdout_lines(index_state)).encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")

    return digest.hexdigest(), {
        "dirty": bool(paths),
        "changed_paths": paths,
        "changed_count": len(paths),
        "conflicted_count": summary["conflicted_count"],
    }, None


def _validate_commit_environment(*, target: GitRepositoryTarget, timeout_seconds: int) -> str | None:
    operation, error = _git_operation_in_progress(target=target, timeout_seconds=timeout_seconds)
    if error is not None:
        return error
    if operation is not None:
        return f"Automatic checkpoint commit is not allowed while a {operation} operation is in progress."
    for ident in ("GIT_AUTHOR_IDENT", "GIT_COMMITTER_IDENT"):
        result = _run_git(target=target, args=["var", ident], timeout_seconds=timeout_seconds)
        if not result.get("ok", False):
            return "Git commit identity is not configured for this repository/runtime."
    return None


def _push_snapshot(repository: Any, timeout_seconds: int = NETWORK_TIMEOUT_SECONDS) -> tuple[GitRepositoryTarget | None, dict[str, str] | None, str | None]:
    target, target_error = _resolve_target(repository)
    if target is None:
        return None, None, target_error.get("error") if isinstance(target_error, dict) else "Git repository resolution failed."

    branch, head, error = _current_branch_and_head(target=target, timeout_seconds=timeout_seconds)
    if branch is None or head is None:
        return None, None, error

    remote, remote_ref, error = _configured_upstream(target=target, branch=branch, timeout_seconds=timeout_seconds)
    if remote is None or remote_ref is None:
        return None, None, error

    remote_url, error = _configured_push_url(target=target, remote=remote, timeout_seconds=timeout_seconds)
    if remote_url is None:
        return None, None, error

    remote_head, error = _remote_branch_head(
        target=target,
        remote=remote,
        remote_ref=remote_ref,
        timeout_seconds=timeout_seconds,
    )
    if remote_head is None:
        return None, None, error

    return target, {
        "branch": branch,
        "head": head,
        "remote": remote,
        "remote_ref": remote_ref,
        "remote_url": remote_url,
        "remote_head": remote_head,
    }, None


# ============================================================
# STANDALONE PUSH
# ============================================================


def evaluate_git_push_policy(repository: Any) -> dict[str, Any]:
    target, snapshot, error = _push_snapshot(repository)
    if target is None or snapshot is None:
        return {"ok": False, "status": "denied", "error": error or "Governed Git push policy denied the request."}
    return {
        "ok": True,
        "status": "allowed",
        "risk": "high",
        "requires_approval": True,
        "execution_arguments": {
            "repository": target.name,
            "expected_branch": snapshot["branch"],
            "expected_head": snapshot["head"],
            "expected_remote": snapshot["remote"],
            "expected_remote_ref": snapshot["remote_ref"],
            "expected_remote_url": snapshot["remote_url"],
            "expected_remote_head": snapshot["remote_head"],
        },
    }


def _push_exact_snapshot(*, repository: str, expected_branch: str, expected_head: str, expected_remote: str, expected_remote_ref: str, expected_remote_url: str, expected_remote_head: str, timeout_seconds: int = NETWORK_TIMEOUT_SECONDS) -> dict[str, Any]:
    target, snapshot, error = _push_snapshot(repository, timeout_seconds)
    if target is None or snapshot is None:
        return {
            "ok": False,
            "status": "denied",
            "repository": repository,
            "mutation_performed": False,
            "verification_ok": None,
            "verification_error": None,
            "error": error or "Could not safely revalidate governed Git push.",
        }

    expected = {
        "branch": expected_branch,
        "head": expected_head,
        "remote": expected_remote,
        "remote_ref": expected_remote_ref,
        "remote_url": expected_remote_url,
        "remote_head": expected_remote_head,
    }
    mismatches = [key for key, value in expected.items() if snapshot.get(key) != value]
    if mismatches:
        return {
            "ok": False,
            "status": "denied",
            "repository": target.name,
            "mutation_performed": False,
            "verification_ok": None,
            "verification_error": None,
            "error": "Git push approval snapshot changed before execution: " + ", ".join(sorted(mismatches)) + ".",
        }

    if expected_head == expected_remote_head:
        return {
            "ok": True,
            "status": "success",
            "repository": target.name,
            "branch": expected_branch,
            "commit": expected_head,
            "remote": expected_remote,
            "remote_ref": expected_remote_ref,
            "remote_url": expected_remote_url,
            "remote_commit": expected_remote_head,
            "mutation_performed": False,
            "verification_ok": True,
            "verification_error": None,
            "files": [],
        }

    mutation = _run_git(
        target=target,
        args=[
            "-c",
            "core.hooksPath=/dev/null",
            "push",
            "--porcelain",
            "--no-verify",
            expected_remote,
            f"{expected_head}:{expected_remote_ref}",
        ],
        timeout_seconds=timeout_seconds,
    )
    if not mutation.get("ok", False):
        failure = _process_failure(repository=target.name, result=mutation)
        return {
            **failure,
            "branch": expected_branch,
            "commit": expected_head,
            "remote": expected_remote,
            "remote_ref": expected_remote_ref,
            "remote_url": expected_remote_url,
            "mutation_performed": False,
            "verification_ok": None,
            "verification_error": None,
            "files": [],
        }

    remote_head, verify_error = _remote_branch_head(
        target=target,
        remote=expected_remote,
        remote_ref=expected_remote_ref,
        timeout_seconds=timeout_seconds,
    )
    verification_ok = remote_head == expected_head
    return {
        "ok": True,
        "status": "success",
        "repository": target.name,
        "branch": expected_branch,
        "commit": expected_head,
        "remote": expected_remote,
        "remote_ref": expected_remote_ref,
        "remote_url": expected_remote_url,
        "remote_commit": remote_head,
        "mutation_performed": True,
        "verification_ok": verification_ok,
        "verification_error": None if verification_ok else (verify_error or "Remote branch does not point to the exact pushed commit."),
        "files": [],
    }


def workspace_git_push(repository: str, expected_branch: str, expected_head: str, expected_remote: str, expected_remote_ref: str, expected_remote_url: str, expected_remote_head: str) -> dict[str, Any]:
    return _push_exact_snapshot(
        repository=repository,
        expected_branch=expected_branch,
        expected_head=expected_head,
        expected_remote=expected_remote,
        expected_remote_ref=expected_remote_ref,
        expected_remote_url=expected_remote_url,
        expected_remote_head=expected_remote_head,
    )


# ============================================================
# GOVERNED USER-FACING SWITCH
# ============================================================


def evaluate_git_governed_switch_policy(repository: Any, branch_name: Any) -> dict[str, Any]:
    target, target_error = _resolve_target(repository)
    if target is None:
        return {"ok": False, "status": "denied", "error": target_error.get("error") if isinstance(target_error, dict) else "Git repository resolution failed."}

    target_branch, target_head, error = _validate_local_target_branch(
        target=target,
        branch_name=branch_name,
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
    )
    if target_branch is None or target_head is None:
        return {"ok": False, "status": "denied", "error": error or "Requested target branch is not allowed."}

    source_branch, source_head, error = _current_branch_and_head(target=target, timeout_seconds=DEFAULT_TIMEOUT_SECONDS)
    if source_branch is None or source_head is None:
        return {"ok": False, "status": "denied", "error": error}

    digest, state, error = _worktree_snapshot(target=target, timeout_seconds=DEFAULT_TIMEOUT_SECONDS)
    if digest is None or state is None:
        return {"ok": False, "status": "denied", "error": error or "Could not bind working tree to approval."}
    if state["conflicted_count"] > 0:
        return {"ok": False, "status": "denied", "error": "Governed branch switching cannot checkpoint conflicted paths."}

    base_args: dict[str, Any] = {
        "repository": target.name,
        "branch_name": target_branch,
        "expected_source_branch": source_branch,
        "expected_source_head": source_head,
        "expected_target_head": target_head,
        "expected_worktree_digest": digest,
        "checkpoint_required": False,
        "expected_remote": None,
        "expected_remote_ref": None,
        "expected_remote_url": None,
        "expected_remote_head": None,
        "checkpoint_message": None,
    }

    if source_branch == target_branch:
        return {"ok": True, "status": "allowed", "risk": "low", "requires_approval": False, "execution_arguments": base_args}

    if not state["dirty"]:
        return {"ok": True, "status": "allowed", "risk": "low", "requires_approval": True, "execution_arguments": base_args}

    error = _validate_commit_environment(target=target, timeout_seconds=DEFAULT_TIMEOUT_SECONDS)
    if error is not None:
        return {"ok": False, "status": "denied", "error": error}

    remote, remote_ref, error = _configured_upstream(target=target, branch=source_branch, timeout_seconds=DEFAULT_TIMEOUT_SECONDS)
    if remote is None or remote_ref is None:
        return {"ok": False, "status": "denied", "error": error}

    remote_url, error = _configured_push_url(target=target, remote=remote, timeout_seconds=DEFAULT_TIMEOUT_SECONDS)
    if remote_url is None:
        return {"ok": False, "status": "denied", "error": error}

    remote_head, error = _remote_branch_head(
        target=target,
        remote=remote,
        remote_ref=remote_ref,
        timeout_seconds=NETWORK_TIMEOUT_SECONDS,
    )
    if remote_head is None:
        return {"ok": False, "status": "denied", "error": error}

    # Strict v1: automatic checkpoint only when source HEAD is already
    # synchronized with its configured upstream. This prevents silently
    # bundling older unpushed local commits into the automatic push.
    if remote_head != source_head:
        return {
            "ok": False,
            "status": "denied",
            "error": "Automatic checkpoint-and-switch requires the current local branch HEAD to match its configured remote branch before checkpointing.",
        }

    base_args.update(
        {
            "checkpoint_required": True,
            "expected_remote": remote,
            "expected_remote_ref": remote_ref,
            "expected_remote_url": remote_url,
            "expected_remote_head": remote_head,
            "checkpoint_message": f"checkpoint before switching to {target_branch}",
        }
    )
    return {"ok": True, "status": "allowed", "risk": "high", "requires_approval": True, "execution_arguments": base_args}


def _failure(*, repository: str, branch_name: str, source_branch: str, error: str, mutation_performed: bool, checkpoint_commit: str | None = None, checkpoint_pushed: bool = False, files: list[str] | None = None, status: str = "error") -> dict[str, Any]:
    return {
        "ok": False,
        "status": status,
        "repository": repository,
        "requested_branch": branch_name,
        "previous_branch": source_branch,
        "current_branch": source_branch,
        "switched_branch": None,
        "switched_to_commit": None,
        "checkpoint_performed": checkpoint_commit is not None,
        "checkpoint_commit": checkpoint_commit,
        "checkpoint_pushed": checkpoint_pushed,
        "mutation_performed": mutation_performed,
        "files": files or [],
        "verification_ok": False if mutation_performed else None,
        "verification_error": error if mutation_performed else None,
        "error": error,
    }


def workspace_git_governed_switch_branch(
    repository: str,
    branch_name: str,
    expected_source_branch: str,
    expected_source_head: str,
    expected_target_head: str,
    expected_worktree_digest: str,
    checkpoint_required: bool,
    expected_remote: str | None = None,
    expected_remote_ref: str | None = None,
    expected_remote_url: str | None = None,
    expected_remote_head: str | None = None,
    checkpoint_message: str | None = None,
) -> dict[str, Any]:
    target, target_error = _resolve_target(repository)
    if target is None:
        return _failure(
            repository=repository,
            branch_name=branch_name,
            source_branch=expected_source_branch,
            error=target_error.get("error") if isinstance(target_error, dict) else "Git repository resolution failed.",
            mutation_performed=False,
            status="denied",
        )

    target_branch, target_head, target_error_text = _validate_local_target_branch(
        target=target,
        branch_name=branch_name,
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
    )
    source_branch, source_head, source_error = _current_branch_and_head(target=target, timeout_seconds=DEFAULT_TIMEOUT_SECONDS)
    digest, state, digest_error = _worktree_snapshot(target=target, timeout_seconds=DEFAULT_TIMEOUT_SECONDS)

    problems: list[str] = []
    if target_branch != branch_name or target_head != expected_target_head:
        problems.append(target_error_text or "target branch changed")
    if source_branch != expected_source_branch:
        problems.append(source_error or "current branch changed")
    if source_head != expected_source_head:
        problems.append("current HEAD changed")
    if digest != expected_worktree_digest:
        problems.append(digest_error or "working tree changed")

    if problems:
        return _failure(
            repository=target.name,
            branch_name=branch_name,
            source_branch=expected_source_branch,
            error="Git branch-switch approval snapshot changed before execution: " + "; ".join(problems) + ".",
            mutation_performed=False,
            status="denied",
        )

    if source_branch == branch_name:
        return {
            "ok": True,
            "status": "success",
            "repository": target.name,
            "requested_branch": branch_name,
            "previous_branch": source_branch,
            "current_branch": source_branch,
            "switched_branch": branch_name,
            "switched_to_commit": source_head,
            "checkpoint_performed": False,
            "checkpoint_commit": None,
            "checkpoint_pushed": False,
            "mutation_performed": False,
            "files": [],
            "verification_ok": True,
            "verification_error": None,
        }

    if not checkpoint_required:
        if state is not None and state.get("dirty"):
            return _failure(
                repository=target.name,
                branch_name=branch_name,
                source_branch=expected_source_branch,
                error="Working tree became dirty after clean-switch approval.",
                mutation_performed=False,
                status="denied",
            )
        result = _clean_switch(repository=repository, branch_name=branch_name)
        if result.get("ok") is True and result.get("verification_ok") is True and result.get("current_branch") == branch_name:
            result["checkpoint_performed"] = False
            result["checkpoint_commit"] = None
            result["checkpoint_pushed"] = False
            return result
        return _failure(
            repository=target.name,
            branch_name=branch_name,
            source_branch=expected_source_branch,
            error=result.get("error") or result.get("verification_error") or "Git branch switch did not complete with verified success.",
            mutation_performed=bool(result.get("mutation_performed")),
        )

    if not all(isinstance(v, str) and v for v in (expected_remote, expected_remote_ref, expected_remote_url, expected_remote_head, checkpoint_message)):
        return _failure(
            repository=target.name,
            branch_name=branch_name,
            source_branch=expected_source_branch,
            error="Dirty checkpoint approval is missing trusted execution bindings.",
            mutation_performed=False,
            status="denied",
        )

    _, upstream, upstream_error = _push_snapshot(repository, NETWORK_TIMEOUT_SECONDS)
    if upstream is None:
        return _failure(
            repository=target.name,
            branch_name=branch_name,
            source_branch=expected_source_branch,
            error=upstream_error or "Could not safely revalidate configured upstream.",
            mutation_performed=False,
            status="denied",
        )

    expected_upstream = {
        "branch": expected_source_branch,
        "head": expected_source_head,
        "remote": expected_remote,
        "remote_ref": expected_remote_ref,
        "remote_url": expected_remote_url,
        "remote_head": expected_remote_head,
    }
    upstream_mismatch = [key for key, value in expected_upstream.items() if upstream.get(key) != value]
    if upstream_mismatch:
        return _failure(
            repository=target.name,
            branch_name=branch_name,
            source_branch=expected_source_branch,
            error="Configured upstream changed before checkpoint execution: " + ", ".join(sorted(upstream_mismatch)) + ".",
            mutation_performed=False,
            status="denied",
        )

    stage = workspace_git_stage_all(repository=repository)
    if not (stage.get("ok") is True and stage.get("verification_ok") is True):
        return _failure(
            repository=target.name,
            branch_name=branch_name,
            source_branch=expected_source_branch,
            error=stage.get("error") or stage.get("verification_error") or "Git stage-all did not verify.",
            mutation_performed=bool(stage.get("mutation_performed")),
            files=stage.get("files") if isinstance(stage.get("files"), list) else [],
        )

    commit = workspace_git_commit(repository=repository, commit_message=checkpoint_message)
    checkpoint_commit = commit.get("commit")
    files = commit.get("files") if isinstance(commit.get("files"), list) else []
    if not (commit.get("ok") is True and commit.get("verification_ok") is True and isinstance(checkpoint_commit, str) and checkpoint_commit):
        return _failure(
            repository=target.name,
            branch_name=branch_name,
            source_branch=expected_source_branch,
            error=commit.get("error") or commit.get("verification_error") or "Automatic checkpoint commit did not verify.",
            mutation_performed=True,
            checkpoint_commit=checkpoint_commit if isinstance(checkpoint_commit, str) else None,
            files=files,
        )

    push = _push_exact_snapshot(
        repository=repository,
        expected_branch=expected_source_branch,
        expected_head=checkpoint_commit,
        expected_remote=expected_remote,
        expected_remote_ref=expected_remote_ref,
        expected_remote_url=expected_remote_url,
        expected_remote_head=expected_remote_head,
    )
    if not (push.get("ok") is True and push.get("verification_ok") is True):
        return _failure(
            repository=target.name,
            branch_name=branch_name,
            source_branch=expected_source_branch,
            error=push.get("error") or push.get("verification_error") or "Checkpoint push did not verify. Branch switch was not attempted.",
            mutation_performed=True,
            checkpoint_commit=checkpoint_commit,
            checkpoint_pushed=False,
            files=files,
        )

    final_target, error = _single_stdout(
        _run_git(
            target=target,
            args=["rev-parse", "--verify", f"refs/heads/{branch_name}"],
            timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        ),
        error="Could not revalidate target branch before switch.",
    )
    if final_target != expected_target_head:
        return _failure(
            repository=target.name,
            branch_name=branch_name,
            source_branch=expected_source_branch,
            error=error or "Target branch changed after checkpoint push. Branch switch was not attempted.",
            mutation_performed=True,
            checkpoint_commit=checkpoint_commit,
            checkpoint_pushed=True,
            files=files,
        )

    switched = _clean_switch(repository=repository, branch_name=branch_name)
    if not (switched.get("ok") is True and switched.get("verification_ok") is True and switched.get("current_branch") == branch_name):
        return _failure(
            repository=target.name,
            branch_name=branch_name,
            source_branch=expected_source_branch,
            error=switched.get("error") or switched.get("verification_error") or "Checkpoint was pushed, but branch switch did not verify.",
            mutation_performed=True,
            checkpoint_commit=checkpoint_commit,
            checkpoint_pushed=True,
            files=files,
        )

    return {
        **switched,
        "checkpoint_performed": True,
        "checkpoint_commit": checkpoint_commit,
        "checkpoint_pushed": True,
        "files": files,
        "mutation_performed": True,
        "verification_ok": True,
        "verification_error": None,
    }
