from types import SimpleNamespace

import tools.git.workflow as workflow

OLD = "0" * 40
NEW = "1" * 40
TARGET = "2" * 40


def _target():
    return SimpleNamespace(name="ai")


def test_push_policy_binds_snapshot(monkeypatch):
    monkeypatch.setattr(
        workflow,
        "_push_snapshot",
        lambda *args, **kwargs: (
            _target(),
            {
                "branch": "main",
                "head": OLD,
                "remote": "origin",
                "remote_ref": "refs/heads/main",
                "remote_url": "https://github.com/example/repo.git",
                "remote_head": OLD,
            },
            None,
        ),
    )
    result = workflow.evaluate_git_push_policy("ai")
    assert result["ok"] is True
    assert result["risk"] == "high"
    assert result["execution_arguments"]["expected_head"] == OLD


def test_push_denies_if_snapshot_changes(monkeypatch):
    monkeypatch.setattr(
        workflow,
        "_push_snapshot",
        lambda *args, **kwargs: (
            _target(),
            {
                "branch": "main",
                "head": NEW,
                "remote": "origin",
                "remote_ref": "refs/heads/main",
                "remote_url": "https://github.com/example/repo.git",
                "remote_head": OLD,
            },
            None,
        ),
    )
    result = workflow.workspace_git_push(
        repository="ai",
        expected_branch="main",
        expected_head=OLD,
        expected_remote="origin",
        expected_remote_ref="refs/heads/main",
        expected_remote_url="https://github.com/example/repo.git",
        expected_remote_head=OLD,
    )
    assert result["ok"] is False
    assert result["status"] == "denied"
    assert result["mutation_performed"] is False


def test_dirty_switch_policy_is_high_risk(monkeypatch):
    monkeypatch.setattr(workflow, "_resolve_target", lambda repository: (_target(), None))
    monkeypatch.setattr(
        workflow,
        "_validate_local_target_branch",
        lambda **kwargs: ("feature/safe", TARGET, None),
    )
    monkeypatch.setattr(
        workflow,
        "_current_branch_and_head",
        lambda **kwargs: ("main", OLD, None),
    )
    monkeypatch.setattr(
        workflow,
        "_worktree_snapshot",
        lambda **kwargs: (
            "digest",
            {"dirty": True, "changed_paths": ["a.py"], "changed_count": 1, "conflicted_count": 0},
            None,
        ),
    )
    monkeypatch.setattr(workflow, "_validate_commit_environment", lambda **kwargs: None)
    monkeypatch.setattr(
        workflow,
        "_configured_upstream",
        lambda **kwargs: ("origin", "refs/heads/main", None),
    )
    monkeypatch.setattr(
        workflow,
        "_configured_push_url",
        lambda **kwargs: ("https://github.com/example/repo.git", None),
    )
    monkeypatch.setattr(workflow, "_remote_branch_head", lambda **kwargs: (OLD, None))

    result = workflow.evaluate_git_governed_switch_policy("ai", "feature/safe")
    assert result["ok"] is True
    assert result["risk"] == "high"
    assert result["execution_arguments"]["checkpoint_required"] is True
    assert result["execution_arguments"]["checkpoint_message"] == "checkpoint before switching to feature/safe"


def test_dirty_switch_orders_stage_commit_push_switch(monkeypatch):
    monkeypatch.setattr(workflow, "_resolve_target", lambda repository: (_target(), None))
    monkeypatch.setattr(
        workflow,
        "_validate_local_target_branch",
        lambda **kwargs: ("feature/safe", TARGET, None),
    )
    monkeypatch.setattr(
        workflow,
        "_current_branch_and_head",
        lambda **kwargs: ("main", OLD, None),
    )
    monkeypatch.setattr(
        workflow,
        "_worktree_snapshot",
        lambda **kwargs: (
            "digest",
            {"dirty": True, "changed_paths": ["a.py"], "changed_count": 1, "conflicted_count": 0},
            None,
        ),
    )
    monkeypatch.setattr(
        workflow,
        "_push_snapshot",
        lambda *args, **kwargs: (
            _target(),
            {
                "branch": "main",
                "head": OLD,
                "remote": "origin",
                "remote_ref": "refs/heads/main",
                "remote_url": "https://github.com/example/repo.git",
                "remote_head": OLD,
            },
            None,
        ),
    )

    calls = []

    def stage(repository):
        calls.append("stage")
        return {"ok": True, "status": "success", "verification_ok": True, "mutation_performed": True, "files": ["a.py"]}

    def commit(repository, commit_message):
        calls.append("commit")
        return {"ok": True, "status": "success", "verification_ok": True, "commit": NEW, "files": ["a.py"]}

    def push(**kwargs):
        calls.append("push")
        return {"ok": True, "status": "success", "verification_ok": True}

    def switch(repository, branch_name):
        calls.append("switch")
        return {
            "ok": True,
            "status": "success",
            "repository": repository,
            "requested_branch": branch_name,
            "previous_branch": "main",
            "current_branch": branch_name,
            "switched_branch": branch_name,
            "switched_to_commit": TARGET,
            "mutation_performed": True,
            "files": [],
            "verification_ok": True,
            "verification_error": None,
        }

    monkeypatch.setattr(workflow, "workspace_git_stage_all", stage)
    monkeypatch.setattr(workflow, "workspace_git_commit", commit)
    monkeypatch.setattr(workflow, "_push_exact_snapshot", push)
    monkeypatch.setattr(
        workflow,
        "_run_git",
        lambda **kwargs: {"ok": True, "stdout": f"{TARGET}\n"},
    )
    monkeypatch.setattr(workflow, "_clean_switch", switch)

    result = workflow.workspace_git_governed_switch_branch(
        repository="ai",
        branch_name="feature/safe",
        expected_source_branch="main",
        expected_source_head=OLD,
        expected_target_head=TARGET,
        expected_worktree_digest="digest",
        checkpoint_required=True,
        expected_remote="origin",
        expected_remote_ref="refs/heads/main",
        expected_remote_url="https://github.com/example/repo.git",
        expected_remote_head=OLD,
        checkpoint_message="checkpoint before switching to feature/safe",
    )

    assert calls == ["stage", "commit", "push", "switch"]
    assert result["ok"] is True
    assert result["current_branch"] == "feature/safe"
    assert result["checkpoint_commit"] == NEW
    assert result["checkpoint_pushed"] is True


def test_push_verification_failure_blocks_switch(monkeypatch):
    monkeypatch.setattr(workflow, "_resolve_target", lambda repository: (_target(), None))
    monkeypatch.setattr(
        workflow,
        "_validate_local_target_branch",
        lambda **kwargs: ("feature/safe", TARGET, None),
    )
    monkeypatch.setattr(
        workflow,
        "_current_branch_and_head",
        lambda **kwargs: ("main", OLD, None),
    )
    monkeypatch.setattr(
        workflow,
        "_worktree_snapshot",
        lambda **kwargs: (
            "digest",
            {"dirty": True, "changed_paths": ["a.py"], "changed_count": 1, "conflicted_count": 0},
            None,
        ),
    )
    monkeypatch.setattr(
        workflow,
        "_push_snapshot",
        lambda *args, **kwargs: (
            _target(),
            {
                "branch": "main",
                "head": OLD,
                "remote": "origin",
                "remote_ref": "refs/heads/main",
                "remote_url": "https://github.com/example/repo.git",
                "remote_head": OLD,
            },
            None,
        ),
    )
    monkeypatch.setattr(
        workflow,
        "workspace_git_stage_all",
        lambda repository: {"ok": True, "verification_ok": True, "mutation_performed": True, "files": ["a.py"]},
    )
    monkeypatch.setattr(
        workflow,
        "workspace_git_commit",
        lambda repository, commit_message: {"ok": True, "verification_ok": True, "commit": NEW, "files": ["a.py"]},
    )
    monkeypatch.setattr(
        workflow,
        "_push_exact_snapshot",
        lambda **kwargs: {"ok": True, "verification_ok": False, "verification_error": "remote verify failed"},
    )

    def no_switch(*args, **kwargs):
        raise AssertionError("switch must not run")

    monkeypatch.setattr(workflow, "_clean_switch", no_switch)

    result = workflow.workspace_git_governed_switch_branch(
        repository="ai",
        branch_name="feature/safe",
        expected_source_branch="main",
        expected_source_head=OLD,
        expected_target_head=TARGET,
        expected_worktree_digest="digest",
        checkpoint_required=True,
        expected_remote="origin",
        expected_remote_ref="refs/heads/main",
        expected_remote_url="https://github.com/example/repo.git",
        expected_remote_head=OLD,
        checkpoint_message="checkpoint before switching to feature/safe",
    )
    assert result["ok"] is False
    assert result["checkpoint_commit"] == NEW
    assert result["checkpoint_pushed"] is False
