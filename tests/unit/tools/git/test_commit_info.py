"""One commit's details: a fixed, read-only git command on a validated commit id."""
from pathlib import Path

import pytest

from services.git_repositories import GitRepositoryTarget
import tools.git as git_tools
from tools.git.catalog import GIT_TOOLS
from tools.semantic_catalog import CAPABILITY_SEMANTICS

FULL = "69de1ffbafcf19f0fe1284116866de74909a7916"
PARENT = "a077b126ec4d2e14ec3fbf8f1e286d25acf6215d"
SHOW_OUTPUT = (
    f"{FULL}\0{PARENT}\0Xenosians\0" "2026-11-01T21:31:33+07:00\0"
    "Verify the directory's certificate before binding\0"
    "ldap3 accepts any certificate unless it is told otherwise.\n\nSecond paragraph.\n\0"
    "\n14\t7\tservices/lua_adapters/profile.py\n-\t-\tdocs/diagram.png\n127\t0\ttests/test_tls.py\n"
)


class Runner:
    def __init__(self, result):
        self.result, self.calls = result, []

    def __call__(self, executable, args, cwd=None, timeout_seconds=60):
        self.calls.append({"executable": executable, "args": args, "cwd": cwd})
        return self.result


@pytest.fixture
def repository(monkeypatch):
    target = GitRepositoryTarget(name="ai", path=Path("/approved/ai"))
    monkeypatch.setattr(git_tools, "resolve_git_repository", lambda repository=None: target)
    return target


def run(monkeypatch, commit, *, stdout=SHOW_OUTPUT, ok=True, stderr=""):
    runner = Runner({"ok": ok, "status": "success" if ok else "error", "stdout": stdout, "stderr": stderr})
    monkeypatch.setattr(git_tools, "run_trusted_process", runner)
    return git_tools.workspace_git_commit_info(commit=commit, repository="ai"), runner


def test_reads_one_commit_with_a_fixed_command(monkeypatch, repository):
    result, runner = run(monkeypatch, "69DE1FF")
    assert runner.calls == [{"executable": "git", "cwd": str(repository.path), "args": [
        "show", "--no-color", "--no-ext-diff", "--no-textconv", "--no-notes", "--diff-merges=first-parent",
        "--numstat", "--format=%H%x00%P%x00%an%x00%aI%x00%s%x00%b%x00", "--end-of-options", "69de1ff^{commit}", "--"]}]
    assert result["ok"] is True and result["repository"] == "ai"
    assert result["commit"] == {
        "hash": FULL, "parents": [PARENT], "author": "Xenosians", "date": "2026-11-01T21:31:33+07:00",
        "subject": "Verify the directory's certificate before binding",
        "body": "ldap3 accepts any certificate unless it is told otherwise.\n\nSecond paragraph."}
    assert result["files"] == [
        {"path": "services/lua_adapters/profile.py", "added": 14, "deleted": 7, "binary": False},
        {"path": "docs/diagram.png", "added": None, "deleted": None, "binary": True},
        {"path": "tests/test_tls.py", "added": 127, "deleted": 0, "binary": False}]
    assert result["file_count"] == 3 and result["added"] == 141 and result["deleted"] == 7
    assert result["truncated"] is False


def test_a_merge_commit_lists_both_parents(monkeypatch, repository):
    merge = SHOW_OUTPUT.replace(f"\0{PARENT}\0", f"\0{PARENT} {'b' * 40}\0")
    result, _ = run(monkeypatch, FULL)
    assert result["commit"]["parents"] == [PARENT]
    result, _ = run(monkeypatch, FULL, stdout=merge)
    assert result["commit"]["parents"] == [PARENT, "b" * 40]


@pytest.mark.parametrize("commit", ["HEAD", "main", "abc", "-p", "--output=/tmp/x", "abc1234^", "abc1234..def5678",
                                    "abc1234 def5678", "g" * 7, "a" * 41, "", None, 1234567])
def test_anything_but_a_commit_id_is_refused_before_git_runs(monkeypatch, repository, commit):
    result, runner = run(monkeypatch, commit)
    assert result["ok"] is False and result["status"] == "denied" and runner.calls == []


def test_an_unknown_commit_is_reported_not_guessed(monkeypatch, repository):
    result, _ = run(monkeypatch, "deadbeefdeadbeef", ok=False, stdout="",
                    stderr="fatal: bad revision 'deadbeefdeadbeef^{commit}'")
    assert result == {"ok": False, "status": "not_found", "repository": "ai",
                      "error": "No commit deadbeefdeadbeef in ai."}


def test_unexpected_git_output_is_refused(monkeypatch, repository):
    result, _ = run(monkeypatch, FULL, stdout="not the requested format")
    assert result["ok"] is False and result["status"] == "error"


def test_a_long_file_list_is_cut_and_marked(monkeypatch, repository):
    many = SHOW_OUTPUT.split("\0\n")[0] + "\0\n" + "".join(f"1\t0\tf{index}.txt\n" for index in range(250))
    result, _ = run(monkeypatch, FULL, stdout=many + "\n...[output truncated]")
    assert len(result["files"]) == git_tools.MAX_COMMIT_FILES and result["file_count"] == 250
    assert result["truncated"] is True


def test_capability_is_a_grounded_read_without_approval():
    tool = GIT_TOOLS["workspace_git_commit_info"]
    assert tool["risk"] == "read" and tool["requires_approval"] is False
    assert set(tool["parameters"]) == {"repository", "commit"}
    assert tool["required_arguments"] == ["commit"]
    assert set(tool["grounded_arguments"]) == {"repository", "commit"}
    assert CAPABILITY_SEMANTICS["workspace_git_commit_info"]["effect"] == "read"


def test_the_summary_reads_as_sentences():
    text = GIT_TOOLS["workspace_git_commit_info"]["result_formatter"]({
        "ok": True, "repository": "ai", "commit": {"hash": FULL, "parents": [PARENT], "author": "Xenosians",
                                                   "date": "2026-11-01T21:31:33+07:00", "subject": "Fix", "body": ""},
        "files": [{"path": "a.py", "added": 2, "deleted": 1, "binary": False}],
        "file_count": 1, "added": 2, "deleted": 1, "truncated": False})
    assert text == ("Commit 69de1ffbafcf in ai by Xenosians on 2026-11-01T21:31:33+07:00: Fix\n"
                    "1 file changed, 2 lines added, 1 removed:\n- a.py (+2 -1)")
