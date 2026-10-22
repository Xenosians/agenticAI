"""Acceptance completion must reflect executed, declared coverage."""
import json

import pytest

from scripts import tool_matrix as matrix


CASE = ("read", "read_tool", "Read a fixture", None)


def summary(verdict, *, tools=("read_tool",), full_scope=True):
    return matrix.acceptance_summary({"read": {"tool": "read_tool", "verdict": verdict}},
                                     [CASE], registered_tools=tools, full_scope=full_scope)


@pytest.mark.parametrize("verdict", ["blocked", "skipped", "plan_limited"])
def test_unexecuted_or_unavailable_cases_are_incomplete(verdict):
    result = summary(verdict)
    assert result["overall"] == "INCOMPLETE" and result["exit_code"] == 2
    assert result["all_registered_tools_passed"] is False


def test_only_nonempty_complete_success_can_exit_zero():
    result = summary("pass")
    assert result["overall"] == "PASS" and result["exit_code"] == 0
    assert result["all_registered_tools_passed"] is True
    empty = matrix.acceptance_summary({}, [], registered_tools=["read_tool"], full_scope=True)
    assert empty["exit_code"] == 2 and empty["all_registered_tools_passed"] is False


def test_failures_remain_failures():
    assert summary("fail")["exit_code"] == 1
    assert summary("fail")["overall"] == "FAIL"


def test_missing_invalid_or_wrong_tool_rows_do_not_pass():
    for rows in [{}, {"read": {"tool": "other", "verdict": "pass"}},
                 {"read": {"tool": "read_tool", "verdict": "not_a_verdict"}}]:
        result = matrix.acceptance_summary(rows, [CASE], registered_tools=["read_tool"], full_scope=True)
        assert result["exit_code"] == 2 and result["invalid_or_missing_cases"] == ["read"]


def test_full_scope_requires_all_registered_tools_while_selected_scope_is_honest():
    full = summary("pass", tools=["read_tool", "uncovered"])
    assert full["exit_code"] == 2 and full["tools_outside_selected_coverage"] == ["uncovered"]
    partial = summary("pass", tools=["read_tool", "uncovered"], full_scope=False)
    assert partial["exit_code"] == 0 and partial["selected_scope_passed"] is True
    assert partial["all_registered_tools_passed"] is False


def test_code_changes_during_a_run_invalidate_acceptance():
    result = matrix.acceptance_summary({"read": {"tool": "read_tool", "verdict": "pass"}},
                                     [CASE], registered_tools=["read_tool"], full_scope=True,
                                     source_unchanged=False)
    assert result["exit_code"] == 2 and result["overall"] == "INCOMPLETE"
    assert result["all_registered_tools_passed"] is False


def test_unknown_and_empty_selection_are_configuration_errors():
    with pytest.raises(ValueError, match="Unknown case"):
        matrix.select_matrix_cases([CASE], ["misspelled"])
    with pytest.raises(ValueError, match="at least one"):
        matrix.select_matrix_cases([], [])
    assert matrix.select_matrix_cases([CASE], []) == [CASE]


def test_resume_preserves_original_run_and_refuses_stale_or_legacy_evidence(tmp_path):
    path = tmp_path / "report.json"
    identity = {"ai": {"head": "fixture", "source_sha256": "first"}}
    original = matrix.load_matrix_report(path, resume=False, identity=identity, run="10191234", project="FIX")
    original["results"] = {"read": {"tool": "read_tool", "verdict": "pass"}}
    path.write_text(json.dumps(original))
    resumed = matrix.load_matrix_report(path, resume=True, identity=identity, run="10191235", project="FIX")
    assert resumed["run"] == "10191234" and resumed["results"] == original["results"]
    for change in [{"schema": "live-tool-matrix.v1"}, {"source_identity": {}}, {"project": "other"}]:
        path.write_text(json.dumps({**original, **change}))
        with pytest.raises(ValueError, match="matching source"):
            matrix.load_matrix_report(path, resume=True, identity=identity, run="10191235", project="FIX")


def test_existing_report_cannot_be_silently_overwritten_and_resume_requires_report(tmp_path):
    path = tmp_path / "report.json"
    with pytest.raises(ValueError, match="missing"):
        matrix.load_matrix_report(path, resume=True, identity={}, run="10191234", project="FIX")
    path.write_text("{}")
    with pytest.raises(ValueError, match="already exists"):
        matrix.load_matrix_report(path, resume=False, identity={}, run="10191234", project="FIX")


@pytest.mark.parametrize("selection", [[], ["does-not-exist"]])
def test_cli_rejects_invalid_scope_before_source_or_credential_access(tmp_path, monkeypatch, selection):
    def forbidden():
        raise AssertionError("Invalid scope must fail before source/credential inspection")
    monkeypatch.setattr(matrix, "source_identity", forbidden)
    monkeypatch.setattr(matrix.sys, "argv", ["tool_matrix", "--report", str(tmp_path / "new.json"),
                                          "--only", *selection])
    with pytest.raises(SystemExit) as result:
        matrix.main()
    assert result.value.code == 2
    assert not (tmp_path / "new.json").exists()


def test_report_cannot_accumulate_results_for_a_different_account():
    report = {}
    matrix.bind_principal(report, "first@example.invalid")
    matrix.bind_principal(report, "FIRST@example.invalid")
    with pytest.raises(ValueError, match="account changed"):
        matrix.bind_principal(report, "second@example.invalid")


def test_lost_submission_is_recorded_and_resume_never_creates_a_replacement(tmp_path, monkeypatch):
    calls = []
    class Client:
        def __init__(self, *args, **kwargs):
            pass
        def preflight(self):
            pass
        def login(self, *args):
            pass
        def create_chat(self, name):
            calls.append("chat")
            return "observed-chat"
        def create_job(self, **kwargs):
            calls.append("job")
            raise RuntimeError("Acknowledgement lost after dispatch")
        def close(self):
            pass
    monkeypatch.setattr(matrix, "BrowserPathClient", Client)
    monkeypatch.setattr(matrix, "source_identity", lambda: {"ai": {"head": "test-source"}})
    monkeypatch.setattr(matrix, "build_cases", lambda *args: [CASE])
    monkeypatch.setattr(matrix, "dotenv_values", lambda *args: {
        "ACCEPTANCE_EMAIL": "operator@example.invalid", "ACCEPTANCE_PASSWORD": "fixture-only"})
    monkeypatch.setattr(matrix, "discover_fixtures", lambda *args: None)
    monkeypatch.setattr(matrix, "atlassian_admin_configured", lambda: False)
    path = tmp_path / "report.json"
    args = ["tool_matrix", "--report", str(path), "--only", "read"]
    monkeypatch.setattr(matrix.sys, "argv", args)
    assert matrix.main() == 1
    saved = json.loads(path.read_text())
    assert saved["results"]["read"]["chat_id"] == "observed-chat"
    assert saved["results"]["read"]["verdict"] == "fail"
    monkeypatch.setattr(matrix.sys, "argv", [*args, "--resume"])
    assert matrix.main() == 1
    assert calls == ["chat", "job"]


def test_fixture_identity_is_stable_for_original_run():
    original = matrix.build_cases("10191234", "FIX")
    same = matrix.build_cases("10191234", "FIX")
    new = matrix.build_cases("10191235", "FIX")
    assert same == original and new != original
    assert len({row[0] for row in original}) == len(original)


def test_source_identity_detects_uncommitted_source_changes_without_collecting_env(tmp_path, monkeypatch):
    ai = tmp_path / "agenticaiPersonal" / "itsm-agent"
    repos = [ai, tmp_path / "agenticBackend" / "itsm_backend", tmp_path / "agenticFrontend"]
    for repo in repos:
        repo.mkdir(parents=True)
        (repo / "source.py").write_text("value = 1\n")
        (repo / ".env").write_text("not-collected")
    monkeypatch.setattr(matrix, "PROJECT_ROOT", ai)
    def git(args, **kwargs):
        return "same-head\n" if args[1] == "rev-parse" else b"source.py\0.env\0"
    monkeypatch.setattr(matrix.subprocess, "check_output", git)
    before = matrix.source_identity()
    (ai / ".env").write_text("different-private-value")
    assert matrix.source_identity() == before
    (ai / "source.py").write_text("value = 2\n")
    after = matrix.source_identity()
    assert after["ai"]["head"] == before["ai"]["head"]
    assert after["ai"]["source_sha256"] != before["ai"]["source_sha256"]
