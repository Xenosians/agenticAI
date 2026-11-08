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


def metric(stage, adapter, *, role=None, outcome="success", repair_of=None, operation="generate"):
    return {"operation": operation, "stage": stage, "repair_of": repair_of, "adapter": adapter,
            "outcome": outcome, "role": role}


LOCAL = {"adapter": "local", "networking": {"adapter": "local"}}
OUTBOUND = {"adapter": "http", "networking": {"adapter": "local"}}


def test_a_case_proves_a_profile_only_when_its_steps_ran_on_that_profile():
    local_run = [metric("CONTROL_ROUTING", "local"), metric("SEMANTIC_ROUTING", "local"),
                 metric("SPECIALIST_REASONING", "local", role="jira-specialist"), metric("SYNTHESIS", "local")]
    assert matrix.inference_evidence(local_run, LOCAL) == {
        "adapters": ["local"],
        "stages": ["CONTROL_ROUTING", "SEMANTIC_ROUTING", "SPECIALIST_REASONING", "SYNTHESIS"]}
    # A read the routing contract fully determines has no specialist step (decision 11).
    determined = [metric("CONTROL_ROUTING", "local"), metric("SYNTHESIS", "local")]
    assert matrix.inference_evidence(determined, LOCAL)["adapters"] == ["local"]
    with pytest.raises(matrix.AcceptanceFailure, match="SPECIALIST_REASONING ran on http, not local"):
        matrix.inference_evidence([metric("SPECIALIST_REASONING", "http", role="jira-specialist")], LOCAL)
    with pytest.raises(matrix.AcceptanceFailure, match="SEMANTIC_ROUTING ran on local, not http"):
        matrix.inference_evidence([metric("SEMANTIC_ROUTING", "local"),
                                   metric("SPECIALIST_REASONING", "http")], OUTBOUND)
    # The networking specialist follows its own assignment; a repair counts as the step it repairs.
    # Host routing always runs on the local model, whatever the profile.
    networking = [metric("CONTROL_ROUTING", "local"), metric("SEMANTIC_ROUTING", "http"),
                  metric("REPAIR", "local", role="networking-specialist", repair_of="NETWORKING")]
    assert matrix.inference_evidence(networking, OUTBOUND)["adapters"] == ["http", "local"]
    # Failed steps, warm-ups and host routing prove nothing about the chosen profile.
    with pytest.raises(matrix.AcceptanceFailure, match="no step ran on the chosen profile"):
        matrix.inference_evidence([metric("SPECIALIST_REASONING", "http", outcome="error"),
                                   metric("SYNTHESIS", "http", operation="warm"),
                                   metric("CONTROL_ROUTING", "local")], OUTBOUND)


@pytest.mark.parametrize("profile", [["--model-id", "connection"], ["--adapter", "local", "--model-id", "connection"],
                                     ["--adapter", "http"], ["--networking", "local"]])
def test_an_incomplete_profile_is_refused_before_source_or_credential_access(tmp_path, monkeypatch, profile):
    def forbidden():
        raise AssertionError("An invalid profile must fail before source/credential inspection")
    monkeypatch.setattr(matrix, "source_identity", forbidden)
    monkeypatch.setattr(matrix.sys, "argv", ["tool_matrix", "--report", str(tmp_path / "new.json"), *profile])
    with pytest.raises(SystemExit) as result:
        matrix.main()
    assert result.value.code == 2
    assert not (tmp_path / "new.json").exists()


def test_every_case_runs_inside_the_chosen_profile_and_the_account_is_restored(tmp_path, monkeypatch):
    events, restoration = [], {"status": "restored"}
    fingerprint = {"schema": "acceptance-inference-profile.v1", "sha256": "fixture", "adapter": "local",
                   "networking": {"adapter": "local"}}

    class Scope:
        def __init__(self, client, *, adapter, model_id, networking):
            events.append(("scope", adapter, model_id, networking))
            self.fingerprint = fingerprint
            self.original = type("Snapshot", (), {"selected": {"model_id": "own", "dataset_id": None},
                                                  "roles": {}})()

        def __enter__(self):
            events.append("selected")
            return self

        def assert_unchanged(self):
            events.append("checked")
            return self.fingerprint

        def restore(self):
            events.append("restored")
            return restoration

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def preflight(self):
            pass

        def login(self, *args):
            pass

        def create_chat(self, name):
            return "observed-chat"

        def create_job(self, **kwargs):
            events.append("job")
            return {"job_id": "a" * 32}

        def wait_for_terminal(self, job_id, poll_seconds):
            return {"job_id": job_id, "status": "completed", "attempts": 1, "message": CASE[2],
                    "result": {"results": [{"proposed_tool": "read_tool",
                                            "tool_result": {"ok": True, "status": "success"}}]}}

        def get_history(self, chat_id):
            return {}

        def close(self):
            pass

    monkeypatch.setattr(matrix, "AcceptanceProfileScope", Scope)
    monkeypatch.setattr(matrix, "BrowserPathClient", Client)
    monkeypatch.setattr(matrix, "job_inference_metrics", lambda client, job_id, attempt: [
        metric("CONTROL_ROUTING", "local"), metric("SPECIALIST_REASONING", "local", role="jira-specialist")])
    monkeypatch.setattr(matrix, "_assert_history_roundtrip", lambda *args, **kwargs: None)
    monkeypatch.setattr(matrix, "source_identity", lambda: {"ai": {"head": "test-source"}})
    monkeypatch.setattr(matrix, "build_cases", lambda *args: [CASE])
    monkeypatch.setattr(matrix, "dotenv_values", lambda *args: {
        "ACCEPTANCE_EMAIL": "operator@example.invalid", "ACCEPTANCE_PASSWORD": "fixture-only"})
    monkeypatch.setattr(matrix, "discover_fixtures", lambda *args: None)
    monkeypatch.setattr(matrix, "atlassian_admin_configured", lambda: False)
    path = tmp_path / "report.json"
    monkeypatch.setattr(matrix.sys, "argv", ["tool_matrix", "--report", str(path), "--only", "read",
                                             "--adapter", "local", "--networking", "local"])
    assert matrix.main() == 0
    assert events == [("scope", "local", None, "local"), "selected", "checked", "job", "restored"]
    saved = json.loads(path.read_text())
    assert saved["inference_profile"] == fingerprint
    assert saved["original_selection"] == {"selected": {"model_id": "own", "dataset_id": None}, "roles": {}}
    assert saved["inference_restoration"] == {"status": "restored"}
    assert saved["results"]["read"]["verdict"] == "pass"
    assert saved["results"]["read"]["inference"]["adapters"] == ["local"]

    # An account left on the test selection must never look like a clean run.
    restoration = {"status": "reconciliation_required"}
    monkeypatch.setattr(matrix.sys, "argv", ["tool_matrix", "--report", str(tmp_path / "second.json"),
                                             "--only", "read", "--adapter", "local", "--networking", "local"])
    assert matrix.main() == 1


def test_every_registered_tool_has_a_live_matrix_case():
    from mcp_server import register_capability_signatures
    from tools.registry import TOOLS

    register_capability_signatures()
    covered = {case[1] for case in matrix.build_cases("10191234", "FIX")}
    assert sorted(set(TOOLS) - covered) == []


def test_a_provider_that_is_not_set_up_is_reported_as_such_not_as_a_failure():
    message = "Proxmox VE is not set up on this Janus host. Missing settings: PROXMOX_BASE_URL."
    # A failed step carries the provider's code and message on the step itself.
    job = {"result": {"results": [{"agent_name": "proxmox-specialist", "proposed_tool": "proxmox_nodes",
                                   "status": "error", "outcome_code": "tool_execution_error",
                                   "error_code": "not_configured", "error": message}]}}
    assert matrix.provider_not_configured(job) == message
    other = {"result": {"results": [{"status": "error", "error_code": "provider_error", "error": "x"}]}}
    assert matrix.provider_not_configured(other) is None
    assert matrix.provider_not_configured({}) is None
    result = matrix.acceptance_summary({"read": {"tool": "read_tool", "verdict": "not_configured"}},
                                       [CASE], registered_tools=("read_tool",), full_scope=True)
    assert result["overall"] == "INCOMPLETE" and result["exit_code"] == 2
    assert result["not_configured"] == 1 and result["all_registered_tools_passed"] is False
