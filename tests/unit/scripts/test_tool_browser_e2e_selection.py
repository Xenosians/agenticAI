"""Browser acceptance preserves evidence and never blindly repeats uncertain work."""
from copy import deepcopy
from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest

from scripts import tool_browser_e2e as harness


def test_harness_imports_its_shared_matrix_helpers():
    # A broken import here once stopped the 84-tool browser run without failing any test.
    assert harness.select_matrix_cases is not None
    assert harness.acceptance_summary is not None


CASE = ("fixture-read", "fixture_tool", "Read the fixture", None)
SOURCE = {"ai": {"head": "fixture", "source_sha256": "same-source"}}


@pytest.mark.parametrize("verdict", ["pass", "plan_limited", "fail", "blocked", "skipped"])
def test_resume_never_automatically_reruns_any_recorded_case(verdict):
    report = {"results": {CASE[0]: {"tool": CASE[1], "verdict": verdict}}}
    next_case = ("untouched", "other", "Read another fixture", None)
    assert harness.unattempted_cases(report, [CASE, next_case]) == [next_case]


@pytest.mark.parametrize("only", [[], ["no-such-case"], ["ticket-get", "misspelled-case"]])
def test_unknown_case_ids_are_rejected_before_the_browser_starts(monkeypatch, only):
    monkeypatch.setattr("sys.argv", ["tool_browser_e2e.py", "--only", *only])
    monkeypatch.setattr(harness, "sync_playwright",
                        lambda: pytest.fail("the browser must not start for an invalid selection"))
    monkeypatch.setattr(harness, "source_identity", lambda: pytest.fail("invalid scope must not read repositories"))
    monkeypatch.setattr(harness, "dotenv_values", lambda _: pytest.fail("invalid scope must not read credentials"))
    with pytest.raises(SystemExit) as exit_info:
        harness.main()
    assert exit_info.value.code == 2


def arguments(tmp_path):
    return SimpleNamespace(frontend_url="http://127.0.0.1:8080/", backend_url="http://127.0.0.1:4000",
                           login_env=tmp_path / "login.env", timeout=1)


def test_browser_report_cannot_reuse_api_legacy_changed_source_or_other_app_evidence(tmp_path):
    path = tmp_path / "browser.json"
    context = harness.execution_context(arguments(tmp_path))
    original = harness.load_matrix_report(path, resume=False, identity=SOURCE, run="10191234", project="FIX",
                                          schema=harness.BROWSER_REPORT_SCHEMA, execution_context=context)
    original["results"][CASE[0]] = {"tool": CASE[1], "verdict": "pass"}
    path.write_text(json.dumps(original))
    resumed = harness.load_matrix_report(path, resume=True, identity=SOURCE, run="10220000", project="FIX",
                                         schema=harness.BROWSER_REPORT_SCHEMA, execution_context=context)
    assert resumed["run"] == "10191234" and resumed["results"] == original["results"]
    for patch in [{"schema": "tool-browser-e2e.v1"}, {"schema": harness.matrix.REPORT_SCHEMA},
                  {"source_identity": {}}, {"project": "OTHER"},
                  {"execution_context": {**context, "backend_url": "http://127.0.0.1:9999"}}]:
        path.write_text(json.dumps({**original, **patch}))
        with pytest.raises(ValueError, match="matching source"):
            harness.load_matrix_report(path, resume=True, identity=SOURCE, run="10220000", project="FIX",
                                        schema=harness.BROWSER_REPORT_SCHEMA, execution_context=context)


def test_new_report_refuses_overwriting_existing_evidence(tmp_path):
    path = tmp_path / "browser.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="already exists"):
        harness.load_matrix_report(path, resume=False, identity=SOURCE, run="10191234", project="FIX",
                                    schema=harness.BROWSER_REPORT_SCHEMA)


def test_execution_context_does_not_accept_credentials_in_urls(tmp_path):
    args = arguments(tmp_path)
    for value in ["https://user:secret@example.test", "https://example.test?token=private", "file:///tmp/app"]:
        args.backend_url = value
        with pytest.raises(ValueError, match="without credentials"):
            harness.execution_context(args)


def test_principal_binding_refuses_switching_accounts_without_storing_email():
    report = {}
    harness.bind_principal(report, "first@example.test")
    assert "first@example.test" not in json.dumps(report)
    harness.bind_principal(report, "FIRST@example.test")
    with pytest.raises(ValueError, match="account changed"):
        harness.bind_principal(report, "other@example.test")


@pytest.mark.parametrize("verdict,expected", [("fail", 1), ("blocked", 2), ("pass", 0)])
def test_resume_only_recorded_cases_never_reads_login_or_starts_browser(tmp_path, monkeypatch, verdict, expected):
    args = arguments(tmp_path)
    path = tmp_path / "browser.json"
    report = harness.load_matrix_report(path, resume=False, identity=SOURCE, run="10191234", project=harness.matrix.PROJECT,
                                        schema=harness.BROWSER_REPORT_SCHEMA, execution_context=harness.execution_context(args))
    report["results"][CASE[0]] = {"tool": CASE[1], "verdict": verdict, "job_id": "original-job"}
    path.write_text(json.dumps(report))
    monkeypatch.setattr(harness, "source_identity", lambda: SOURCE)
    monkeypatch.setattr(harness, "build_cases", lambda *_: [CASE])
    monkeypatch.setattr(harness, "TOOLS", {CASE[1]: {}})
    monkeypatch.setattr(harness, "dotenv_values", lambda _: pytest.fail("must not read credentials when no new work exists"))
    monkeypatch.setattr(harness, "sync_playwright", lambda: pytest.fail("must not start a browser to replay recorded work"))
    monkeypatch.setattr(harness.matrix, "RUN", "10220000")
    monkeypatch.setattr("sys.argv", ["tool_browser_e2e.py", "--resume", "--report", str(path),
                                     "--login-env", str(args.login_env)])
    assert harness.main() == expected
    saved = json.loads(path.read_text())
    assert saved["run"] == "10191234" and saved["results"][CASE[0]]["job_id"] == "original-job"
    assert saved["selection"] == {"scope": "full", "cases": [CASE[0]]}


def test_inflight_evidence_is_not_certified_until_source_is_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "TOOLS", {CASE[1]: {}})
    path = tmp_path / "browser.json"
    report = {"results": {CASE[0]: {"tool": CASE[1], "verdict": "pass"}}}
    harness.save_browser_report(report, path, [CASE], full_scope=True)
    assert json.loads(path.read_text())["acceptance"]["exit_code"] == 2
    harness.save_browser_report(report, path, [CASE], full_scope=True, source_unchanged=True)
    assert json.loads(path.read_text())["acceptance"]["exit_code"] == 0


def test_socket_evidence_excludes_historical_frames_and_supports_persistent_sockets():
    traffic = SimpleNamespace(sockets=[{"frames_in": 9}])
    assert harness.socket_evidence(traffic, [9]) == {"opened": 0, "frames_in": 0}
    traffic.sockets[0]["frames_in"] += 2
    traffic.sockets.append({"frames_in": 1})
    assert harness.socket_evidence(traffic, [9]) == {"opened": 1, "frames_in": 3}


def test_lost_submission_response_preserves_predispatch_record_and_observed_identities(tmp_path, monkeypatch):
    row = {"verdict": "blocked"}
    checkpoints = []
    monkeypatch.setattr(harness, "new_chat", lambda *_: "original-chat")
    def lost_response(page, message, timeout_ms, *, on_submission):
        assert checkpoints[-1]["stage"] == "submitting"
        assert checkpoints[-1]["submission_outcome"] == "outcome_unknown"
        on_submission({"chat_id": "original-chat", "submission_id": "original-submission"})
        raise TimeoutError("submission response lost")
    monkeypatch.setattr(harness, "send", lost_response)
    with pytest.raises(TimeoutError):
        harness.run_case(None, SimpleNamespace(sockets=[]), arguments(tmp_path), CASE, {}, row=row,
                          checkpoint=lambda: checkpoints.append(deepcopy(row)))
    assert checkpoints[-1]["submission_id"] == "original-submission"
    assert row["chat_id"] == "original-chat" and row["reconciliation_required"] is True
    assert row["verdict"] != "pass"


def test_timeout_after_job_acknowledgement_keeps_job_identity(tmp_path, monkeypatch):
    row = {"verdict": "blocked"}
    checkpoints = []
    monkeypatch.setattr(harness, "new_chat", lambda *_: "original-chat")
    monkeypatch.setattr(harness, "send", lambda *_, **__: ("original-job", "original-chat"))
    def stalled(*_):
        raise TimeoutError("job still running")
    traffic = SimpleNamespace(sockets=[], wait=stalled)
    with pytest.raises(TimeoutError):
        harness.run_case(None, traffic, arguments(tmp_path), CASE, {}, row=row,
                          checkpoint=lambda: checkpoints.append(deepcopy(row)))
    assert checkpoints[-1]["job_id"] == "original-job"
    assert row["submission_outcome"] == "acknowledged" and row["reconciliation_required"] is True


def test_a_job_acknowledged_in_another_chat_is_never_verified(tmp_path, monkeypatch):
    row = {"verdict": "blocked"}
    checkpoints = []
    monkeypatch.setattr(harness, "new_chat", lambda *_: "selected-chat")
    monkeypatch.setattr(harness, "send", lambda *_, **__: ("original-job", "other-chat"))
    def unexpected(*_):
        raise AssertionError("a job in another chat must not be awaited")
    with pytest.raises(AssertionError, match="different chat than the selected empty chat"):
        harness.run_case(None, SimpleNamespace(sockets=[], wait=unexpected), arguments(tmp_path), CASE, {},
                          row=row, checkpoint=lambda: checkpoints.append(deepcopy(row)))
    # The acknowledged job stays recorded for reconciliation; nothing passes.
    assert checkpoints[-1]["job_id"] == "original-job" and checkpoints[-1]["chat_id"] == "other-chat"
    assert row["reconciliation_required"] is True and row["verdict"] != "pass"


@pytest.mark.parametrize("frames,expected", [(0, "fail"), (1, "pass")])
def test_provider_success_without_live_update_evidence_does_not_pass(tmp_path, monkeypatch, frames, expected):
    row = {"verdict": "blocked"}
    traffic = SimpleNamespace(sockets=[{"frames_in": 3}])
    final = {"job_id": "job", "chat_id": "chat", "status": "completed"}
    def completed(*_):
        traffic.sockets[0]["frames_in"] += frames
        return final
    traffic.wait = completed
    monkeypatch.setattr(harness, "new_chat", lambda *_: "chat")
    monkeypatch.setattr(harness, "send", lambda *_, **__: ("job", "chat"))
    monkeypatch.setattr(harness, "chat_history", lambda *_: {})
    monkeypatch.setattr(harness, "_assert_history_roundtrip", lambda *_ , **__: None)
    monkeypatch.setattr(harness, "worker_evidence", lambda _: {"proposed_tool": CASE[1], "tool_result": {"ok": True, "status": "executed"}})
    monkeypatch.setattr(harness, "record_created", lambda *_: None)
    if expected == "fail":
        with pytest.raises(AssertionError, match="WebSocket frame"):
            harness.run_case(None, traffic, arguments(tmp_path), CASE, {}, row=row, checkpoint=lambda: None)
        assert row["verdict"] != "pass"
    else:
        assert harness.run_case(None, traffic, arguments(tmp_path), CASE, {}, row=row,
                                 checkpoint=lambda: None)["verdict"] == "pass"
    assert row["websocket"]["frames_in"] == frames


def test_main_persists_intent_before_dispatch_and_keeps_unknown_job_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "browser.json"
    page = SimpleNamespace(screenshot=lambda **_: None)
    browser = SimpleNamespace(new_context=lambda **_: SimpleNamespace(new_page=lambda: page), close=lambda: None)
    playwright = SimpleNamespace(chromium=SimpleNamespace(launch=lambda **_: browser))
    monkeypatch.setattr(harness, "sync_playwright", lambda: nullcontext(playwright))
    monkeypatch.setattr(harness, "BrowserTraffic", lambda _: SimpleNamespace(sockets=[]))
    monkeypatch.setattr(harness, "sign_in", lambda *args: None)
    monkeypatch.setattr(harness, "source_identity", lambda: SOURCE)
    monkeypatch.setattr(harness, "build_cases", lambda *_: [CASE])
    monkeypatch.setattr(harness, "TOOLS", {CASE[1]: {}})
    monkeypatch.setattr(harness, "discover_fixtures", lambda _: None)
    monkeypatch.setattr(harness, "atlassian_admin_configured", lambda: False)
    monkeypatch.setattr(harness, "host_unset_provider_families", lambda: {})
    monkeypatch.setattr(harness, "dotenv_values", lambda _: {"ACCEPTANCE_EMAIL": "test@example.test", "ACCEPTANCE_PASSWORD": "fixture-only"})
    monkeypatch.setattr(harness.matrix, "RUN", "10220000")
    def interrupted(*args, row, checkpoint, evidence=None, unset_families=None):
        saved = json.loads(path.read_text())
        assert saved["results"][CASE[0]]["stage"] == "prepared"
        assert saved["results"][CASE[0]]["verdict"] == "blocked"
        row.update(job_id="original-job", submission_id="original-submission", reconciliation_required=True)
        checkpoint()
        raise TimeoutError("lost durable completion response")
    monkeypatch.setattr(harness, "run_case", interrupted)
    command = ["tool_browser_e2e.py", "--report", str(path), "--login-env", str(tmp_path / "login.env")]
    monkeypatch.setattr("sys.argv", command)
    assert harness.main() == 1
    saved = json.loads(path.read_text())
    assert saved["results"][CASE[0]]["job_id"] == "original-job"
    assert saved["results"][CASE[0]]["submission_id"] == "original-submission"
    assert saved["results"][CASE[0]]["reconciliation_required"] is True
    monkeypatch.setattr(harness, "dotenv_values", lambda _: pytest.fail("resume must not create another job"))
    monkeypatch.setattr(harness, "run_case", lambda *_ , **__: pytest.fail("unknown case must not replay"))
    monkeypatch.setattr("sys.argv", command + ["--resume"])
    assert harness.main() == 1


def test_send_records_request_identity_even_when_its_response_is_lost():
    callbacks = {}
    observed = []
    request = SimpleNamespace(url="http://127.0.0.1:4000/api/v1/jobs", method="POST",
                              post_data=json.dumps({"chat_id": "chat", "submission_id": "submission", "message": "private text"}))
    button = SimpleNamespace(is_enabled=lambda: True, click=lambda: callbacks["request"](request))
    class LostResponse:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            raise TimeoutError("response lost after dispatch")
    page = SimpleNamespace(
        locator=lambda _: SimpleNamespace(fill=lambda _: None),
        get_by_role=lambda *_, **__: button,
        on=lambda name, callback: callbacks.update({name: callback}),
        remove_listener=lambda name, _: callbacks.pop(name),
        expect_response=lambda *_, **__: LostResponse(),
    )
    with pytest.raises(TimeoutError):
        harness.send(page, "Read the fixture", 1000, on_submission=observed.append)
    assert observed == [{"chat_id": "chat", "submission_id": "submission"}]
    assert callbacks == {}


FINGERPRINT = {"schema": "acceptance-inference-profile.v1", "sha256": "fixture", "adapter": "http",
               "networking": {"adapter": "http"}}


def scoped_run(tmp_path, monkeypatch, *, restoration, metrics):
    """Run main() for one case on a chosen profile with every external effect faked."""
    events = []

    class Scope:
        def __init__(self, client, *, adapter, model_id, networking):
            events.append(("scope", adapter, model_id, networking))
            self.fingerprint = FINGERPRINT
            self.original = SimpleNamespace(selected={"model_id": "own", "dataset_id": None}, roles={"r": "local"})

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
            events.append("api-login")

        def close(self):
            events.append("api-closed")

    page = SimpleNamespace(screenshot=lambda **_: None)
    browser = SimpleNamespace(new_context=lambda **_: SimpleNamespace(new_page=lambda: page), close=lambda: None)
    playwright = SimpleNamespace(chromium=SimpleNamespace(launch=lambda **_: browser))

    def case(*args, row, checkpoint, evidence=None, unset_families=None):
        events.append("case")
        row.update(verdict="pass", job_id="a" * 32, inference=evidence("a" * 32, 1))
        return row

    monkeypatch.setattr(harness, "sync_playwright", lambda: nullcontext(playwright))
    monkeypatch.setattr(harness, "BrowserTraffic", lambda _: SimpleNamespace(sockets=[]))
    monkeypatch.setattr(harness, "sign_in", lambda *args: events.append("browser-login"))
    monkeypatch.setattr(harness, "AcceptanceProfileScope", Scope)
    monkeypatch.setattr(harness, "BrowserPathClient", Client)
    monkeypatch.setattr(harness, "job_inference_metrics", lambda client, job_id, attempt: metrics)
    monkeypatch.setattr(harness, "run_case", case)
    monkeypatch.setattr(harness, "source_identity", lambda: SOURCE)
    monkeypatch.setattr(harness, "build_cases", lambda *_: [CASE])
    monkeypatch.setattr(harness, "TOOLS", {CASE[1]: {}})
    monkeypatch.setattr(harness, "discover_fixtures", lambda _: None)
    monkeypatch.setattr(harness, "atlassian_admin_configured", lambda: False)
    monkeypatch.setattr(harness, "host_unset_provider_families", lambda: {})
    monkeypatch.setattr(harness, "dotenv_values", lambda _: {"ACCEPTANCE_EMAIL": "test@example.test",
                                                             "ACCEPTANCE_PASSWORD": "fixture-only"})
    monkeypatch.setattr(harness.matrix, "RUN", "10220000")
    path = tmp_path / "browser.json"
    monkeypatch.setattr("sys.argv", ["tool_browser_e2e.py", "--report", str(path), "--login-env",
                                     str(tmp_path / "login.env"), "--adapter", "http", "--model-id", "chosen",
                                     "--networking", "follow"])
    return harness.main(), json.loads(path.read_text()), events


def metric(stage, adapter, *, role=None):
    return {"operation": "generate", "stage": stage, "repair_of": None, "adapter": adapter,
            "outcome": "success", "role": role}


def test_a_chosen_profile_scopes_every_browser_case_and_restores_the_account(tmp_path, monkeypatch):
    code, saved, events = scoped_run(tmp_path, monkeypatch, restoration={"status": "restored"},
                                     metrics=[metric("CONTROL_ROUTING", "local"),
                                              metric("SPECIALIST_REASONING", "http", role="account-specialist")])
    assert code == 0
    # Selected before the browser signs in, confirmed before the case, restored after it.
    assert events == ["api-login", ("scope", "http", "chosen", "follow"), "selected", "browser-login",
                      "checked", "case", "restored", "api-closed"]
    assert saved["inference_profile"] == FINGERPRINT
    assert saved["original_selection"] == {"selected": {"model_id": "own", "dataset_id": None},
                                           "roles": {"r": "local"}}
    assert saved["inference_restoration"] == {"status": "restored"}
    assert saved["results"][CASE[0]]["inference"] == {"adapters": ["http"],
                                                      "stages": ["CONTROL_ROUTING", "SPECIALIST_REASONING"]}
    assert saved["execution_context"]["inference_profile"] == {"adapter": "http", "model_id": "chosen",
                                                               "networking": "follow"}


def test_an_account_left_on_the_test_profile_fails_the_browser_run(tmp_path, monkeypatch):
    code, saved, _ = scoped_run(tmp_path, monkeypatch, restoration={"status": "reconciliation_required"},
                                metrics=[metric("SPECIALIST_REASONING", "http", role="account-specialist")])
    assert code == 1
    assert saved["acceptance"]["overall"] == "FAIL"
    assert "restored" in saved["acceptance"]["reason"]


def test_a_case_whose_steps_ran_elsewhere_does_not_pass(tmp_path, monkeypatch):
    code, saved, _ = scoped_run(tmp_path, monkeypatch, restoration={"status": "restored"},
                                metrics=[metric("SPECIALIST_REASONING", "local", role="account-specialist")])
    assert code == 1
    row = saved["results"][CASE[0]]
    assert row["verdict"] == "fail" and "ran on local, not http" in row["reason"]


@pytest.mark.parametrize("profile", [["--model-id", "connection"], ["--adapter", "local", "--model-id", "connection"],
                                     ["--adapter", "http"], ["--networking", "local"]])
def test_an_incomplete_browser_profile_is_refused_before_source_credentials_or_browser(tmp_path, monkeypatch, profile):
    monkeypatch.setattr(harness, "source_identity", lambda: pytest.fail("must not read repositories"))
    monkeypatch.setattr(harness, "dotenv_values", lambda _: pytest.fail("must not read credentials"))
    monkeypatch.setattr(harness, "sync_playwright", lambda: pytest.fail("must not start a browser"))
    monkeypatch.setattr("sys.argv", ["tool_browser_e2e.py", "--report", str(tmp_path / "new.json"), *profile])
    with pytest.raises(SystemExit) as exit_info:
        harness.main()
    assert exit_info.value.code == 2
    assert not (tmp_path / "new.json").exists()


def test_a_provider_this_host_has_not_set_up_is_reported_not_passed(tmp_path, monkeypatch):
    row = {"verdict": "blocked"}
    traffic = SimpleNamespace(sockets=[{"frames_in": 0}])
    message = "Proxmox VE is not set up on this Janus host. Missing settings: PROXMOX_BASE_URL."
    final = {"job_id": "job", "chat_id": "chat", "status": "failed", "error": message}

    def failed(*_):
        traffic.sockets[0]["frames_in"] += 1
        return final

    traffic.wait = failed
    monkeypatch.setattr(harness, "new_chat", lambda *_: "chat")
    monkeypatch.setattr(harness, "send", lambda *_, **__: ("job", "chat"))
    monkeypatch.setattr(harness, "chat_history", lambda *_: {})
    monkeypatch.setattr(harness, "_assert_history_roundtrip", lambda *_, **__: None)
    monkeypatch.setattr(harness, "worker_evidence", lambda _: None)
    monkeypatch.setattr(harness, "_tool_name", lambda _: "proxmox_nodes")
    recorded = []
    monkeypatch.setattr(harness, "record_created", lambda *args: recorded.append(args))
    case = ("proxmox-nodes", "proxmox_nodes", "List the Proxmox VE nodes.", None)
    result = harness.run_case(None, traffic, arguments(tmp_path), case, {}, row=row, checkpoint=lambda: None,
                              unset_families={"proxmox": ["PROXMOX_BASE_URL"]})
    assert (result["verdict"], result["reason"]) == ("not_configured", message)
    assert recorded == []
