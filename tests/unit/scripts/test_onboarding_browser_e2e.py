"""Driver safeguards use explicit fixtures only; no browser/provider is started."""
from contextlib import nullcontext
from email.message import EmailMessage
import json
from types import SimpleNamespace

import pytest

from scripts import onboarding_browser_e2e as driver
from scripts.workflow_acceptance import CompletedWorkflow


def args(tmp_path):
    return ["--adapter", "local", "--project", "LAB", "--report", str(tmp_path / "report.json"),
            "--private-env", str(tmp_path / "private.env")]


def test_existing_report_stops_before_configuration_or_setup(tmp_path, monkeypatch):
    report = tmp_path / "report.json"
    report.write_text("existing evidence")
    monkeypatch.setattr(driver, "lab_settings", lambda _: pytest.fail("must not load config"))
    with pytest.raises(SystemExit) as result:
        driver.main(args(tmp_path))
    assert result.value.code == 2
    assert report.read_text() == "existing evidence"
    assert not (tmp_path / "private.env").exists()


def test_report_and_separate_private_file_precede_effects_and_failure_retains_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "lab_settings", lambda _: SimpleNamespace(mailbox_lab_domain="fixture.test"))
    monkeypatch.setattr(driver, "source_identity", lambda: {"fixture": "source"})
    def setup(address, password, settings, journal):
        recorded = json.loads((tmp_path / "report.json").read_text())
        private = (tmp_path / "private.env").read_text()
        assert recorded["stage"] == "intent_recorded"
        assert address in private and password in private
        assert address not in json.dumps(recorded) and password not in json.dumps(recorded)
        journal.save("mailbox_credentials_intent", reconciliation_required=True)
    def fail_after_dispatch(_args, journal, *_):
        journal.save("submission_observed", chat_id="fixture-chat", submission_id="fixture-submission")
        journal.save("account_approval_intent", job_id="fixture-job", account_approval_id="fixture-approval")
        raise RuntimeError("private-key-and-recipient-must-never-be-reported")
    monkeypatch.setattr(driver, "create_lab_mailbox", setup)
    monkeypatch.setattr(driver, "execute", fail_after_dispatch)
    assert driver.main(args(tmp_path)) == 2
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["job_id"] == "fixture-job" and report["submission_id"] == "fixture-submission"
    assert report["chat_id"] == "fixture-chat" and report["account_approval_id"] == "fixture-approval"
    assert report["result"] == "incomplete" and report["reconciliation_required"]
    assert "private-key" not in json.dumps(report)
    assert report["error_type"] == "RuntimeError"


def test_source_drift_never_passes_even_when_workflow_returned(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "lab_settings", lambda _: SimpleNamespace(mailbox_lab_domain="fixture.test"))
    source = iter([{"source": "before"}, {"source": "after"}])
    monkeypatch.setattr(driver, "source_identity", lambda: next(source))
    monkeypatch.setattr(driver, "create_lab_mailbox", lambda *_: None)
    monkeypatch.setattr(driver, "execute", lambda *_: None)
    assert driver.main(args(tmp_path)) == 2
    assert json.loads((tmp_path / "report.json").read_text())["result"] == "incomplete"


def test_journal_wont_overwrite_an_existing_attempt(tmp_path):
    path = tmp_path / "report.json"
    original = driver.Journal(path, {"stage": "intent", "job_id": "original"})
    original.save("dispatched")
    with pytest.raises(FileExistsError):
        driver.Journal(path, {"job_id": "replacement"})
    assert json.loads(path.read_text())["job_id"] == "original"


@pytest.mark.parametrize("existing", [(True, False), (False, True), (True, True)])
def test_bootstrap_never_replaces_existing_lab_identity(tmp_path, monkeypatch, existing):
    from services import mailbox
    class Service:
        def __init__(self, _): pass
        def _validate(self, _): pass
        def _identity_state(self, _): return existing
        def _admin(self, *_args, **_kwargs): pytest.fail("must not create")
    monkeypatch.setattr(mailbox, "MaddyMailboxService", Service)
    monkeypatch.setattr(mailbox, "_provisioning_lock", lambda _: nullcontext())
    journal = driver.Journal(tmp_path / "report.json", {})
    with pytest.raises(ValueError, match="already exists"):
        driver.create_lab_mailbox("fixture@lab.test", "synthetic", SimpleNamespace(mailbox_provisioning_lock_path="unused"), journal)


def test_partial_bootstrap_persists_intent_before_each_command(tmp_path, monkeypatch):
    from services import mailbox
    journal = driver.Journal(tmp_path / "report.json", {})
    calls = []
    class Service:
        def __init__(self, _): pass
        def _validate(self, _): pass
        def _identity_state(self, _): return False, False
        def _admin(self, group, action, address, **_kwargs):
            assert action == "create" and address == "fixture@lab.test"
            calls.append(group)
            state = json.loads(journal.path.read_text())
            assert state["stage"] == ("mailbox_credentials_intent" if group == "creds" else "mailbox_storage_intent")
            if group == "imap-acct": raise RuntimeError("uncertain create")
    monkeypatch.setattr(mailbox, "MaddyMailboxService", Service)
    monkeypatch.setattr(mailbox, "_provisioning_lock", lambda _: nullcontext())
    with pytest.raises(RuntimeError):
        driver.create_lab_mailbox("fixture@lab.test", "synthetic", SimpleNamespace(mailbox_provisioning_lock_path="unused"), journal)
    assert calls == ["creds", "imap-acct"]
    assert json.loads(journal.path.read_text())["mailbox_credentials_created"] is True


def test_connection_is_admitted_from_explicit_config_without_provider_assumptions(tmp_path):
    path = tmp_path / "connection.env"
    path.write_text("MODEL_BACKEND=http\nMODEL_ID=fixture-model\nMODEL_URL=https://provider.invalid/v1\n"
                    "MODEL_API_KEY=synthetic\nMODEL_API_FORMAT=anthropic\nMODEL_PROVIDER_ID=fixture\n"
                    'MODEL_OPTIONS_JSON={"admission_max_wait_seconds":30}\n')
    payload = driver.connection_payload(path, "http")
    assert payload["model"] == "fixture-model" and payload["api_key"] == "synthetic"
    assert payload["api_format"] == "anthropic" and payload["provider_id"] == "fixture"
    assert payload["admission_max_wait_seconds"] == 30
    with pytest.raises(ValueError): driver.connection_payload(path, "pi")
    with pytest.raises(ValueError): driver.connection_payload(path, "local")
    assert driver.connection_payload(None, "local") is None


def test_browser_api_uses_existing_session_csrf_and_no_owner_override():
    calls = []
    class Request:
        def get(self, url):
            assert url == "http://backend.invalid/api/v1/auth/me"
            return SimpleNamespace(ok=True, status=200, json=lambda: {
                "csrf_token": "fixture-csrf", "user": {"user_id": "fixture-owner"}})
        def post(self, url, **kwargs):
            calls.append((url, kwargs))
            return SimpleNamespace(ok=True, status=200, json=lambda: {"id": "new-owned-model"})
    api = driver.BrowserAPI(SimpleNamespace(request=Request()), "http://backend.invalid")
    payload = {"backend": "http", "model": "fixture", "api_key": "synthetic-new-credential"}
    response = api.post("/api/v1/extensions", json={"action": "model", "payload": payload},
                        headers=api._mutation_headers())
    response.raise_for_status()
    assert response.json()["id"] == "new-owned-model"
    assert calls == [("http://backend.invalid/api/v1/extensions", {
        "headers": {"X-CSRF-Token": "fixture-csrf"}, "data": {"action": "model", "payload": payload}})]
    assert api.owner == "fixture-owner"


@pytest.mark.parametrize("options", ['{"backend":"local"}', '{"api_key":"other"}', '[]'])
def test_options_cannot_change_connection_identity_or_secret(tmp_path, options):
    path = tmp_path / "connection.env"
    path.write_text("MODEL_BACKEND=http\nMODEL_ID=fixture\nMODEL_URL=https://fixture.invalid\nMODEL_OPTIONS_JSON=" + options)
    with pytest.raises(ValueError): driver.connection_payload(path, "http")


@pytest.mark.parametrize("domain,enabled", [("public.com", "true"), ("lab.test", "false"), ("", "true")])
def test_lab_domain_and_enablement_are_explicit(tmp_path, domain, enabled):
    path = tmp_path / "lab.env"
    path.write_text(f"MAILBOX_LAB_DOMAIN={domain}\nMAILBOX_LAB_ENABLED={enabled}\n")
    with pytest.raises(ValueError): driver.lab_settings(path)


@pytest.mark.parametrize("field", ["given_name", "family_name", "department", "role"])
def test_account_approval_must_match_every_requested_field(field):
    expected = {"given_name": "Fixture", "family_name": "Person", "department": "IT", "role": "Engineer"}
    job = {"status": "waiting_approval", "proposed_tool": {"tool": "account_create", "approval_id": "fixture",
           "arguments": dict(expected)}, "result": {"workflow": {"kind": "account_ticket",
           "state": "awaiting_account_approval", "source_approval_id": "fixture"}}}
    assert driver.exact_account_proposal(job, expected) == job["proposed_tool"]
    job["proposed_tool"]["arguments"][field] = "Substituted"
    with pytest.raises(ValueError): driver.exact_account_proposal(job, expected)


def test_verification_link_must_belong_to_declared_frontend():
    message = EmailMessage()
    message.set_content("http://frontend.invalid/?verify_token=synthetic-token")
    assert driver.verification_link(message, "http://frontend.invalid").endswith("verify_token=synthetic-token")
    with pytest.raises(ValueError): driver.verification_link(message, "http://other.invalid")


def test_inbox_requires_rfc_message_id_and_all_three_correlations():
    completed = CompletedWorkflow("fixture-job", 1, "fixture.person", "private@fixture.test", "LAB-1")
    message = EmailMessage()
    message["Message-ID"] = "<fixture@lab.test>"
    message.set_content("fixture-job fixture.person LAB-1")
    assert driver.correlated_inbox(message, completed)
    message.replace_header("Message-ID", "not-a-message-id")
    assert not driver.correlated_inbox(message, completed)
    message.replace_header("Message-ID", "<fixture@lab.test>")
    message.set_content("someone-else fixture.person LAB-1")
    assert not driver.correlated_inbox(message, completed)
