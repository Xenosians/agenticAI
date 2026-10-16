"""Runner contract tests use synthetic HTTP responses, not live acceptance evidence."""
from argparse import Namespace
from copy import deepcopy
import json

import httpx
import pytest

from scripts import cross_repo_acceptance as acceptance


CONTRACTS = acceptance.BACKEND_REPO / "contracts"


def job_payload(status="completed", *, version=2):
    job = {
        "job_id": "job-test", "user_id": "owner-test", "conversation_id": "chat-test",
        "chat_id": "chat-test", "message": "Synthetic contract fixture", "status": status,
        "attempts": 1, "created_at": "2026-10-15T00:00:00Z", "claimed_at": None,
        "lease_expires_at": None, "completed_at": "2026-10-15T00:00:01Z",
        "selected_agent": None, "proposed_tool": None, "result": None, "error": None,
    }
    if version == 2:
        job["contract_version"] = 2
    return job


def client_for(handler, *, version=2):
    client = acceptance.BrowserPathClient(
        "http://backend.test", timeout_seconds=0.5, job_api_version=version,
        contracts_dir=CONTRACTS, transport=httpx.MockTransport(handler),
    )
    client.csrf_token = "test-csrf"
    return client


def test_created_job_uses_public_chat_identifier_instead_of_storage_record_id():
    requests = []

    def handler(request):
        requests.append((request.url.path, json.loads(request.content)))
        if request.url.path == "/api/v1/chats":
            return httpx.Response(201, json={"chat": {
                "id": "chat:fixture-canonical", "chat_id": "fixture-canonical",
            }})
        return httpx.Response(202, json={"job_id": "job-test", "status": "pending"})

    client = client_for(handler)
    try:
        chat_id = client.create_chat("Fixture chat")
        client.create_job(chat_id=chat_id, message="Fixture request")
    finally:
        client.close()

    assert chat_id == "fixture-canonical"
    assert requests[1][1]["chat_id"] == "fixture-canonical"


@pytest.mark.parametrize("status", ["denied", "outcome_unknown"])
def test_v2_terminal_outcomes_stop_polling_without_retry(status):
    requests = []

    def handler(request):
        requests.append((request.method, request.url.path))
        return httpx.Response(200, json=job_payload(status))

    client = client_for(handler)
    try:
        result = client.wait_for_terminal("job-test", poll_seconds=0.001)
    finally:
        client.close()
    assert result["status"] == status
    assert requests == [("GET", "/api/v2/jobs/job-test")]


def test_v2_read_and_approval_use_v2_but_creation_uses_v1():
    requests = []

    def handler(request):
        requests.append((request.method, request.url.path))
        if request.method == "POST":
            assert request.headers["X-CSRF-Token"] == "test-csrf"
        if request.url.path == "/api/v1/jobs":
            return httpx.Response(202, json={"job_id": "job-test", "status": "pending"})
        return httpx.Response(200, json=job_payload())

    client = client_for(handler)
    try:
        client.create_job(chat_id="chat-test", message="Fixture request")
        client.get_job("job-test")
        client.approve_job("job-test")
    finally:
        client.close()
    assert requests == [
        ("POST", "/api/v1/jobs"), ("GET", "/api/v2/jobs/job-test"),
        ("POST", "/api/v2/jobs/job-test/approve"),
    ]


@pytest.mark.parametrize("status", ["completed", "failed", "reconciliation_required"])
def test_v1_compatibility_keeps_legacy_route_and_contract(status):
    def handler(request):
        assert request.url.path == "/api/v1/jobs/job-test"
        return httpx.Response(200, json=job_payload(status, version=1))

    client = client_for(handler, version=1)
    try:
        assert client.wait_for_terminal("job-test", poll_seconds=0.001)["status"] == status
    finally:
        client.close()


@pytest.mark.parametrize("changes", [
    {"contract_version": 1}, {"status": "reconciliation_required"},
    {"status": "invented_success"}, {"attempts": -1},
    {"created_at": "not-a-date"}, {"untrusted_extra": True},
    {"job_id": "another-job"},
])
def test_v2_rejects_invalid_or_mismatched_public_envelope(changes):
    payload = job_payload()
    payload.update(changes)
    client = client_for(lambda request: httpx.Response(200, json=payload))
    try:
        with pytest.raises(acceptance.AcceptanceFailure):
            client.get_job("job-test")
    finally:
        client.close()


def test_schema_error_does_not_echo_response_contents():
    payload = job_payload()
    payload["status"] = "private-response-content"
    client = client_for(lambda request: httpx.Response(200, json=payload))
    try:
        with pytest.raises(acceptance.AcceptanceFailure) as error:
            client.get_job("job-test")
    finally:
        client.close()
    assert "private-response-content" not in str(error.value)


@pytest.mark.parametrize("proposal", [
    None, {"tool": "workspace_mkdir", "arguments": {}},
    {"tool": "workspace_mkdir", "arguments": {}, "approval_id": ""},
    {"tool": "workspace_mkdir", "arguments": [], "approval_id": "approval-test"},
])
def test_approval_state_requires_valid_persisted_proposal(proposal):
    payload = job_payload("waiting_approval")
    payload["proposed_tool"] = proposal
    client = client_for(lambda request: httpx.Response(200, json=payload))
    try:
        with pytest.raises(acceptance.AcceptanceFailure):
            client.get_job("job-test")
    finally:
        client.close()


def test_pending_approval_with_exact_proposal_remains_a_gate():
    payload = job_payload("waiting_approval")
    payload["proposed_tool"] = {
        "tool": "workspace_mkdir", "arguments": {"path": "fixture-directory"},
        "approval_id": "approval-test",
    }
    methods = []

    def handler(request):
        methods.append(request.method)
        return httpx.Response(200, json=payload)

    client = client_for(handler)
    try:
        assert client.wait_for_terminal("job-test", poll_seconds=0.001) == payload
    finally:
        client.close()
    assert methods == ["GET"]


def test_create_ack_is_validated_before_polling():
    client = client_for(lambda request: httpx.Response(
        202, json={"job_id": "job-test", "status": "completed"},
    ))
    try:
        with pytest.raises(acceptance.AcceptanceFailure, match="job creation"):
            client.create_job(chat_id="chat-test", message="Fixture request")
    finally:
        client.close()


def read_case():
    return {"id": "read", "prompt": "Read the fixture", "expected_tool": "workspace_project_info",
            "expected_terminal_status": ["completed"], "require_tool_result": True}


@pytest.mark.parametrize("worker", [
    None, {"proposed_tool": "workspace_project_info", "status": "failed", "tool_result": {}},
    {"proposed_tool": "workspace_project_info", "status": "success", "tool_result": None},
    {"proposed_tool": "different_tool", "status": "success", "tool_result": {}},
])
def test_completed_read_is_not_success_without_tool_evidence(worker):
    payload = job_payload()
    payload["proposed_tool"] = {"tool": "workspace_project_info", "arguments": {}}
    payload["result"] = {"status": "success", "answer": "Fixture answer", "results": [worker]}
    with pytest.raises(acceptance.AcceptanceFailure, match="(structured tool|worker) evidence"):
        acceptance._assert_case(read_case(), payload)


def test_completed_read_requires_answer_and_matching_successful_result():
    payload = job_payload()
    payload["proposed_tool"] = {"tool": "workspace_project_info", "arguments": {}}
    payload["result"] = {"answer": "Fixture answer", "results": [{
        "proposed_tool": "workspace_project_info", "status": "success", "tool_result": {},
    }]}
    acceptance._assert_case(read_case(), payload)
    payload["result"]["answer"] = " "
    with pytest.raises(acceptance.AcceptanceFailure, match="application answer"):
        acceptance._assert_case(read_case(), payload)


def test_fail_closed_case_cannot_pass_because_the_runtime_failed():
    case = {"expected_terminal_status": ["completed", "denied"],
            "forbid_tool_execution": True, "forbid_waiting_approval": True}
    with pytest.raises(acceptance.AcceptanceFailure, match="terminal status"):
        acceptance._assert_case(case, job_payload("failed"))
    payload = job_payload("completed")
    payload["result"] = {"results": [{"tool_result": {"ran": True}}]}
    with pytest.raises(acceptance.AcceptanceFailure, match="executed tool evidence"):
        acceptance._assert_case(case, payload)


def test_fail_closed_case_requires_a_visible_safe_resolution():
    case = {"expected_terminal_status": ["completed"], "forbid_tool_execution": True}
    payload = job_payload("completed")
    with pytest.raises(acceptance.AcceptanceFailure, match="resolution evidence"):
        acceptance._assert_case(case, payload)
    payload["result"] = {"routes": [], "results": [], "answer": "Which account?"}
    acceptance._assert_case(case, payload)
    payload["result"]["routes"] = ["account-specialist"]
    payload["result"]["results"] = [{"status": "clarification_required", "tool_result": None}]
    acceptance._assert_case(case, payload)
    payload["result"]["results"][0]["status"] = "error"
    with pytest.raises(acceptance.AcceptanceFailure, match="empty plan"):
        acceptance._assert_case(case, payload)


def test_cli_defaults_to_current_contract_and_can_request_legacy(monkeypatch):
    monkeypatch.setattr(acceptance.sys, "argv", ["cross_repo_acceptance.py"])
    assert acceptance.parse_args().job_api_version == 2
    monkeypatch.setattr(acceptance.sys, "argv", ["cross_repo_acceptance.py", "--job-api-version", "1"])
    assert acceptance.parse_args().job_api_version == 1


@pytest.mark.parametrize("cases,selected", [([], set()), ([read_case()], {"typo"}),
    ([read_case(), read_case()], set()), ([{"id": "missing-status", "prompt": "Read"}], set()),
    ([dict(read_case(), enabled=False)], set()),
])
def test_empty_unknown_duplicate_or_underspecified_cases_cannot_pass(cases, selected):
    with pytest.raises(acceptance.AcceptanceFailure):
        acceptance._select_cases({"cases": cases}, selected)


def test_shipped_cases_have_explicit_outcomes_and_visible_coverage_limits():
    config = json.loads(acceptance.DEFAULT_CASE_FILE.read_text(encoding="utf-8"))
    cases = acceptance._select_cases(config, set())
    assert config["coverage_limits"]
    assert all(case["require_tool_result"] for case in cases if case["kind"] == "read")
    assert all(not case.get("allow_execute") for case in cases)


def test_skipped_selected_case_writes_incomplete_report_and_fails(monkeypatch, tmp_path):
    case = dict(read_case(), requires_env=["ACCEPTANCE_MISSING_FIXTURE"])
    manifest = tmp_path / "cases.json"
    manifest.write_text(json.dumps({"cases": [case]}), encoding="utf-8")
    monkeypatch.setenv("ACCEPTANCE_EMAIL", "fixture@example.invalid")
    monkeypatch.setenv("ACCEPTANCE_PASSWORD", "fixture-password")
    monkeypatch.delenv("ACCEPTANCE_MISSING_FIXTURE", raising=False)
    monkeypatch.setattr(acceptance, "AI_REPO", tmp_path)
    calls = []

    class FakeClient:
        csrf_token = "fixture-csrf"

        def __init__(self, *args, **kwargs):
            pass

        def preflight(self):
            pass

        def login(self, *args):
            pass

        def integrations(self):
            return {}

        def create_chat(self, title):
            return "chat-test"

        def create_job(self, **kwargs):
            calls.append("create_job")
            raise AssertionError("A missing prerequisite must not create a job")

        def close(self):
            pass

    monkeypatch.setattr(acceptance, "BrowserPathClient", FakeClient)
    args = Namespace(email=None, non_interactive=True, case_file=manifest, case=None,
                     backend_url="http://backend.test", timeout=1, job_api_version=2,
                     contracts_dir=CONTRACTS, poll_interval=0.001, execute_approvals=False,
                     cleanup_chat=False)
    with pytest.raises(acceptance.AcceptanceFailure, match="incomplete"):
        acceptance.run_live(args)
    reports = list((tmp_path / ".runtime/acceptance/cross-repo").glob("*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["complete"] is False
    assert report["job_api_version"] == 2
    assert report["results"][0]["outcome"] == "skipped"
    assert calls == []


def test_non_interactive_missing_password_does_not_prompt(monkeypatch):
    monkeypatch.setenv("ACCEPTANCE_EMAIL", "fixture@example.invalid")
    monkeypatch.delenv("ACCEPTANCE_PASSWORD", raising=False)
    monkeypatch.delenv("V18_03_PASSWORD", raising=False)
    monkeypatch.setattr(acceptance.getpass, "getpass", lambda *args: pytest.fail("Must not prompt"))
    with pytest.raises(acceptance.AcceptanceFailure, match="ACCEPTANCE_PASSWORD"):
        acceptance.run_live(Namespace(email=None, non_interactive=True))


def test_descriptive_configuration_overrides_legacy_fixture_aliases(monkeypatch):
    monkeypatch.setenv("V18_03_EMAIL", "legacy@example.invalid")
    monkeypatch.delenv("ACCEPTANCE_EMAIL", raising=False)
    assert acceptance._acceptance_env("ACCEPTANCE_EMAIL") == "legacy@example.invalid"
    monkeypatch.setenv("ACCEPTANCE_EMAIL", "current@example.invalid")
    assert acceptance._acceptance_env("ACCEPTANCE_EMAIL") == "current@example.invalid"


def test_shipped_case_prerequisites_accept_legacy_environment_aliases(monkeypatch):
    monkeypatch.setenv("V18_03_ACCOUNT_USER_ID", "fixture-user")
    monkeypatch.delenv("ACCEPTANCE_ACCOUNT_USER_ID", raising=False)
    case = {"requires_env": ["ACCEPTANCE_ACCOUNT_USER_ID"]}
    assert acceptance._skip_reason(case, {}) is None
    assert acceptance._expand_prompt(
        "Check ${ACCEPTANCE_ACCOUNT_USER_ID}", run_token="fixture-run"
    ) == "Check fixture-user"


def pending_approval():
    job = job_payload("waiting_approval")
    job["selected_agent"] = "developer-specialist"
    job["proposed_tool"] = {
        "tool": "workspace_mkdir", "arguments": {"path": "private-fixture-directory"},
        "approval_id": "approval-test",
    }
    job["result"] = {"answer": "Approval required."}
    return job


def completed_approval():
    job = pending_approval()
    job["status"] = "completed"
    job["result"] = {"answer": "Directory created.", "approval_execution": {
        "approval_id": "approval-test", "protocol_status": "processed",
        "execution_status": "succeeded", "approval_status": "approved", "replayed": False,
        "result": {"ok": True, "status": "success", "created": True}, "error": None,
    }}
    return job


def history_for(job):
    return {"chat_id": "chat-test", "messages": [
        {"role": "user", "content": job["message"], "job_id": job["job_id"],
         "status": job["status"]},
        {"role": "assistant", "content": job.get("error") or job["result"]["answer"],
         "job_id": job["job_id"], "status": job["status"],
         "proposed_tool": deepcopy(job["proposed_tool"])},
    ]}


@pytest.mark.parametrize("version", [1, 2])
def test_history_uses_selected_version_and_canonical_chat_identifier(version):
    requests = []

    def handler(request):
        requests.append(request.url.path)
        assert request.url.params["limit"] == "200"
        return httpx.Response(200, json=history_for(completed_approval()))

    client = client_for(handler, version=version)
    try:
        history = client.get_history("chat-test")
        acceptance._assert_history_roundtrip(history, completed_approval(), chat_id="chat-test")
    finally:
        client.close()
    assert requests == [f"/api/v{version}/chats/chat-test/history"]


@pytest.mark.parametrize("change", ["storage-id", "missing-user", "missing-answer", "stale-status",
                                    "changed-proposal", "duplicate-answer", "changed-message"])
def test_invisible_stale_or_misbound_history_cannot_pass(change):
    job = completed_approval()
    history = history_for(job)
    if change == "storage-id":
        history["chat_id"] = "chat:chat-test"
    elif change == "missing-user":
        history["messages"].pop(0)
    elif change == "missing-answer":
        history["messages"].pop()
    elif change == "stale-status":
        history["messages"][1]["status"] = "waiting_approval"
    elif change == "changed-proposal":
        history["messages"][1]["proposed_tool"]["arguments"]["path"] = "different-directory"
    elif change == "duplicate-answer":
        history["messages"].append(deepcopy(history["messages"][1]))
    else:
        history["messages"][0]["content"] = "Different request"
    with pytest.raises(acceptance.AcceptanceFailure, match="(history|chat identifier)"):
        acceptance._assert_history_roundtrip(history, job, chat_id="chat-test")


@pytest.mark.parametrize("status", ["waiting_approval", "completed", "denied", "outcome_unknown"])
def test_history_preserves_visible_outcomes_and_proposal(status):
    job = pending_approval() if status == "waiting_approval" else completed_approval()
    job["status"] = status
    if status in {"denied", "outcome_unknown"}:
        job["error"] = "Trusted terminal explanation."
    acceptance._assert_history_roundtrip(history_for(job), job, chat_id="chat-test")


@pytest.mark.parametrize("change", ["approval-id", "proposal", "attempt", "protocol", "authority",
                                    "outcome", "replayed", "missing-result", "failed-tool", "answer"])
def test_completed_approval_requires_bound_authoritative_successful_execution(change):
    resolved = completed_approval()
    outcome = resolved["result"]["approval_execution"]
    if change == "approval-id":
        outcome["approval_id"] = "another-approval"
    elif change == "proposal":
        resolved["proposed_tool"]["arguments"]["path"] = "different-directory"
    elif change == "attempt":
        resolved["attempts"] = 2
    elif change == "protocol":
        outcome["protocol_status"] = "accepted"
    elif change == "authority":
        outcome["approval_status"] = "executing"
    elif change == "outcome":
        outcome["execution_status"] = "unresolved"
    elif change == "replayed":
        outcome["replayed"] = "false"
    elif change == "missing-result":
        outcome["result"] = None
    elif change == "failed-tool":
        outcome["result"] = {"ok": False, "status": "error"}
    else:
        resolved["result"]["answer"] = " "
    with pytest.raises(acceptance.AcceptanceFailure):
        acceptance._assert_approval_execution(pending_approval(), resolved, {})


def test_approved_execution_reopens_and_replay_conflict_preserves_terminal_result():
    requests = []
    resolved = completed_approval()
    posts = 0

    def handler(request):
        nonlocal posts
        requests.append((request.method, request.url.path))
        if request.method == "POST":
            posts += 1
            assert request.headers["X-CSRF-Token"] == "test-csrf"
            if posts == 2:
                return httpx.Response(409, json={
                    "error": "job_not_waiting_approval", "status": "completed",
                })
        return httpx.Response(200, json=deepcopy(resolved))

    client = client_for(handler)
    original = pending_approval()
    try:
        result, checks = acceptance._execute_approval_case(
            client, {"allow_execute": True}, original, poll_seconds=0.001,
        )
    finally:
        client.close()
    assert original == pending_approval()
    assert result == resolved
    assert checks["approval_id"] == "approval-test"
    assert checks["structured_tool_success_verified"] is True
    assert len(checks["proposal_sha256"]) == 64
    assert "private-fixture-directory" not in json.dumps(checks)
    assert checks["external_execution_count_verified"] is False
    assert checks["model_not_rerun_verified"] is False
    assert requests == [
        ("POST", "/api/v2/jobs/job-test/approve"), ("GET", "/api/v2/jobs/job-test"),
        ("GET", "/api/v2/jobs/job-test"), ("POST", "/api/v2/jobs/job-test/approve"),
        ("GET", "/api/v2/jobs/job-test"),
    ]


@pytest.mark.parametrize("fault", ["changed-proposal", "reopen-result", "replay-success",
                                   "replay-wrong-error", "replay-result"])
def test_approval_execution_fails_on_changed_binding_result_or_accepted_replay(fault):
    posts = gets = 0

    def handler(request):
        nonlocal posts, gets
        resolved = completed_approval()
        if request.method == "POST":
            posts += 1
            if posts == 2:
                if fault == "replay-success":
                    return httpx.Response(200, json=resolved)
                return httpx.Response(409, json={
                    "error": "another_conflict" if fault == "replay-wrong-error"
                    else "job_not_waiting_approval", "status": "completed",
                })
        else:
            gets += 1
            if fault == "changed-proposal" and gets == 1:
                resolved["proposed_tool"]["arguments"]["path"] = "substituted-directory"
            if (fault == "reopen-result" and gets == 2) or (fault == "replay-result" and gets == 3):
                resolved["result"]["approval_execution"]["result"]["changed"] = True
        return httpx.Response(200, json=resolved)

    client = client_for(handler)
    try:
        with pytest.raises(acceptance.AcceptanceFailure):
            acceptance._execute_approval_case(
                client, {"allow_execute": True}, pending_approval(), poll_seconds=0.001,
            )
    finally:
        client.close()


def test_approval_execution_without_case_opt_in_makes_no_request():
    client = client_for(lambda request: pytest.fail("Execution was not explicitly enabled"))
    try:
        with pytest.raises(acceptance.AcceptanceFailure, match="allow_execute"):
            acceptance._execute_approval_case(client, {}, pending_approval(), poll_seconds=0.001)
    finally:
        client.close()


@pytest.mark.parametrize("invisible", [False, True])
def test_live_case_requires_history_roundtrip_before_reporting_pass(monkeypatch, tmp_path, invisible):
    case = read_case()
    manifest = tmp_path / "cases.json"
    manifest.write_text(json.dumps({"cases": [case]}), encoding="utf-8")
    monkeypatch.setenv("ACCEPTANCE_EMAIL", "fixture@example.invalid")
    monkeypatch.setenv("ACCEPTANCE_PASSWORD", "fixture-password")
    monkeypatch.setattr(acceptance, "AI_REPO", tmp_path)
    job = job_payload()
    job["message"] = case["prompt"]
    job["proposed_tool"] = {"tool": case["expected_tool"], "arguments": {}}
    job["result"] = {"answer": "Fixture answer", "results": [{
        "proposed_tool": case["expected_tool"], "status": "success", "tool_result": {},
    }]}

    def handler(request):
        if request.url.path.endswith("/history"):
            history = history_for(job)
            if invisible:
                history["messages"] = []
            return httpx.Response(200, json=history)
        if request.url.path == "/api/v1/auth/login":
            return httpx.Response(200, json={"csrf_token": "test-csrf"})
        if request.url.path == "/api/v1/chats":
            return httpx.Response(201, json={"chat": {"chat_id": "chat-test"}})
        if request.url.path == "/api/v1/jobs":
            return httpx.Response(202, json={"job_id": "job-test", "status": "pending"})
        if request.url.path == "/api/v2/jobs/job-test":
            return httpx.Response(200, json=job)
        return httpx.Response(200, json={})

    client = client_for(handler)
    monkeypatch.setattr(acceptance, "BrowserPathClient", lambda *args, **kwargs: client)
    args = Namespace(email=None, non_interactive=True, case_file=manifest, case=None,
                     backend_url="http://backend.test", timeout=1, job_api_version=2,
                     contracts_dir=CONTRACTS, poll_interval=0.001, execute_approvals=False,
                     cleanup_chat=False)
    if invisible:
        with pytest.raises(acceptance.AcceptanceFailure, match="history"):
            acceptance.run_live(args)
    else:
        acceptance.run_live(args)
    report = json.loads(next((tmp_path / ".runtime/acceptance/cross-repo").glob("*.json")).read_text())
    assert report["complete"] is not invisible
    if not invisible:
        assert report["results"][0]["canonical_history_roundtrip"] is True
