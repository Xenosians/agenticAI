#!/usr/bin/env python3
"""Live UI registration -> verified account -> Jira -> correlated lab inbox.

One new report/private credential file per attempt. Unknown effects require
operator reconciliation; there is no resume or automatic replacement.
"""
from __future__ import annotations

import argparse
import email
import hashlib
import imaplib
import json
import math
import os
import re
import secrets
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import dotenv_values
from playwright.sync_api import sync_playwright
from scripts.acceptance_profiles import AcceptanceProfileScope
from scripts.cross_repo_acceptance import _assert_history_roundtrip
from scripts.tool_browser_e2e import BrowserTraffic, TERMINAL, approve, chat_history, send, socket_evidence
from scripts.tool_matrix import source_identity
from scripts.workflow_acceptance import account_proposal, completed_workflow, confirm_inbox_message, ticket_proposal
from agent.requested_workflow import RequestedWorkflowContract


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def write_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


class Journal:
    def __init__(self, path: Path, initial: dict):
        self.path, self.data = path, initial
        write_exclusive(path, json.dumps(initial, indent=2) + "\n")

    def save(self, stage: str, **fields) -> None:
        self.data.update(stage=stage, **fields)
        temporary = self.path.with_suffix(self.path.suffix + ".pending")
        write_exclusive(temporary, json.dumps(self.data, indent=2) + "\n")
        temporary.replace(self.path)


def base_url(value: str) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}):
        raise ValueError("Application URLs must be HTTP(S) origins without credentials")
    return value.rstrip("/")


def connection_payload(path: Path | None, adapter: str) -> dict | None:
    if adapter == "local":
        if path is not None:
            raise ValueError("Local scope uses configured defaults, without a connection file")
        return None
    if path is None or not path.is_file():
        raise ValueError("Outbound scope needs an existing private --connection-env")
    values = dotenv_values(path)
    if values.get("MODEL_BACKEND") != adapter or not values.get("MODEL_ID"):
        raise ValueError("Connection backend/model must explicitly match the adapter")
    payload = {"name": "Acceptance connection", "backend": adapter, "model": values["MODEL_ID"]}
    for source, target in (("MODEL_URL", "url"), ("MODEL_API_KEY", "api_key"),
                           ("MODEL_API_FORMAT", "api_format"), ("MODEL_PROVIDER_ID", "provider_id")):
        if values.get(source):
            payload[target] = values[source]
    if adapter == "http" and not payload.get("url"):
        raise ValueError("HTTP scope requires MODEL_URL")
    if values.get("MODEL_OPTIONS_JSON"):
        from services.extensions.options import MODEL_OPTION_FIELDS
        options = json.loads(values["MODEL_OPTIONS_JSON"])
        if not isinstance(options, dict) or set(options) - MODEL_OPTION_FIELDS:
            raise ValueError("Connection options must use supported inference option fields")
        payload.update(options)
    return payload


def lab_settings(path: Path) -> SimpleNamespace:
    values = dotenv_values(path)
    domain = values.get("MAILBOX_LAB_DOMAIN", "")
    if (values.get("MAILBOX_LAB_ENABLED", "").lower() not in {"true", "1"}
            or not re.fullmatch(r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+test", domain)
            or len(domain) > 253):
        raise ValueError("Explicit enabled .test mailbox-lab configuration is required")
    port = int(values.get("MAILBOX_IMAP_PORT", "1143"))
    if not 1024 <= port <= 65535:
        raise ValueError("Lab IMAP port is invalid")
    return SimpleNamespace(mailbox_lab_domain=domain, mailbox_lab_enabled=True,
                           mailbox_imap_port=port, mailbox_timeout_seconds=10,
                           mailbox_provisioning_lock_path=PROJECT_ROOT / ".runtime/mailbox-provisioning.lock")


def create_lab_mailbox(address: str, password: str, settings, journal: Journal) -> None:
    # Reuse the provider's fixed command allowlist, validation and process lock.
    # Registration needs a mailbox before an authenticated product owner exists.
    from services.mailbox import MaddyMailboxService, _provisioning_lock
    service = MaddyMailboxService(settings)
    service._validate(address)
    with _provisioning_lock(settings.mailbox_provisioning_lock_path):
        if any(service._identity_state(address)):
            raise ValueError("Lab identity already exists; never replace or retry it")
        journal.save("mailbox_credentials_intent", reconciliation_required=True)
        service._admin("creds", "create", address, password=password)
        journal.save("mailbox_storage_intent", mailbox_credentials_created=True)
        service._admin("imap-acct", "create", address)
        if service._identity_state(address) != (True, True):
            raise ValueError("Lab mailbox creation lacks exact readback")
    journal.save("mailbox_created", mailbox_setup="Maddy admin CLI; not tool execution credit")


def text_body(message: email.message.Message) -> str:
    for part in message.walk():
        if part.get_content_type() == "text/plain":
            return (part.get_payload(decode=True) or b"").decode(part.get_content_charset() or "utf-8", errors="replace")
    return ""


def wait_for_mail(address: str, password: str, settings, timeout: float, predicate):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with imaplib.IMAP4("127.0.0.1", settings.mailbox_imap_port, timeout=min(10, timeout)) as imap:
            if imap.login(address, password)[0] != "OK" or imap.select("INBOX", readonly=True)[0] != "OK":
                raise ValueError("Lab IMAP authentication/read failed")
            status, data = imap.search(None, "ALL")
            if status != "OK":
                raise ValueError("Lab IMAP search failed")
            for identity in reversed(data[0].split() if data and data[0] else []):
                status, parts = imap.fetch(identity, "(RFC822)")
                if status != "OK":
                    raise ValueError("Lab IMAP message read failed")
                raw = next((part[1] for part in parts if isinstance(part, tuple) and isinstance(part[1], bytes)), None)
                if raw is not None:
                    message = email.message_from_bytes(raw)
                    if predicate(message):
                        return message
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    raise TimeoutError("Expected correlated mail was not observed within the declared budget")


def verification_link(message, frontend_url: str) -> str | None:
    for value in re.findall(r"https?://[^\s<>]+", text_body(message)):
        parsed = urlsplit(value)
        if parse_qs(parsed.query).get("verify_token"):
            if base_url(f"{parsed.scheme}://{parsed.netloc}") != frontend_url:
                raise ValueError("Verification link does not belong to the declared frontend")
            return value
    return None


class BrowserAPI:
    """Authenticated product API sharing the UI session, never an owner override."""
    def __init__(self, page, backend: str):
        self.page, self.backend, self.client = page, backend, self
        response = self.get("/api/v1/auth/me")
        response.raise_for_status()
        body = response.json()
        self.csrf = body.get("csrf_token")
        self.owner = (body.get("user") or {}).get("user_id")
        if not self.csrf or not self.owner:
            raise ValueError("Authenticated owner/session metadata is incomplete")

    def _mutation_headers(self):
        return {"X-CSRF-Token": self.csrf}

    def request(self, method, path, **kwargs):
        response = getattr(self.page.request, method)(self.backend + path, **kwargs)
        def require_ok():
            if not response.ok:
                raise ValueError(f"Authenticated product API returned HTTP {response.status}")
        return SimpleNamespace(raise_for_status=require_ok, json=response.json)

    def get(self, path):
        return self.request("get", path)

    def post(self, path, *, json, headers):
        return self.request("post", path, data=json, headers=headers)


def exact_account_proposal(job: dict, expected: dict) -> dict:
    proposal = account_proposal(job)
    if any(proposal["arguments"].get(key) != value for key, value in expected.items()):
        raise ValueError("Account proposal differs from the exact requested identity")
    return proposal


def correlated_inbox(message, completed) -> bool:
    if not re.fullmatch(r"<[^<>\s]+@[^<>\s]+>", message.get("Message-ID", "")):
        return False
    try:
        confirm_inbox_message(message["Message-ID"], text_body(message), completed)
        return True
    except ValueError:
        return False


def fill_auth(page, label: str, value: str) -> None:
    field = page.locator("label.auth-field", has_text=label).locator("input")
    field.fill(value)
    field.press("Tab")


def onboarding_contract(expected: dict[str, str], project: str) -> RequestedWorkflowContract:
    """Declare the disposable workflow being tested, never an observed result."""
    return RequestedWorkflowContract.model_validate({"version": 1, "steps": [
        {"id": "step_1", "capability": "account_create", "grounded_arguments": {
            key: [value] for key, value in expected.items()
        }, "after_success_of": None},
        {"id": "step_2", "capability": "ticket_create", "grounded_arguments": {
            "project_key": [project],
        }, "after_success_of": "step_1"},
    ]})


def select_reviewed_workflow(page, contract: RequestedWorkflowContract, timeout_ms: int) -> None:
    """Use the visible composer controls; never inject application state."""
    rows = page.locator(".required-operations-summary .required-operation")
    if rows.count() != 0:
        raise ValueError("A previous operation draft exists; do not replace it")
    toggle = page.get_by_role("button", name="Required operations", exact=True)
    if toggle.get_attribute("aria-expanded") != "true":
        toggle.click()
    for index, step in enumerate(contract.steps):
        choice = page.get_by_role("combobox", name="Required operation", exact=True)
        choice.locator(f'option[value="{step.capability}"]').wait_for(state="attached", timeout=timeout_ms)
        choice.select_option(step.capability)
        for key, values in step.grounded_arguments.items():
            field = page.get_by_role("textbox", name="Exact " + key, exact=True)
            field.fill("\n".join(values))
            field.press("Tab")
        if step.after_success_of is not None:
            page.get_by_role("combobox", name="Success dependency", exact=True).select_option(step.after_success_of)
        page.get_by_role("button", name="Add required operation", exact=True).click()
        rows.nth(index).wait_for(state="visible", timeout=timeout_ms)
    if rows.count() != len(contract.steps):
        raise ValueError("The composer did not retain the reviewed operations")


def assert_reviewed_binding(record: dict, contract: RequestedWorkflowContract) -> None:
    if (not isinstance(record, dict) or record.get("workflow_contract") != contract.model_dump()
            or record.get("workflow_contract_hash") != contract.content_hash()):
        raise ValueError("Observed workflow scope differs from the exact reviewed request")


def assert_reviewed_submission(observations: list[dict], contract: RequestedWorkflowContract) -> None:
    if (len(observations) != 1 or not isinstance(observations[0], dict)
            or observations[0].get("submitted") != contract.model_dump()):
        raise ValueError("The browser did not submit exactly one reviewed workflow")
    assert_reviewed_binding(observations[0].get("ack"), contract)


def execute(args, journal, settings, private, connection):
    timeout_ms = int(args.timeout * 1000)
    expected = {"given_name": "Onboard", "family_name": "Run" + journal.data["run"],
                "department": "IT", "role": "Engineer"}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not args.headed)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        try:
            page = context.new_page()
            traffic = BrowserTraffic(page)
            workflow_observations = []
            def observed_response(response):
                if response.url != args.backend_url + "/api/v1/jobs" or response.request.method != "POST":
                    return
                try:
                    body = response.json()
                except Exception:
                    return
                if isinstance(body, dict) and isinstance(body.get("job_id"), str):
                    journal.save("job_response_observed", job_id=body["job_id"])
                    if args.reviewed_workflow:
                        try:
                            submitted = json.loads(response.request.post_data or "{}")
                        except (TypeError, ValueError):
                            submitted = {}
                        workflow_observations.append({
                            "submitted": submitted.get("workflow_contract") if isinstance(submitted, dict) else None,
                            "ack": {key: body.get(key) for key in ("workflow_contract", "workflow_contract_hash")},
                        })
            page.on("response", observed_response)
            page.goto(args.frontend_url, wait_until="domcontentloaded")
            page.get_by_role("button", name="Create account", exact=True).click()
            for label, value in (("Display name", "Acceptance " + journal.data["run"]),
                                 ("Email", private["email"]), ("Password", private["password"])):
                fill_auth(page, label, value)
            journal.save("registration_intent")
            with page.expect_response(lambda r: r.url == args.backend_url + "/api/v1/auth/register",
                                      timeout=timeout_ms) as observed:
                page.get_by_role("button", name="Register", exact=True).click()
            if not observed.value.ok:
                raise ValueError("Application registration was not accepted")
            journal.save("registration_acknowledged")
            verification = wait_for_mail(private["email"], private["mailbox_password"], settings, args.timeout,
                                         lambda message: verification_link(message, args.frontend_url) is not None)
            journal.save("verification_link_intent", verification_message_sha256=digest(verification.get("Message-ID", "")))
            page.goto(verification_link(verification, args.frontend_url), wait_until="domcontentloaded")
            page.locator(".auth-notice").wait_for(state="visible", timeout=timeout_ms)
            journal.save("login_intent")
            for label, value in (("Email", private["email"]), ("Password", private["password"])):
                fill_auth(page, label, value)
            page.get_by_role("button", name="Sign in", exact=True).click()
            page.locator("#messageInput").wait_for(state="visible", timeout=timeout_ms)
            api = BrowserAPI(page, args.backend_url)
            journal.save("authenticated", owner_sha256=digest(api.owner))
            model_id = None
            if connection is not None:
                journal.save("model_admission_intent")
                response = api.post("/api/v1/extensions", headers=api._mutation_headers(),
                                    json={"action": "model", "payload": connection})
                response.raise_for_status()
                resource = response.json()
                model_id = resource.get("id")
                if not model_id or resource.get("backend") != args.adapter:
                    raise ValueError("Model admission did not confirm the requested connection")
                journal.save("model_admitted", model_id=model_id)
            scope = AcceptanceProfileScope(api, adapter=args.adapter, model_id=model_id)
            try:
                journal.save("profile_selection_intent")
                with scope:
                    journal.save("profile_selected", profile=scope.fingerprint, requested_account=expected)
                    prompt = (f"Create a corporate account for given name {expected['given_name']}, family name "
                              f"{expected['family_name']}, department {expected['department']} and role {expected['role']}. "
                              f"After the account is successfully created, create a Jira ticket in project {args.project} "
                              "for that created account, then notify me by email.")
                    before = [entry["frames_in"] for entry in traffic.sockets]
                    scope.assert_unchanged()
                    contract = onboarding_contract(expected, args.project) if args.reviewed_workflow else None
                    if contract is not None:
                        journal.save("workflow_review_intent", workflow_contract=contract.model_dump(),
                                     workflow_contract_hash=contract.content_hash())
                        select_reviewed_workflow(page, contract, timeout_ms)
                        journal.save("workflow_reviewed")
                    journal.save("submission_intent")
                    job_id, chat_id = send(page, prompt, timeout_ms,
                        on_submission=lambda values: journal.save("submission_observed", **values))
                    journal.save("job_acknowledged", job_id=job_id, chat_id=chat_id)
                    if contract is not None:
                        assert_reviewed_submission(workflow_observations, contract)
                    initial = traffic.wait(page, args.backend_url, job_id, {"waiting_approval"} | TERMINAL, args.timeout)
                    if contract is not None:
                        assert_reviewed_binding(initial, contract)
                    first = exact_account_proposal(initial, expected)
                    page.reload(wait_until="domcontentloaded")
                    page.get_by_role("button", name="Approve action", exact=True).wait_for(timeout=timeout_ms)
                    traffic.refresh(page, args.backend_url, job_id)
                    if contract is not None:
                        assert_reviewed_binding(traffic.jobs[job_id], contract)
                    if exact_account_proposal(traffic.jobs[job_id], expected) != first:
                        raise ValueError("Reload changed the exact account proposal")
                    scope.assert_unchanged()
                    journal.save("account_approval_intent", account_approval_id=first["approval_id"],
                                 account_arguments_sha256=digest(json.dumps(first["arguments"], sort_keys=True)))
                    approve(page, timeout_ms)
                    waiting = traffic.wait(page, args.backend_url, job_id, {"waiting_approval"} | TERMINAL,
                                           args.timeout, approval_other_than=first["approval_id"])
                    if contract is not None:
                        assert_reviewed_binding(waiting, contract)
                    second = ticket_proposal(waiting, initial, args.project)
                    scope.assert_unchanged()
                    journal.save("ticket_approval_intent", ticket_approval_id=second["approval_id"],
                                 ticket_arguments_sha256=digest(json.dumps(second["arguments"], sort_keys=True)))
                    approve(page, timeout_ms)
                    final = traffic.wait(page, args.backend_url, job_id, TERMINAL, args.timeout)
                    if contract is not None:
                        assert_reviewed_binding(final, contract)
                    completed = completed_workflow(final, initial, waiting, args.project)
                    _assert_history_roundtrip(chat_history(page, args.backend_url, chat_id), final, chat_id=chat_id)
                    journal.save("workflow_verified", account_id=completed.account_id, ticket_key=completed.ticket_key)
                    message = wait_for_mail(private["email"], private["mailbox_password"], settings,
                                            args.timeout, lambda message: correlated_inbox(message, completed))
                    socket = socket_evidence(traffic, before)
                    if socket["frames_in"] < 1:
                        raise ValueError("No WebSocket progress was observed during this workflow")
                    scope.assert_unchanged()
                    if source_identity() != journal.data["source"]:
                        raise ValueError("Source changed during live acceptance")
                    journal.save("inbox_verified", inbox_message_id=message["Message-ID"], websocket=socket,
                                 correlated_job_account_ticket=True)
                    journal.save("profile_restoration_intent")
            finally:
                journal.save("profile_scope_finished", profile_restoration=scope.restoration)
            if scope.restoration.get("status") != "restored":
                raise ValueError("Inference profile restoration requires reconciliation")
        finally:
            if "traffic" in locals():
                journal.save(journal.data["stage"], observed_jobs={key: {
                    "status": value.get("status"),
                    "approval_id": (value.get("proposed_tool") or {}).get("approval_id")
                    if isinstance(value.get("proposed_tool"), dict) else None,
                } for key, value in traffic.jobs.items()})
            context.close()
            browser.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--adapter", choices=("local", "http", "pi", "opencode"), required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--private-env", type=Path, required=True, help="New protected reconciliation credential file")
    parser.add_argument("--connection-env", type=Path)
    parser.add_argument("--lab-env", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--frontend-url", default="http://127.0.0.1:8080")
    parser.add_argument("--backend-url", default="http://127.0.0.1:4000")
    parser.add_argument("--timeout", type=float, default=300, help="Declared budget per workflow/mail phase; no automatic extension")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--reviewed-workflow", action="store_true",
                        help="Select and verify the exact account/ticket operations through the composer UI")
    args = parser.parse_args(argv)
    try:
        if not math.isfinite(args.timeout) or args.timeout <= 0 or not re.fullmatch(r"[A-Z][A-Z0-9_]*", args.project):
            raise ValueError("Explicit valid project and positive finite timeout are required")
        args.frontend_url, args.backend_url = base_url(args.frontend_url), base_url(args.backend_url)
        if args.report.exists() or args.private_env.exists() or args.report.resolve() == args.private_env.resolve():
            raise ValueError("Report/private files must be distinct new paths; reconcile previous attempts")
        settings = lab_settings(args.lab_env)
        connection = connection_payload(args.connection_env, args.adapter)
        identity = source_identity()
    except (ValueError, OSError):
        parser.error("Acceptance configuration is invalid; no setup or dispatch performed")
    run = secrets.token_hex(5)
    private = {"email": f"acceptance.{run}@{settings.mailbox_lab_domain}",
               "password": secrets.token_urlsafe(24), "mailbox_password": secrets.token_urlsafe(24)}
    journal = Journal(args.report, {"schema": "onboarding-browser-e2e.v2", "run": run,
        "project": args.project, "adapter": args.adapter, "source": identity,
        "timeout_seconds": args.timeout, "result": "incomplete", "stage": "intent_recorded",
        "reviewed_workflow": args.reviewed_workflow,
        "recipient_sha256": digest(private["email"]), "reconciliation_required": False})
    try:
        write_exclusive(args.private_env, "\n".join(f"{key}={value}" for key, value in {
            "ACCEPTANCE_EMAIL": private["email"], "ACCEPTANCE_PASSWORD": private["password"],
            "MAILBOX_PASSWORD": private["mailbox_password"]}.items()) + "\n")
        create_lab_mailbox(private["email"], private["mailbox_password"], settings, journal)
        execute(args, journal, settings, private, connection)
        if source_identity() != identity:
            raise ValueError("Source changed before final checkpoint")
        journal.save("complete", result="pass", reconciliation_required=False,
                     retained_fixtures=True, source_unchanged=True)
    except Exception as exc:
        # Provider errors may include private URLs/tokens/recipients. Retain only
        # stage and immutable identities for reconciliation, never exception text.
        journal.save(journal.data["stage"], result="incomplete", error_type=type(exc).__name__,
                     reconciliation_required=True)
    print("Acceptance", journal.data["result"], "report:", args.report)
    return 0 if journal.data["result"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
