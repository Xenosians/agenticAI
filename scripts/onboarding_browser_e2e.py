#!/usr/bin/env python3
"""Browser end-to-end: new user with a lab mailbox -> account -> Jira ticket -> mail.

Everything a person would do happens in the real frontend (Chromium via
Playwright):

1. Register a new application user whose email is a fresh local lab mailbox.
2. Open the verification link from the email that arrived in that mailbox.
3. Sign in, open Models, and assign specialists to connections (the selected
   model runs the hub; Jira/ticket work runs on the Groq connection).
4. In chat, request a corporate account followed by a Jira ticket about it.
   Approve the account, then approve the ticket that the workflow proposes.
5. Confirm the completion email naming the account and the ticket arrived in
   the user's lab mailbox.

Trusted setup that has no UI is done directly and stated in the report: the
lab mailbox is created with Maddy's admin CLI (the product's mailbox tool
requires an existing app user, and this user does not exist yet), and model
connections are copied from an existing account's encrypted store into the
new account so no API key passes through this script's output.

Usage (AI repository root, AI service Python):
  python scripts/onboarding_browser_e2e.py --project SUP [--headed]
"""
from __future__ import annotations

import argparse
import email
import imaplib
import json
import os
import re
import secrets
import subprocess
import sys
import time
from datetime import datetime, timezone
from email.header import decode_header, make_header
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

MADDY = ["docker", "exec", "-i", "itsm-mailbox-lab", "/bin/maddy", "-config", "/data/maddy.conf"]
BACKEND_REPO = Path(os.getenv("ACCEPTANCE_BACKEND_REPO", str(PROJECT_ROOT.parents[1] / "agenticBackend" / "itsm_backend")))
REPORT_DIR = PROJECT_ROOT / ".runtime" / "acceptance" / "onboarding-e2e"
TERMINAL = {"completed", "failed", "denied", "outcome_unknown", "reconciliation_required"}
GROQ_ROLES = ("jira-specialist", "ticket-specialist")


def step(report: dict, name: str, **detail) -> None:
    entry = {"step": name, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **detail}
    report["steps"].append(entry)
    print(f"[{entry['at']}] {name} " + json.dumps(detail, default=str)[:300], flush=True)


# ---------------------------------------------------------------- lab mailbox
def create_lab_mailbox(address: str) -> str:
    password = secrets.token_urlsafe(24)
    for args, stdin in ((["creds", "create", address], password + "\n"), (["imap-acct", "create", address], None)):
        done = subprocess.run(MADDY + args, input=stdin, capture_output=True, text=True, timeout=60)
        if done.returncode != 0:
            raise RuntimeError(f"maddy {' '.join(args[:2])} failed")
    return password


def wait_for_mail(address: str, password: str, subject_part: str, timeout_s: float) -> email.message.Message:
    port = int(os.environ.get("MAILBOX_IMAP_PORT", "1143"))
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with imaplib.IMAP4("127.0.0.1", port, timeout=15) as imap:
            imap.login(address, password)
            imap.select("INBOX", readonly=True)
            status, data = imap.search(None, "ALL")
            for message_id in reversed(data[0].split() if status == "OK" and data[0] else []):
                _, parts = imap.fetch(message_id, "(RFC822)")
                message = email.message_from_bytes(parts[0][1])
                if subject_part.lower() in str(make_header(decode_header(message.get("Subject", "")))).lower():
                    return message
        time.sleep(3)
    raise TimeoutError(f"no email with subject containing {subject_part!r} within {timeout_s:.0f}s")


def text_body(message: email.message.Message) -> str:
    for part in message.walk():
        if part.get_content_type() == "text/plain":
            return part.get_payload(decode=True).decode(errors="replace")
    return ""


# ---------------------------------------------------------------- model mix
def copy_connections(source_owner: str, target_owner: str) -> dict:
    """Copy the source account's Claude and Groq HTTP connections into the new account."""
    from services.extensions.service import extension_store

    store = extension_store()
    copied: dict[str, str] = {}
    for row in store.list(source_owner):
        if row.get("kind") != "model" or row.get("backend") != "http" or row.get("state") != "ready":
            continue
        kind = "claude" if row.get("api_format") == "anthropic" else "groq" if "api.groq.com" in row.get("url", "") else None
        if kind is None or kind in copied:
            continue
        source = store.get(source_owner, row["id"], secret=True)
        api_key = source.pop("api_key", "")
        data = {k: v for k, v in source.items() if k not in {"id", "kind", "credential_configured"}}
        copied[kind] = store.put(target_owner, "model", data, api_key)["id"]
    if set(copied) != {"claude", "groq"}:
        raise RuntimeError(f"source account needs a Claude and a Groq connection, found {sorted(copied)}")
    return copied


def remove_copied_connections(owner: str, connection_ids: dict) -> int:
    """Delete the key copies made for the test user; the user and evidence remain."""
    from services.extensions.service import extension_store

    store = extension_store()
    removed = 0
    for identity in connection_ids.values():
        try:
            store.delete(owner, identity)
            removed += 1
        except ValueError:
            pass
    return removed


def owner_for(email_address: str) -> str:
    import hashlib

    import httpx
    from dotenv import dotenv_values

    config = dotenv_values(BACKEND_REPO / ".env")
    key = hashlib.sha256(email_address.strip().lower().encode()).hexdigest()
    response = httpx.post(
        config["SURREALDB_URL"].rstrip("/") + "/rpc",
        auth=(config["SURREALDB_USERNAME"], config["SURREALDB_PASSWORD"]),
        headers={"surreal-ns": config["SURREALDB_NAMESPACE"], "surreal-db": config["SURREALDB_DATABASE"]},
        json={"id": 1, "method": "query",
              "params": ["SELECT user_id FROM ONLY type::record('auth_identity', $id);", {"id": key}]},
        timeout=10, trust_env=False,
    )
    result = response.json()["result"][0]["result"]
    if not isinstance(result, dict) or not result.get("user_id"):
        raise RuntimeError("app user not found")
    return result["user_id"]


# ---------------------------------------------------------------- browser
class Jobs:
    """Job snapshots the frontend itself fetched (never polled by this script)."""

    def __init__(self, page, backend_url: str) -> None:
        self.snapshots: dict[str, dict] = {}
        self.backend_url = backend_url.rstrip("/")
        page.on("response", self._on_response)

    def _on_response(self, response) -> None:
        if "/api/v2/jobs/" not in response.url or response.request.method != "GET":
            return
        try:
            payload = response.json()
        except Exception:
            return
        if isinstance(payload, dict) and isinstance(payload.get("job_id"), str):
            self.snapshots[payload["job_id"]] = payload

    def refresh(self, page, job_id: str) -> None:
        """Read the job with the browser's own session (the UI may update over WebSocket)."""
        response = page.request.get(f"{self.backend_url}/api/v2/jobs/{job_id}")
        if response.ok:
            payload = response.json()
            if isinstance(payload, dict) and payload.get("job_id") == job_id:
                self.snapshots[job_id] = payload

    def wait(self, page, job_id: str, statuses: set[str], timeout_s: float, *, approval_other_than=None) -> dict:
        deadline = time.monotonic() + timeout_s
        polls = 0
        while time.monotonic() < deadline:
            polls += 1
            if polls % 6 == 0:
                self.refresh(page, job_id)
            job = self.snapshots.get(job_id) or {}
            approval = (job.get("proposed_tool") or {}).get("approval_id") if isinstance(job.get("proposed_tool"), dict) else None
            if job.get("status") in statuses and (approval_other_than is None or approval != approval_other_than):
                return job
            page.wait_for_timeout(500)
        raise TimeoutError(f"job did not reach {sorted(statuses)}; last={self.snapshots.get(job_id, {}).get('status')}")


def fill_auth(page, label: str, value: str) -> None:
    field = page.locator("label.auth-field", has_text=label).locator("input")
    field.fill(value)
    field.press("Tab")  # the auth field commits on change


def send_prompt(page, prompt: str, timeout_ms: int) -> str:
    page.locator("#messageInput").fill(prompt)
    send = page.get_by_role("button", name="Send message")
    deadline = time.monotonic() + timeout_ms / 1000
    while not send.is_enabled():
        if time.monotonic() > deadline:
            raise TimeoutError("Send button never became enabled")
        page.wait_for_timeout(100)
    with page.expect_response(lambda r: r.url.rstrip("/").endswith("/api/v1/jobs") and r.request.method == "POST",
                              timeout=timeout_ms) as info:
        send.click()
    response = info.value
    if response.status >= 400:
        raise RuntimeError(f"job submission returned HTTP {response.status}")
    return response.json()["job_id"]


def approve_in_ui(page, timeout_ms: int) -> None:
    button = page.get_by_role("button", name="Approve action", exact=True).last
    button.wait_for(state="visible", timeout=timeout_ms)
    button.click()


def assign_roles_in_ui(page, groq_name: str, timeout_ms: int) -> list[str]:
    page.keyboard.press("Control+Comma")
    page.get_by_role("button", name="Personal", exact=True).click()
    section = page.locator(".plugin-role-section")
    section.wait_for(state="visible", timeout=timeout_ms)
    assigned = []
    for role in GROQ_ROLES:
        label = role.replace("-specialist", "").capitalize()
        row = section.locator(".plugin-role-row", has=page.locator("strong", has_text=re.compile(f"^{label}$")))
        with page.expect_response(lambda r: r.url.endswith("/api/v1/extensions") and r.request.method == "POST",
                                  timeout=timeout_ms):
            row.locator("select").select_option(label=groq_name)
        page.wait_for_timeout(400)
        assigned.append(role)
    # Back to chat through the command palette's "Focus message composer".
    page.keyboard.press("Control+k")
    page.locator("#commandSearch").fill("composer")
    page.locator("button.command-center-row", has_text="Focus message composer").click()
    page.locator("#messageInput").wait_for(state="visible", timeout=timeout_ms)
    return assigned


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--project", required=True, help="Jira project key for the ticket")
    parser.add_argument("--frontend-url", default="http://127.0.0.1:8080")
    parser.add_argument("--backend-url", default="http://127.0.0.1:4000")
    parser.add_argument("--source-login-env", type=Path,
                        default=BACKEND_REPO / ".runtime" / "acceptance" / "login.env",
                        help="env file with ACCEPTANCE_EMAIL of the account whose connections are copied")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()

    from dotenv import dotenv_values

    domain = dotenv_values(PROJECT_ROOT / ".env").get("MAILBOX_LAB_DOMAIN") or "itsm.test"
    run = datetime.now(timezone.utc).strftime("%m%d%H%M%S")
    user_email = f"e2e.{run}@{domain}"
    app_password = secrets.token_urlsafe(18)
    given, family = "Onboard", f"Run{run[-6:]}"
    timeout_ms = int(args.timeout * 1000)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"run": run, "project": args.project, "user_email": user_email, "steps": [], "result": "running"}
    report_path = REPORT_DIR / f"onboarding-{run}.json"
    shots = REPORT_DIR / f"onboarding-{run}"
    shots.mkdir(exist_ok=True)

    mailbox_password = create_lab_mailbox(user_email)
    step(report, "lab mailbox created", email=user_email, how="maddy admin CLI (trusted setup)")

    copied: dict = {}
    test_owner: str | None = None
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not args.headed)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        jobs = Jobs(page, args.backend_url)
        try:
            # 1. Register in the UI.
            page.goto(args.frontend_url, wait_until="domcontentloaded")
            page.get_by_role("button", name="Create account", exact=True).click()
            fill_auth(page, "Display name", f"{given} {family}")
            fill_auth(page, "Email", user_email)
            fill_auth(page, "Password", app_password)
            with page.expect_response(lambda r: r.url.endswith("/api/v1/auth/register"), timeout=timeout_ms) as info:
                page.get_by_role("button", name="Register", exact=True).click()
            registered = info.value.json()
            step(report, "registered in UI", http=info.value.status, mail_delivery=registered.get("mail_delivery"))
            page.screenshot(path=str(shots / "01-registered.png"))

            # 2. Verification email -> link.
            verification = wait_for_mail(user_email, mailbox_password, "Verify your Agentic ITSM account", 120)
            link = re.search(r"https?://\S+verify_token=\S+", text_body(verification))
            if not link:
                raise AssertionError("verification email has no link")
            step(report, "verification email received in lab mailbox", subject=verification["Subject"])
            page.goto(link.group(0), wait_until="domcontentloaded")
            page.locator(".auth-notice").wait_for(state="visible", timeout=timeout_ms)
            step(report, "email verified via link", notice=page.locator(".auth-notice").inner_text())

            # 3. Sign in.
            fill_auth(page, "Email", user_email)
            fill_auth(page, "Password", app_password)
            page.get_by_role("button", name="Sign in", exact=True).click()
            page.locator("#messageInput").wait_for(state="visible", timeout=timeout_ms)
            step(report, "signed in")

            # 4. Model mix: connections copied from the source account; roles assigned in the UI.
            source_email = dotenv_values(args.source_login_env)["ACCEPTANCE_EMAIL"]
            test_owner = owner_for(user_email)
            connections = copied = copy_connections(owner_for(source_email), test_owner)
            from services.extensions.service import extension_store

            store = extension_store()
            owner = test_owner
            store.select(owner, connections["claude"], None)
            groq_name = store.get(owner, connections["groq"])["name"]
            step(report, "connections copied (trusted setup)", selected="claude", available=sorted(connections))
            assigned = assign_roles_in_ui(page, groq_name, timeout_ms)
            roles = store.role_assignments(owner)
            if any(roles.get(role) != connections["groq"] for role in GROQ_ROLES):
                raise AssertionError(f"role assignment not saved: {roles}")
            step(report, "specialist models assigned in UI", groq_roles=assigned, hub="claude")
            page.screenshot(path=str(shots / "02-models.png"))

            # 5. Account -> ticket workflow in chat.
            prompt = (f"Create a corporate account for given name {given}, family name {family}, department IT "
                      f"and role Engineer. After the account is successfully created, create a Jira ticket in "
                      f"project {args.project} for that created account, then notify me by email.")
            job_id = send_prompt(page, prompt, timeout_ms)
            step(report, "request sent", job_id=job_id)
            first = jobs.wait(page, job_id, {"waiting_approval"} | TERMINAL, args.timeout)
            proposal = first.get("proposed_tool") or {}
            if first.get("status") != "waiting_approval" or proposal.get("tool") != "account_create":
                raise AssertionError(f"expected account_create approval, got {first.get('status')} {proposal.get('tool')}")
            step(report, "account approval shown", tool=proposal["tool"])
            page.screenshot(path=str(shots / "03-account-approval.png"))
            approve_in_ui(page, timeout_ms)

            second = jobs.wait(page, job_id, {"waiting_approval"} | TERMINAL, args.timeout,
                               approval_other_than=proposal.get("approval_id"))
            ticket_proposal = second.get("proposed_tool") or {}
            if second.get("status") != "waiting_approval" or ticket_proposal.get("tool") != "ticket_create":
                raise AssertionError(f"expected ticket_create approval, got {second.get('status')} "
                                     f"{ticket_proposal.get('tool')} error={second.get('error')}")
            step(report, "account created; ticket approval shown", tool=ticket_proposal["tool"],
                 project=(ticket_proposal.get("arguments") or {}).get("project_key"))
            page.screenshot(path=str(shots / "04-ticket-approval.png"))
            approve_in_ui(page, timeout_ms)

            final = jobs.wait(page, job_id, TERMINAL, args.timeout)
            result = final.get("result") or {}
            workflow = result.get("workflow") or {}
            ticket_key = ((result.get("approval_execution") or {}).get("result") or {}).get("ticket_key")
            account = (workflow.get("account") or {}).get("user_id")
            if final.get("status") != "completed" or not ticket_key or not account:
                raise AssertionError(f"workflow ended {final.get('status')} account={account} ticket={ticket_key}")
            step(report, "workflow completed", account=account, ticket=ticket_key, workflow_state=workflow.get("state"))
            page.screenshot(path=str(shots / "05-completed.png"))

            # 6. Completion mail in the same lab mailbox.
            done = wait_for_mail(user_email, mailbox_password, "request completed", 180)
            body = text_body(done)
            if ticket_key not in body or account not in body:
                raise AssertionError("completion email does not name the account and ticket")
            step(report, "completion email received", subject=done["Subject"], names_account=True, names_ticket=True)
            report.update(result="pass", account=account, ticket=ticket_key, job_id=job_id)
        except Exception as exc:
            report.update(result="fail", error=f"{type(exc).__name__}: {str(exc)[:400]}")
            try:
                page.screenshot(path=str(shots / "failure.png"))
            except Exception:
                pass
            print("FAIL", report["error"], flush=True)
        finally:
            context.close()
            browser.close()
            if test_owner and copied:
                report["copied_connections_removed"] = remove_copied_connections(test_owner, copied)
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print("report:", report_path, flush=True)
    return 0 if report["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
