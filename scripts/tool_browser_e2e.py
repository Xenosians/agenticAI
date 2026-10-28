#!/usr/bin/env python3
"""Browser end-to-end for every registered tool, one request at a time.

Each case is typed into the real frontend (Chromium via Playwright) as a
natural request, exactly as a person would. For tools that change something,
the approval card appears and the harness clicks "Approve action" in the UI.
Every case then has to finish with the expected tool's own structured result:
the Hub must route to the right specialist, SemanticGuard must allow it, the
ToolGateway/MCP must execute it, and the provider must report success.

The cases (prompt, expected tool, approval) are the same ones as the live API
matrix in scripts/tool_matrix.py, so both suites test the same requests.

The browser does all the work. The harness only reads what the frontend
itself fetched (job snapshots), the chat history, and disposable lab fixtures
(git working tree, Jira statuses). The live-update WebSocket is checked on
every case.

Usage (AI repository root, AI service Python):
  python scripts/tool_browser_e2e.py [--only case-id ...] [--resume] [--headed]

Exit status (same rules as scripts/tool_matrix.py):
  0  every selected case passed, and a full run covered every registered tool
  1  at least one case failed
  2  incomplete: a case was blocked, skipped or plan-limited, or a tool is uncovered
An unknown or empty --only selection is rejected before the browser starts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import dotenv_values  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

from scripts.cross_repo_acceptance import _assert_history_roundtrip, _tool_name  # noqa: E402
from scripts import tool_matrix as matrix  # noqa: E402
from scripts.tool_matrix import (  # noqa: E402
    ATLASSIAN_ADMIN_TOOLS,
    GIT_MUTATION_CASES,
    RUN,
    TERMINAL_OK,
    VERDICTS,
    acceptance_summary,
    atlassian_admin_configured,
    build_cases,
    discover_fixtures,
    load_matrix_report,
    plan_limited,
    prepare_git_fixture,
    provider_ok,
    record_created,
    select_matrix_cases,
    source_identity,
    worker_evidence,
)
from tools.registry import TOOLS  # noqa: E402

BACKEND_REPO = Path(os.getenv("ACCEPTANCE_BACKEND_REPO", str(PROJECT_ROOT.parents[1] / "agenticBackend" / "itsm_backend")))
REPORT = PROJECT_ROOT / ".runtime" / "acceptance" / "tool-browser-e2e.json"
TERMINAL = {"completed", "failed", "denied", "outcome_unknown", "reconciliation_required"}
BROWSER_REPORT_SCHEMA = "tool-browser-e2e.v2"


def execution_context(args) -> dict:
    """Bind evidence to its app origins/configuration without storing credentials."""
    origins = {}
    for name in ("frontend_url", "backend_url"):
        value = getattr(args, name).rstrip("/")
        parsed = urlsplit(value)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment):
            raise ValueError(f"{name} must be an HTTP(S) base URL without credentials, query or fragment")
        origins[name] = value
    return {**origins, "login_configuration": str(args.login_env.resolve()),
            "git_fixture": str(matrix.GIT_FIXTURE.resolve())}


def bind_principal(report: dict, email: str) -> None:
    """Changing the acceptance account requires a new report, not merged evidence."""
    principal = hashlib.sha256(email.strip().casefold().encode()).hexdigest()
    prior = report.get("principal_sha256")
    if prior is not None and prior != principal:
        raise ValueError("Acceptance account changed; use a new report")
    report["principal_sha256"] = principal


def unattempted_cases(report: dict, selected: list) -> list:
    # Even failed/interrupted rows may have performed a side effect. A resume
    # continues untouched cases; it never creates an independent replacement job.
    return [case for case in selected if case[0] not in report["results"]]


def save_browser_report(report: dict, path: Path, selected: list, *, full_scope: bool,
                        source_unchanged: bool = False) -> None:
    rows = report["results"].values()
    report["summary"] = {verdict: sum(row.get("verdict") == verdict for row in rows) for verdict in VERDICTS}
    report["registered_tools"] = len(TOOLS)
    report["selection"] = {"scope": "full" if full_scope else "selected", "cases": [case[0] for case in selected]}
    report["acceptance"] = acceptance_summary(report["results"], selected, registered_tools=TOOLS,
                                               full_scope=full_scope, source_unchanged=source_unchanged)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def socket_evidence(traffic, before: list[int]) -> dict:
    """Count only frames received during this case, including persistent sockets."""
    incoming = sum(max(0, entry["frames_in"] - (before[index] if index < len(before) else 0))
                   for index, entry in enumerate(traffic.sockets))
    return {"opened": len(traffic.sockets) - len(before), "frames_in": incoming}


class BrowserTraffic:
    """What the frontend itself fetched and the sockets it opened."""

    def __init__(self, page) -> None:
        self.jobs: dict[str, dict] = {}
        self.sockets: list[dict] = []
        page.on("response", self._on_response)
        page.on("websocket", self._on_websocket)

    def _on_response(self, response) -> None:
        if "/api/v2/jobs/" not in response.url or response.request.method != "GET":
            return
        try:
            payload = response.json()
        except Exception:
            return
        if isinstance(payload, dict) and isinstance(payload.get("job_id"), str):
            self.jobs[payload["job_id"]] = payload

    def _on_websocket(self, socket) -> None:
        entry = {"url": socket.url.split("?")[0], "frames_in": 0}
        self.sockets.append(entry)

        def received(_payload) -> None:
            entry["frames_in"] += 1

        socket.on("framereceived", received)

    def refresh(self, page, backend_url: str, job_id: str) -> None:
        response = page.request.get(f"{backend_url}/api/v2/jobs/{job_id}")
        if response.ok:
            payload = response.json()
            if isinstance(payload, dict) and payload.get("job_id") == job_id:
                self.jobs[job_id] = payload

    def wait(self, page, backend_url: str, job_id: str, statuses: set[str], timeout_s: float,
             *, approval_other_than: str | None = None) -> dict:
        deadline = time.monotonic() + timeout_s
        ticks = 0
        while time.monotonic() < deadline:
            ticks += 1
            if ticks % 8 == 0:  # the UI may be live over WebSocket; re-read with its session
                self.refresh(page, backend_url, job_id)
            job = self.jobs.get(job_id) or {}
            proposal = job.get("proposed_tool") if isinstance(job.get("proposed_tool"), dict) else {}
            if job.get("status") in statuses and (
                    job.get("status") in TERMINAL or approval_other_than is None
                    or proposal.get("approval_id") != approval_other_than):
                # A terminal outcome belongs to the whole job even when it
                # retains the previous approval proposal for audit/history.
                return job
            page.wait_for_timeout(500)
        raise TimeoutError(f"job did not reach {sorted(statuses)}; last={self.jobs.get(job_id, {}).get('status')}")


    def wait_after_approval(self, page, backend_url: str, job_id: str, approval_id: str | None,
                            timeout_s: float) -> dict:
        deadline = time.monotonic() + timeout_s
        ticks = 0
        while time.monotonic() < deadline:
            ticks += 1
            if ticks % 8 == 0:
                self.refresh(page, backend_url, job_id)
            job = self.jobs.get(job_id) or {}
            proposal = job.get("proposed_tool") if isinstance(job.get("proposed_tool"), dict) else {}
            if job.get("status") in TERMINAL:
                return job
            if job.get("status") == "waiting_approval" and proposal.get("approval_id") not in (None, approval_id):
                return job
            page.wait_for_timeout(500)
        raise TimeoutError(f"job did not finish after approval; last={self.jobs.get(job_id, {}).get('status')}")


def enable_required_operations(page, frontend_url: str) -> None:
    """Expose reviewed workflow controls only in this acceptance page.

    The product hides these controls by default. Preserve its actual runtime
    configuration, including future fields, and append the explicit test opt-in
    to the response for this frontend's config.js. No file or server changes.
    Failed configuration responses retain their original status and body.
    """
    config_url = urljoin(frontend_url.rstrip("/") + "/", "config.js")
    pattern = re.compile(rf"^{re.escape(config_url)}(?:\?[^#]*)?$")

    def configured(route):
        response = route.fetch()
        if not response.ok:
            route.fulfill(response=response)
            return
        suffix = b"""
;(() => {
  const config = window.AGENTIC_CONFIG;
  if (typeof config !== 'object' || config === null || Array.isArray(config)) {
    throw new Error('Acceptance runtime configuration is missing or invalid.');
  }
  window.AGENTIC_CONFIG = Object.freeze({...config, requiredOperations: true});
})();
"""
        headers = {key: value for key, value in response.headers.items()
                   if key.lower() not in {"content-length", "content-encoding", "etag", "last-modified"}}
        headers["cache-control"] = "no-store"
        route.fulfill(response=response, headers=headers, body=response.body() + suffix)

    page.route(pattern, configured)


def sign_in(page, frontend_url: str, email: str, password: str, timeout_ms: int) -> None:
    enable_required_operations(page, frontend_url)
    page.goto(frontend_url, wait_until="domcontentloaded")
    try:
        page.locator("#messageInput").wait_for(state="visible", timeout=4000)
        return
    except Exception:
        pass
    for label, value in (("Email", email), ("Password", password)):
        field = page.locator("label.auth-field", has_text=label).locator("input")
        field.fill(value)
        field.press("Tab")
    page.get_by_role("button", name="Sign in", exact=True).click()
    page.locator("#messageInput").wait_for(state="visible", timeout=timeout_ms)


def new_chat(page, timeout_ms: int) -> str:
    """Open an empty chat through the header menu's command.

    A saved chat that kept its default title is also a button named
    "New chat", so the command is located inside the menu. The menu is only
    opened when closed: a case that failed after opening it leaves it open.
    An already empty selected chat is deliberately reused by the product.
    Otherwise require its real creation receipt and the exact selected identity
    before returning; detached old messages alone cannot prove navigation.
    """
    workspace = page.locator("main.workspace")
    previous = workspace.get_attribute("data-chat-id")
    if not previous:
        raise AssertionError("The workspace has no selected chat identity")
    reuse = page.locator(".message-row").count() == 0
    more = page.get_by_role("button", name="More", exact=True)
    if more.get_attribute("aria-expanded") != "true":
        more.click()
    command = page.locator(".header-menu").get_by_role("button", name="New chat", exact=True)
    if reuse:
        command.click()
        chat_id = previous
    else:
        with page.expect_response(
            lambda response: urlsplit(response.url).path.rstrip("/") == "/api/v1/chats"
            and response.request.method == "POST", timeout=timeout_ms
        ) as creation:
            command.click()
        response = creation.value
        if response.status != 201:
            raise RuntimeError(f"chat creation returned HTTP {response.status}")
        payload = response.json()
        chat = payload.get("chat") if isinstance(payload, dict) else None
        chat_id = chat.get("chat_id") if isinstance(chat, dict) else None
        if not isinstance(chat_id, str) or not chat_id.strip() or chat_id == previous:
            raise AssertionError("Chat creation returned an invalid new chat identity")
    page.wait_for_function(
        "chatId => document.querySelector('main.workspace')?.getAttribute('data-chat-id') === chatId",
        arg=chat_id, timeout=timeout_ms,
    )
    page.locator("#messageInput").wait_for(state="visible", timeout=timeout_ms)
    page.locator(".message-row").first.wait_for(state="detached", timeout=timeout_ms)
    return chat_id


def send(page, prompt: str, timeout_ms: int, *, on_submission) -> tuple[str, str]:
    page.locator("#messageInput").fill(prompt)
    button = page.get_by_role("button", name="Send message")
    deadline = time.monotonic() + timeout_ms / 1000
    while not button.is_enabled():
        if time.monotonic() > deadline:
            raise TimeoutError("Send button never became enabled")
        page.wait_for_timeout(100)
    def observed(request):
        if request.url.rstrip("/").endswith("/api/v1/jobs") and request.method == "POST":
            try:
                payload = json.loads(request.post_data or "{}")
            except (TypeError, ValueError):
                payload = {}
            # Retain only reconciliation identities, never request headers/login data.
            on_submission({key: payload[key] for key in ("chat_id", "submission_id")
                           if isinstance(payload, dict) and isinstance(payload.get(key), str)})

    page.on("request", observed)
    try:
        with page.expect_response(lambda r: r.url.rstrip("/").endswith("/api/v1/jobs") and r.request.method == "POST",
                                  timeout=timeout_ms) as info:
            button.click()
        response = info.value
    finally:
        page.remove_listener("request", observed)
    if response.status >= 400:
        raise RuntimeError(f"job submission returned HTTP {response.status}")
    # The browser names the chat in its request; the response carries the job.
    submitted = json.loads(response.request.post_data or "{}")
    return response.json()["job_id"], submitted.get("chat_id") or ""


def chat_history(page, backend_url: str, chat_id: str) -> dict:
    response = page.request.get(f"{backend_url}/api/v2/chats/{chat_id}/history?limit=200")
    if not response.ok:
        raise RuntimeError(f"chat history returned HTTP {response.status}")
    return response.json()


def approve(page, timeout_ms: int) -> None:
    button = page.get_by_role("button", name="Approve action", exact=True).last
    button.wait_for(state="visible", timeout=timeout_ms)
    button.click()


def run_case(page, traffic: BrowserTraffic, args, case: tuple, context: dict, *, row: dict, checkpoint) -> dict:
    identity, tool, prompt, approve_mode = case
    message = prompt.format(**context)
    row.update(tool=tool, prompt=message, stage="preparing_browser")
    started = time.monotonic()
    timeout_ms = int(args.timeout * 1000)

    expected_chat_id = new_chat(page, timeout_ms)
    sockets_before = [entry["frames_in"] for entry in traffic.sockets]

    def observed(identities):
        row.update(identities)
        row["submission_outcome"] = "outcome_unknown"
        checkpoint()

    row.update(stage="submitting", submission_outcome="outcome_unknown", reconciliation_required=True)
    checkpoint()  # Durable before a click can create work, even if no response arrives.
    job_id, chat_id = send(page, message, timeout_ms, on_submission=observed)
    row.update(job_id=job_id, chat_id=chat_id, submission_outcome="acknowledged", stage="waiting_for_job")
    checkpoint()
    if chat_id != expected_chat_id:
        raise AssertionError("The job was submitted to a different chat than the selected empty chat")
    first = traffic.wait(page, args.backend_url, job_id, {"waiting_approval"} | TERMINAL, args.timeout)
    row["first_status"] = first.get("status")
    row["first_tool"] = _tool_name(first)

    if approve_mode == "auto":
        approve_mode = True if first.get("status") == "waiting_approval" else None
    final = first
    if approve_mode is not None:
        if first.get("status") != "waiting_approval":
            raise AssertionError(f"expected an approval card, job ended {first.get('status')}: {first.get('error')}")
        if _tool_name(first) != tool:
            raise AssertionError(f"approval card proposes {_tool_name(first)} instead of {tool}")
        approval_id = (first.get("proposed_tool") or {}).get("approval_id")
        row.update(stage="approving", approval_id=approval_id)
        checkpoint()
        approve(page, timeout_ms)
        # Done when the job is terminal, or waiting on a *different* approval
        # (a follow-up step); the approval just clicked no longer counts.
        final = traffic.wait_after_approval(page, args.backend_url, job_id, approval_id, args.timeout)
    row["status"] = final.get("status")
    row["stage"] = "verifying"
    checkpoint()

    if chat_id and final.get("chat_id") != chat_id:
        raise AssertionError("durable job is not bound to the chat the browser submitted to")
    history = chat_history(page, args.backend_url, final["chat_id"])
    _assert_history_roundtrip(history, final, chat_id=final["chat_id"])
    worker = worker_evidence(final)
    used = (worker or {}).get("proposed_tool") or _tool_name(final)
    ok, detail = provider_ok(worker)
    row["provider"] = detail
    if used != tool:
        raise AssertionError(f"used {used} instead of {tool}")
    limited = plan_limited(worker)
    if limited:
        row["verdict"] = "plan_limited"
        row["reason"] = limited
    elif final.get("status") not in TERMINAL_OK or not ok:
        raise AssertionError(f"status={final.get('status')} {detail}")

    row["websocket"] = socket_evidence(traffic, sockets_before)
    if not limited:
        record_created(tool, worker["tool_result"], context)
    if row["websocket"]["frames_in"] < 1:
        raise AssertionError("No live-update WebSocket frame was observed during this case")
    if not limited:
        row["verdict"] = "pass"
    row.update(stage="verified", reconciliation_required=False)
    row["seconds"] = round(time.monotonic() - started, 1)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--resume", action="store_true",
                        help="continue only unattempted cases; recorded outcomes require reconciliation")
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--frontend-url", default="http://127.0.0.1:8080")
    parser.add_argument("--backend-url", default="http://127.0.0.1:4000")
    parser.add_argument("--login-env", type=Path, default=BACKEND_REPO / ".runtime" / "acceptance" / "login.env")
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()
    try:
        if args.only == []:
            raise ValueError("--only needs at least one case identity")
        # Reject invalid scope before reading credentials, repositories or starting a browser.
        select_matrix_cases(build_cases(RUN, matrix.PROJECT), args.only or [])
        identity = source_identity()
        report = load_matrix_report(args.report, resume=args.resume, identity=identity,
                                    run=RUN, project=matrix.PROJECT, schema=BROWSER_REPORT_SCHEMA,
                                    execution_context=execution_context(args))
        selected = select_matrix_cases(build_cases(report["run"], report["project"]), args.only or [])
    except ValueError as exc:
        parser.error(str(exc))

    # Shared fixture helpers resolve their names from the original run, too.
    matrix.RUN = report["run"]
    report.setdefault("started_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    report["run_state"] = "in_progress"
    shots = args.report.with_suffix("")
    pending = unattempted_cases(report, selected)

    def save(*, final=False) -> None:
        save_browser_report(report, args.report, selected, full_scope=args.only is None,
                            source_unchanged=final and source_identity() == identity)

    save()
    try:
        if pending:
            login = dotenv_values(args.login_env)
            bind_principal(report, login["ACCEPTANCE_EMAIL"])
            context = report.setdefault("context", {})
            for key, value in (("org_id", os.environ.get("ATLASSIAN_ORG_ID")),
                               ("account_id", os.environ.get("ATLASSIAN_ACCOUNT_ID")),
                               ("assignee", os.environ.get("ACCEPTANCE_JIRA_ASSIGNEE")),
                               ("status", os.environ.get("ACCEPTANCE_JIRA_STATUS"))):
                if value and key not in context:
                    context[key] = value
            context.setdefault("user_email", login["ACCEPTANCE_EMAIL"])
            discover_fixtures(context)
            if any(case[0] in GIT_MUTATION_CASES for case in pending):
                # Do not alter a fixture whose earlier mutation may still be in flight.
                attempted_git = [case for case in build_cases(report["run"], report["project"])
                                 if case[0] in GIT_MUTATION_CASES and case[0] in report["results"]]
                preparation = report.get("git_fixture_preparation", {})
                if not attempted_git and not preparation:
                    report["git_fixture_preparation"] = {"status": "outcome_unknown"}
                    save()  # Fixture preparation itself mutates the disposable repository.
                    preparation = {"status": prepare_git_fixture()}
                    report["git_fixture_preparation"] = preparation
                    save()
                if attempted_git or preparation.get("status") != "ready":
                    for case in pending:
                        if case[0] in GIT_MUTATION_CASES:
                            report["results"][case[0]] = {
                                "tool": case[1], "verdict": "blocked", "reconciliation_required": True,
                                "reason": "Reconcile the previously attempted Git fixture before new mutations.",
                            }
                    pending = unattempted_cases(report, pending)
                    save()
            admin_ready = atlassian_admin_configured()
            shots.mkdir(parents=True, exist_ok=True)

            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=not args.headed)
                try:
                    page = browser.new_context(viewport={"width": 1440, "height": 900}).new_page()
                    traffic = BrowserTraffic(page)
                    sign_in(page, args.frontend_url, login["ACCEPTANCE_EMAIL"], login["ACCEPTANCE_PASSWORD"],
                            int(args.timeout * 1000))
                    for index, case in enumerate(pending, 1):
                        case_id, tool = case[0], case[1]
                        if tool in ATLASSIAN_ADMIN_TOOLS and not admin_ready:
                            report["results"][case_id] = {"tool": tool, "verdict": "blocked",
                                                         "reason": "ATLASSIAN_ADMIN_API_KEY is not configured"}
                            save()
                            continue
                        try:
                            case[2].format(**context)
                        except KeyError as missing:
                            report["results"][case_id] = {"tool": tool, "verdict": "skipped", "reason": f"needs {missing}"}
                            save()
                            print(f"SKIP {case_id:<34} needs {missing}", flush=True)
                            continue
                        row = {"tool": tool, "verdict": "blocked", "stage": "prepared",
                               "reason": "Case may be in progress; reconcile before any replacement request."}
                        report["results"][case_id] = row
                        save()  # Persist intent before new_chat/send can create application work.
                        try:
                            run_case(page, traffic, args, case, context, row=row, checkpoint=save)
                            if row.get("verdict") == "pass":
                                row.pop("reason", None)
                        except Exception as exc:
                            row.update(verdict="fail", reason=f"{type(exc).__name__}: {str(exc)[:300]}")
                            try:
                                page.screenshot(path=str(shots / f"{case_id}.png"))
                                row["screenshot"] = str(shots / f"{case_id}.png")
                            except Exception:
                                pass
                        save()
                        print(f"[{index:02d}/{len(pending)}] {row['verdict'].upper():<12} {case_id:<34} "
                              f"{row.get('seconds', '-'):>6}s ws={row.get('websocket', {}).get('frames_in', '-')} "
                              f"{row.get('reason', '')[:150]}", flush=True)
                finally:
                    browser.close()
        report["run_state"] = "finished"
    finally:
        if report["results"].get("jira-project-archive", {}).get("verdict") == "pass":
            report["cleanup_required"] = {"archived_project": f"MY{report['run'][-4:]}",
                                          "action": "Reconcile and clean up this disposable project explicitly."}
        save(final=True)
    acceptance = report["acceptance"]
    print("summary:", report["summary"], "| registered tools:", report["registered_tools"], "| report:", args.report)
    print(f"acceptance: {acceptance['overall']} ({acceptance['scope']} scope, "
          f"{acceptance['pass']}/{acceptance['selected_cases']} selected cases passed)")
    return acceptance["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
