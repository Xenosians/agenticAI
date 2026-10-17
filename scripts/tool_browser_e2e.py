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
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import dotenv_values  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

from scripts.cross_repo_acceptance import _assert_history_roundtrip, _tool_name  # noqa: E402
from scripts.tool_matrix import (  # noqa: E402
    ATLASSIAN_ADMIN_TOOLS,
    CASES,
    FINAL_VERDICTS,
    GIT_MUTATION_CASES,
    RUN,
    TERMINAL_OK,
    VERDICTS,
    atlassian_admin_configured,
    discover_fixtures,
    plan_limited,
    prepare_git_fixture,
    provider_ok,
    record_created,
    remove_jira_fixture,
    worker_evidence,
)

BACKEND_REPO = Path(os.getenv("ACCEPTANCE_BACKEND_REPO", str(PROJECT_ROOT.parents[1] / "agenticBackend" / "itsm_backend")))
REPORT = PROJECT_ROOT / ".runtime" / "acceptance" / "tool-browser-e2e.json"
SHOTS = PROJECT_ROOT / ".runtime" / "acceptance" / "tool-browser-e2e"
TERMINAL = {"completed", "failed", "denied", "outcome_unknown", "reconciliation_required"}


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
                    approval_other_than is None or proposal.get("approval_id") != approval_other_than):
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


def sign_in(page, frontend_url: str, email: str, password: str, timeout_ms: int) -> None:
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


def new_chat(page, timeout_ms: int) -> None:
    page.get_by_title("More").click()
    page.get_by_role("button", name="New chat", exact=True).click()
    page.locator("#messageInput").wait_for(state="visible", timeout=timeout_ms)
    page.wait_for_timeout(300)


def send(page, prompt: str, timeout_ms: int) -> tuple[str, str]:
    page.locator("#messageInput").fill(prompt)
    button = page.get_by_role("button", name="Send message")
    deadline = time.monotonic() + timeout_ms / 1000
    while not button.is_enabled():
        if time.monotonic() > deadline:
            raise TimeoutError("Send button never became enabled")
        page.wait_for_timeout(100)
    with page.expect_response(lambda r: r.url.rstrip("/").endswith("/api/v1/jobs") and r.request.method == "POST",
                              timeout=timeout_ms) as info:
        button.click()
    response = info.value
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


def run_case(page, traffic: BrowserTraffic, args, case: tuple, context: dict) -> dict:
    identity, tool, prompt, approve_mode = case
    message = prompt.format(**context)
    row: dict = {"tool": tool, "prompt": message}
    started = time.monotonic()
    timeout_ms = int(args.timeout * 1000)

    new_chat(page, timeout_ms)
    sockets_before = len(traffic.sockets)
    job_id, chat_id = send(page, message, timeout_ms)
    row["job_id"] = job_id
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
        approve(page, timeout_ms)
        # Done when the job is terminal, or waiting on a *different* approval
        # (a follow-up step); the approval just clicked no longer counts.
        final = traffic.wait_after_approval(page, args.backend_url, job_id, approval_id, args.timeout)
    row["status"] = final.get("status")

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

    opened = traffic.sockets[sockets_before:]
    row["websocket"] = {"opened": len(opened), "frames_in": sum(s["frames_in"] for s in opened)}
    if not limited:
        record_created(tool, worker["tool_result"], context)
    row["seconds"] = round(time.monotonic() - started, 1)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--resume", action="store_true",
                        help="skip cases whose last verdict is final (pass or plan_limited)")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--frontend-url", default="http://127.0.0.1:8080")
    parser.add_argument("--backend-url", default="http://127.0.0.1:4000")
    parser.add_argument("--login-env", type=Path, default=BACKEND_REPO / ".runtime" / "acceptance" / "login.env")
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()

    # Results merge by case id, so a partial run (--only) never discards other cases.
    report = {"schema": "tool-browser-e2e.v1", "results": {}}
    if REPORT.exists():
        report = json.loads(REPORT.read_text(encoding="utf-8"))
    report["run"] = RUN
    report["started_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    SHOTS.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        results = report["results"].values()
        report["summary"] = {verdict: sum(1 for r in results if r.get("verdict") == verdict)
                             for verdict in VERDICTS}
        report["registered_tools"] = len({case[1] for case in CASES})
        temporary = REPORT.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(REPORT)

    login = dotenv_values(args.login_env)
    # Only a resumed run reuses fixtures (ticket key, status) from the previous run.
    context: dict = dict(report.get("context") or {}) if args.resume else {}
    for key, value in (("org_id", os.environ.get("ATLASSIAN_ORG_ID")),
                       ("account_id", os.environ.get("ATLASSIAN_ACCOUNT_ID")),
                       ("assignee", os.environ.get("ACCEPTANCE_JIRA_ASSIGNEE")),
                       ("status", os.environ.get("ACCEPTANCE_JIRA_STATUS"))):
        if value and key not in context:
            context[key] = value
    # Notifications go to the signed-in lab user, a registered application user.
    context.setdefault("user_email", login["ACCEPTANCE_EMAIL"])
    discover_fixtures(context)
    selected = [case for case in CASES if not args.only or case[0] in args.only]
    if any(case[0] in GIT_MUTATION_CASES for case in selected):
        print("git fixture:", prepare_git_fixture(), flush=True)
    admin_ready = atlassian_admin_configured()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not args.headed)
        page = browser.new_context(viewport={"width": 1440, "height": 900}).new_page()
        traffic = BrowserTraffic(page)
        try:
            sign_in(page, args.frontend_url, login["ACCEPTANCE_EMAIL"], login["ACCEPTANCE_PASSWORD"],
                    int(args.timeout * 1000))
            for index, case in enumerate(selected, 1):
                identity, tool = case[0], case[1]
                prior = report["results"].get(identity)
                if args.resume and prior and prior.get("verdict") in FINAL_VERDICTS:
                    continue
                if tool in ATLASSIAN_ADMIN_TOOLS and not admin_ready:
                    report["results"][identity] = {"tool": tool, "verdict": "blocked",
                                                   "reason": "ATLASSIAN_ADMIN_API_KEY is not configured"}
                    save()
                    continue
                try:
                    case[2].format(**context)
                except KeyError as missing:
                    report["results"][identity] = {"tool": tool, "verdict": "skipped", "reason": f"needs {missing}"}
                    save()
                    print(f"SKIP {identity:<34} needs {missing}", flush=True)
                    continue
                try:
                    row = run_case(page, traffic, args, case, context)
                    row.setdefault("verdict", "pass")
                except Exception as exc:
                    row = {"tool": tool, "verdict": "fail", "reason": f"{type(exc).__name__}: {str(exc)[:300]}"}
                    try:
                        page.screenshot(path=str(SHOTS / f"{identity}.png"))
                        row["screenshot"] = str(SHOTS / f"{identity}.png")
                    except Exception:
                        pass
                report["results"][identity] = row
                report["context"] = context
                save()
                print(f"[{index:02d}/{len(selected)}] {row['verdict'].upper():<12} {identity:<34} "
                      f"{row.get('seconds', '-'):>6}s ws={row.get('websocket', {}).get('opened', '-')} "
                      f"{row.get('reason', '')[:150]}", flush=True)
        finally:
            browser.close()
            archived = report["results"].get("jira-project-archive", {})
            if archived.get("verdict") == "pass" and archived.get("prompt", "").endswith(f"MY{RUN[-4:]}."):
                print("fixture cleanup:", remove_jira_fixture(f"MY{RUN[-4:]}"), flush=True)
            save()
    print("summary:", report["summary"], "| registered tools:", report["registered_tools"], "| report:", REPORT)
    return 0 if report["summary"].get("fail", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
