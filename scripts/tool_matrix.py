#!/usr/bin/env python3
"""Live tool matrix: every registered tool through the real product path.

Browser API (Phoenix) -> durable job -> Hub routing -> specialist -> SemanticGuard
-> approval (mutations) -> ToolGateway/MCP -> configured provider -> history.

Each case is a natural request for one tool against a disposable lab fixture.
Mutations are approved through the same endpoint the UI uses, then the
structured provider result is verified. Results append to a JSON report so a
run can resume; secrets are never printed or stored.

Usage (AI repository root, AI service Python):
  python scripts/tool_matrix.py --report .runtime/acceptance/tool-matrix.json [--only id ...] [--resume]
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

from scripts.cross_repo_acceptance import (  # noqa: E402
    AcceptanceFailure,
    BrowserPathClient,
    _assert_history_roundtrip,
    _tool_arguments,
    _tool_name,
)

RUN = datetime.now(timezone.utc).strftime("%m%d%H%M")
PROJECT = os.environ.get("ACCEPTANCE_JIRA_PROJECT_KEY", "SUP")

# (id, tool, prompt, approve, expected_terminal_after)
# approve=None for reads; True to approve the proposal; "reject" to reject it.
CASES: list[tuple[str, str, str, object]] = [
    # --- workspace / developer (reads) ---
    ("workspace-list", "workspace_list", "List the files at the root of the mvp repository.", None),
    ("workspace-read", "workspace_read_text", "Show me the contents of calculator.py in the mvp repository.", None),
    ("workspace-search", "workspace_search", "Search the mvp repository for the text def add.", None),
    ("workspace-file-info", "workspace_file_info", "Show file information for README.md in the mvp repository.", None),
    ("project-info", "workspace_project_info", "What kind of project is the mvp repository and which developer operations does it support?", None),
    ("process-snapshot", "workspace_process_snapshot", "Show the current workspace process snapshot.", None),
    ("service-status", "workspace_service_status", "Show the Docker Compose service status for projectOps/.runtime/mvp-workspace.", None),
    ("service-logs", "workspace_service_logs", "Show recent logs of the git service in projectOps/.runtime/mvp-workspace.", None),
    ("process-exec", "process_exec", "Run pwd in the workspace.", "auto"),
    ("git-status", "workspace_git_status", "What's the current Git status of mvp?", None),
    ("git-branches", "workspace_git_branches", "Which local Git branches exist in mvp?", None),
    ("git-log", "workspace_git_log", "Show me recent mvp Git commits.", None),
    ("git-changed", "workspace_git_changed_files", "Just list the changed files in mvp.", None),
    ("git-diff", "workspace_git_diff", "Show mvp's unstaged Git patch.", None),
    ("git-staged-diff", "workspace_git_staged_diff", "What's currently staged in mvp? Show the Git patch.", None),
    # --- workspace / developer (mutations, isolated mvp repo) ---
    ("mkdir", "workspace_mkdir", f"Create a new directory named matrix_{RUN} in the developer workspace.", True),
    ("run-tests", "workspace_run_tests", "Run the tests of the mvp repository.", True),
    ("run-build", "workspace_run_build", "Run the build for the mvp repository.", True),
    # prepare_git_fixture() leaves one pending README.md change on main.
    ("git-stage-files", "workspace_git_stage_files", "Stage only README.md in the mvp repository.", True),
    ("git-unstage-files", "workspace_git_unstage_files", "Unstage only README.md in the mvp repository.", True),
    ("git-stage-all", "workspace_git_stage_all", "Stage all current changes in the mvp repository. Do not commit.", True),
    ("git-commit", "workspace_git_commit", f"Commit the staged changes in mvp with the message: matrix check {RUN}", True),
    ("git-push", "workspace_git_push", "Push the committed changes in the mvp repository to its configured upstream.", True),
    ("git-create-branch", "workspace_git_create_branch", f"Create a new Git branch named matrix-{RUN} in mvp.", True),
    ("git-switch-branch", "workspace_git_switch_branch", f"Switch the mvp repository to branch matrix-{RUN}.", True),
    # --- directory / access (Samba AD lab) ---
    ("account-status", "account_status", "Is jdoe locked or disabled?", None),
    ("check-access", "check_access", "Does jdoe have vpn access?", None),
    ("grant-access", "grant_access", "Grant jdoe admin access.", True),
    ("revoke-access", "revoke_access", "Revoke admin access from jdoe.", True),
    ("unlock-user", "unlock_user", "Unlock the account asmith.", True),
    ("disable-user", "disable_user", "Disable the account mflowseventeen.", True),
    ("enable-user", "enable_user", "Enable the account mflowseventeen.", True),
    ("reset-password", "reset_password", "Reset the password for mflowseventeen.", True),
    ("account-create", "account_create", f"Create a corporate account for given name Matrix, family name Run{RUN}, department IT and role Engineer.", True),
    # --- mailbox (Maddy lab) ---
    ("mailbox-create", "mailbox_create", f"Create a local mailbox for matrix.{RUN}@itsm.test.", True),
    ("mailbox-status", "mailbox_status", f"Check whether the local mailbox matrix.{RUN}@itsm.test exists.", None),
    # --- assets / knowledge ---
    ("asset-get", "asset_get", "Show the managed asset record MVP-LAB-LAPTOP-001. Do not change its assignment.", None),
    ("asset-search", "asset_search", "Find available laptop assets in the inventory.", None),
    ("asset-assign", "asset_assign", "Assign asset MVP-LAB-LAPTOP-001 to employee jdoe.", True),
    ("asset-unassign", "asset_unassign", "Remove the current owner assignment from asset MVP-LAB-LAPTOP-001.", True),
    ("knowledge-get", "knowledge_get", "Retrieve ITSM knowledge document KB-GIT-STATUS.", None),
    ("knowledge-search", "knowledge_search", "Search ITSM knowledge with the exact query Git staged and unstaged differences.", None),
    ("runbook-get", "runbook_get", "Read runbook RB-GIT-REVIEW; do not perform any runbook steps.", None),
    # --- networking (OpenWrt lab, read only) ---
    ("network-system", "network_system_info", "Read the OpenWrt router hardware identity and firmware release.", None),
    ("network-interface", "network_interface_status", "Read the current state of the OpenWrt lan interface.", None),
    ("network-firewall", "network_firewall_config", "Inspect the saved OpenWrt firewall configuration. Do not change anything.", None),
    # --- Jira projects ---
    ("jira-project-list", "jira_project_list", "List the Jira projects I can see.", None),
    ("jira-project-get", "jira_project_get", f"Get Jira project {PROJECT}.", None),
    ("jira-project-create", "jira_project_create", f"Create a Jira project with key MX{RUN[-4:]} named Matrix {RUN} using the software-kanban template.", True),
    ("jira-project-update", "jira_project_update", f"Rename Jira project MX{RUN[-4:]} to Matrix renamed {RUN}.", True),
    ("jira-project-delete", "jira_project_delete", f"Delete Jira project MX{RUN[-4:]}.", True),
    # Archived projects cannot be deleted, so archiving uses its own project.
    ("jira-project-create-archive-target", "jira_project_create", f"Create a Jira project with key MY{RUN[-4:]} named Matrix archive {RUN} using the software-kanban template.", True),
    ("jira-project-archive", "jira_project_archive", f"Archive Jira project MY{RUN[-4:]}.", True),
    # --- tickets (Jira) ---
    ("ticket-search", "ticket_search", f"Search Jira tickets in project {PROJECT}.", None),
    ("ticket-create", "ticket_create", f"Create a Jira ticket in project {PROJECT} with summary: Matrix check {RUN}.", True),
    ("ticket-get", "ticket_get", "Get Jira ticket {ticket}.", None),
    ("ticket-comment", "ticket_add_comment", "Add a comment to Jira ticket {ticket}: matrix comment " + RUN, True),
    ("ticket-comments", "ticket_comments", "What discussion is on Jira ticket {ticket}?", None),
    ("ticket-assign", "ticket_assign", "Assign Jira ticket {ticket} to {assignee}.", True),
    ("ticket-transition", "ticket_transition", "Move Jira ticket {ticket} to status {status}.", True),
    ("ticket-history", "ticket_history", "Show the history of Jira ticket {ticket}.", None),
    # --- Atlassian organization (requires admin key) ---
    ("atlassian-credentials", "atlassian_credential_status", "Check which Atlassian credentials are configured. Return configuration status only.", None),
    ("atlassian-orgs", "atlassian_org_list", "List the Atlassian organizations visible to the configured administrator.", None),
    ("atlassian-org", "atlassian_org_get", "Read Atlassian organization {org_id}.", None),
    ("atlassian-workspaces", "atlassian_workspace_list", "List existing Atlassian workspaces in organization {org_id}.", None),
    ("atlassian-token-metadata", "atlassian_api_token_metadata", "List API token metadata for Atlassian account {account_id}. Do not return token values.", None),
    ("atlassian-invite", "atlassian_user_invite", f"Invite matrix.{RUN}@itsm.test to the configured Jira cloud product.", True),
    # --- Jira depth ---
    # --- identity depth ---
    # --- operations ---
    # --- service desk ---
    # --- network clients ---
]

TERMINAL_OK = {"completed"}
# Organization-admin tools need ATLASSIAN_ADMIN_API_KEY (an org admin key).
ATLASSIAN_ADMIN_TOOLS = {"atlassian_org_list", "atlassian_org_get", "atlassian_workspace_list",
                         "atlassian_api_token_metadata", "atlassian_user_invite"}
GIT_MUTATION_CASES = {"git-stage-files", "git-unstage-files", "git-stage-all", "git-commit", "git-push"}
GIT_FIXTURE = Path(os.environ.get("ACCEPTANCE_GIT_FIXTURE",
                                  "/mnt/c/project/projectOps/.runtime/mvp-workspace/workspace"))


def prepare_git_fixture() -> str:
    """Leave the isolated mvp repository on main with one pending README change."""
    import subprocess

    switched = subprocess.run(["git", "-C", str(GIT_FIXTURE), "switch", "main"],
                              capture_output=True, text=True, timeout=60)
    if switched.returncode != 0:
        return "git switch main failed: " + switched.stderr.strip()[:200]
    with (GIT_FIXTURE / "README.md").open("a", encoding="utf-8") as handle:
        handle.write(f"\nMatrix run {RUN}.\n")
    return "ready"


def available_status(ticket_key: str) -> str | None:
    """A status the ticket can move to now (read-only fixture discovery)."""
    import httpx
    from config import get_settings

    settings = get_settings()
    token = settings.jira_api_token
    token = token.get_secret_value() if hasattr(token, "get_secret_value") else token
    with httpx.Client(base_url=str(settings.jira_base_url).rstrip("/"), auth=(settings.jira_email, token),
                      timeout=20, trust_env=False, headers={"Accept": "application/json"}) as client:
        response = client.get(f"/rest/api/3/issue/{ticket_key}/transitions")
    transitions = response.json().get("transitions", []) if response.status_code == 200 else []
    return transitions[0]["to"]["name"] if transitions else None


def jira_admin_client():
    import httpx
    from config import get_settings

    settings = get_settings()
    token = settings.jira_api_token
    token = token.get_secret_value() if hasattr(token, "get_secret_value") else token
    return httpx.Client(base_url=str(settings.jira_base_url).rstrip("/"), auth=(settings.jira_email, token),
                        timeout=30, trust_env=False, headers={"Accept": "application/json"})


def remove_jira_fixture(project_key: str) -> str:
    """Restore an archived matrix project and move it to Jira's trash (undoable for 60 days)."""
    with jira_admin_client() as client:
        found = client.get(f"/rest/api/3/project/{project_key}")
        archived = found.status_code == 200 and found.json().get("archived") is True
        if found.status_code == 404 or archived:
            restored = client.post(f"/rest/api/3/project/{project_key}/restore")
            if restored.status_code not in (200, 204):
                return f"restore HTTP {restored.status_code}"
        deleted = client.delete(f"/rest/api/3/project/{project_key}", params={"enableUndo": "true"})
        return "removed" if deleted.status_code in (200, 202, 204) else f"delete HTTP {deleted.status_code}"


def atlassian_admin_configured() -> bool:
    from config import get_settings

    return bool(get_settings().atlassian_admin_api_key)


def discover_fixtures(context: dict) -> None:
    """Fill prompt placeholders that need read-only lookups of disposable lab data."""
    # --- Jira depth fixtures ---
    # --- identity fixtures ---
    # --- service desk fixtures ---


def record_created(tool: str, tool_result: dict, context: dict) -> None:
    """Remember identifiers a passing case created, for later cases in the same run."""
    if tool == "ticket_create":
        key = (tool_result.get("ticket_key") or (tool_result.get("ticket") or {}).get("ticket_key")
               or (tool_result.get("result") or {}).get("ticket_key"))
        if key:
            context["ticket"] = key
            status = available_status(key)
            if status and "status" not in context:
                context["status"] = status
    # --- service desk results ---


def worker_evidence(job: dict) -> dict | None:
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    for worker in result.get("results", []) or []:
        if isinstance(worker, dict) and isinstance(worker.get("tool_result"), dict):
            return worker
    execution = result.get("approval_execution")
    if isinstance(execution, dict) and isinstance(execution.get("result"), dict):
        return {"tool_result": execution["result"], "proposed_tool": _tool_name(job)}
    return None


def summarize(job: dict) -> dict:
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    workers = result.get("results", []) or []
    return {
        "status": job.get("status"),
        "agent": job.get("selected_agent"),
        "tool": _tool_name(job),
        "arguments": _tool_arguments(job),
        "worker_status": [w.get("status") for w in workers if isinstance(w, dict)],
        "outcome_codes": [w.get("outcome_code") for w in workers if isinstance(w, dict)],
        "error": (job.get("error") or "")[:240] or None,
        "answer": (result.get("answer") or "")[:240] or None,
    }


# Verdicts. plan_limited: the request was routed, approved and executed, but the
# provider account's plan (for example free Atlassian) does not include the API.
VERDICTS = ("pass", "fail", "plan_limited", "blocked", "skipped")
# A resumed run does not repeat cases whose verdict cannot change by retrying.
FINAL_VERDICTS = {"pass", "plan_limited"}


def plan_limited(worker: dict | None) -> str | None:
    """Provider reason when the account's plan lacks this API (not a product failure)."""
    tool_result = (worker or {}).get("tool_result") or {}
    if tool_result.get("error_code") != "plan_required":
        return None
    return str(tool_result.get("error") or "provider plan does not include this API")[:200]


def provider_ok(worker: dict | None) -> tuple[bool, str]:
    if not worker:
        return False, "no structured tool result"
    tool_result = worker["tool_result"]
    # Directory reads report ok without a status field; every other tool sets status.
    ok = tool_result.get("ok") is True and tool_result.get("status") in {None, "success", "executed"}
    detail = f"ok={tool_result.get('ok')} status={tool_result.get('status')} provider={tool_result.get('provider')}"
    if not ok and tool_result.get("error"):
        detail += f" error={str(tool_result.get('error'))[:160]}"
    return ok, detail


def main() -> int:
    parser = argparse.ArgumentParser(description="Live tool matrix through the product path.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--backend-url", default="http://127.0.0.1:4000")
    parser.add_argument("--login-env", type=Path,
                        default=Path("/mnt/c/project/agenticBackend/itsm_backend/.runtime/acceptance/login.env"))
    args = parser.parse_args()

    report = {"schema": "live-tool-matrix.v1", "results": {}}
    if args.report.exists():
        report = json.loads(args.report.read_text(encoding="utf-8"))
    report["run"] = RUN
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        tmp = args.report.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(args.report)

    login = dotenv_values(args.login_env)
    client = BrowserPathClient(args.backend_url, timeout_seconds=args.timeout)
    client.preflight()
    client.login(login["ACCEPTANCE_EMAIL"], login["ACCEPTANCE_PASSWORD"])
    context: dict = {}
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
    try:
        for identity, tool, prompt, approve in selected:
            if tool in ATLASSIAN_ADMIN_TOOLS and not admin_ready:
                report["results"][identity] = {"tool": tool, "verdict": "blocked",
                                               "reason": "ATLASSIAN_ADMIN_API_KEY is not configured"}
                save()
                print(f"BLOCK {identity:<25} needs ATLASSIAN_ADMIN_API_KEY", flush=True)
                continue
            prior = report["results"].get(identity)
            if args.resume and prior and prior.get("verdict") in FINAL_VERDICTS:
                continue
            try:
                message = prompt.format(**context)
            except KeyError as missing:
                report["results"][identity] = {"tool": tool, "verdict": "skipped",
                                               "reason": f"needs context {missing}"}
                save()
                print(f"SKIP {identity:<26} needs {missing}", flush=True)
                continue
            started = time.monotonic()
            row: dict = {"tool": tool, "prompt": message, "approve": approve}
            try:
                chat = client.create_chat(f"Tool matrix {RUN}: {identity}")
                job = client.create_job(chat_id=chat, message=message)
                terminal = client.wait_for_terminal(job["job_id"], poll_seconds=2)
                row["job_id"] = job["job_id"]
                row["first"] = summarize(terminal)
                if approve == "auto":
                    # Policy decides whether this request needs approval.
                    approve = True if terminal.get("status") == "waiting_approval" else None
                if approve is not None:
                    if terminal.get("status") != "waiting_approval":
                        raise AcceptanceFailure(f"expected an approval proposal, got {terminal.get('status')}")
                    if _tool_name(terminal) != tool:
                        raise AcceptanceFailure(f"proposed {_tool_name(terminal)} instead of {tool}")
                    if approve == "reject":
                        terminal = client.reject_job(job["job_id"])
                    else:
                        client.approve_job(job["job_id"])
                    terminal = client.wait_for_terminal(job["job_id"], poll_seconds=2)
                    row["after_decision"] = summarize(terminal)
                _assert_history_roundtrip(client.get_history(chat), terminal, chat_id=chat)
                worker = worker_evidence(terminal)
                actual_tool = (worker or {}).get("proposed_tool") or _tool_name(terminal)
                ok, detail = provider_ok(worker)
                row["provider"] = detail
                if actual_tool != tool:
                    raise AcceptanceFailure(f"used {actual_tool} instead of {tool}")
                limited = plan_limited(worker)
                if limited:
                    row["verdict"] = "plan_limited"
                    row["reason"] = limited
                elif terminal.get("status") not in TERMINAL_OK or not ok:
                    raise AcceptanceFailure(f"status={terminal.get('status')} {detail}")
                else:
                    row["verdict"] = "pass"
                if row["verdict"] == "pass":
                    record_created(tool, worker["tool_result"], context)
            except Exception as exc:  # record and continue with the next tool
                row["verdict"] = "fail"
                row["reason"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            row["seconds"] = round(time.monotonic() - started, 1)
            report["results"][identity] = row
            report["context"] = context
            save()
            first = row.get("first", {})
            print(f"{row['verdict'].upper():<4} {identity:<26} {row['seconds']:>6}s tool={first.get('tool')} "
                  f"status={(row.get('after_decision') or first).get('status')} {row.get('reason', '')[:160]}",
                  flush=True)
    finally:
        client.close()
        archived = report["results"].get("jira-project-archive", {})
        if archived.get("verdict") == "pass" and archived.get("prompt", "").endswith(f"MY{RUN[-4:]}."):
            print("fixture cleanup:", remove_jira_fixture(f"MY{RUN[-4:]}"), flush=True)
        results = report["results"].values()
        report["summary"] = {verdict: sum(1 for r in results if r.get("verdict") == verdict)
                             for verdict in VERDICTS}
        save()
        print("summary:", report["summary"])
    return 0 if report["summary"].get("fail", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
