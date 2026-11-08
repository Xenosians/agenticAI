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

Add --adapter local, or --adapter http --model-id <connection>, to run every case
on that inference profile (SRS 1.9 V19-02). The account's own selection is
restored afterwards, and a case passes only if the job's recorded steps ran there.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import dotenv_values  # noqa: E402

from scripts.acceptance_profiles import ADAPTERS, AcceptanceProfileScope  # noqa: E402
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
def build_cases(run: str, project: str) -> list[tuple[str, str, str, object]]:
    """Keep fixture identities stable when resuming the same evidence run."""
    return [
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
        ("git-commit-info", "workspace_git_commit_info", "What did commit ad203b7 change in the mvp repository?", None),
        ("git-changed", "workspace_git_changed_files", "Just list the changed files in mvp.", None),
        ("git-diff", "workspace_git_diff", "Show mvp's unstaged Git patch.", None),
        ("git-staged-diff", "workspace_git_staged_diff", "What's currently staged in mvp? Show the Git patch.", None),
        # --- workspace / developer (mutations, isolated mvp repo) ---
        ("mkdir", "workspace_mkdir", f"Create a new directory named matrix_{run} in the developer workspace.", True),
        ("run-tests", "workspace_run_tests", "Run the tests of the mvp repository.", True),
        ("run-build", "workspace_run_build", "Run the build for the mvp repository.", True),
        # prepare_git_fixture() leaves one pending README.md change on main.
        ("git-stage-files", "workspace_git_stage_files", "Stage only README.md in the mvp repository.", True),
        ("git-unstage-files", "workspace_git_unstage_files", "Unstage only README.md in the mvp repository.", True),
        ("git-stage-all", "workspace_git_stage_all", "Stage all current changes in the mvp repository. Do not commit.", True),
        ("git-commit", "workspace_git_commit", f"Commit the staged changes in mvp with the message: matrix check {run}", True),
        ("git-push", "workspace_git_push", "Push the committed changes in the mvp repository to its configured upstream.", True),
        ("git-create-branch", "workspace_git_create_branch", f"Create a new Git branch named matrix-{run} in mvp.", True),
        ("git-switch-branch", "workspace_git_switch_branch", f"Switch the mvp repository to branch matrix-{run}.", True),
        # --- directory / access (Samba AD lab) ---
        ("account-status", "account_status", "Is jdoe locked or disabled?", None),
        ("check-access", "check_access", "Does jdoe have vpn access?", None),
        ("grant-access", "grant_access", "Grant jdoe admin access.", True),
        ("revoke-access", "revoke_access", "Revoke admin access from jdoe.", True),
        ("unlock-user", "unlock_user", "Unlock the account asmith.", True),
        ("disable-user", "disable_user", "Disable the account mflowseventeen.", True),
        ("enable-user", "enable_user", "Enable the account mflowseventeen.", True),
        ("reset-password", "reset_password", "Reset the password for mflowseventeen.", True),
        ("account-create", "account_create", f"Create a corporate account for given name Matrix, family name Run{run}, department IT and role Engineer.", True),
        # --- mailbox (Maddy lab) ---
        ("mailbox-create", "mailbox_create", f"Create a local mailbox for matrix.{run}@itsm.test.", True),
        ("mailbox-status", "mailbox_status", f"Check whether the local mailbox matrix.{run}@itsm.test exists.", None),
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
        ("jira-project-get", "jira_project_get", f"Get Jira project {project}.", None),
        ("jira-project-create", "jira_project_create", f"Create a Jira project with key MX{run[-4:]} named Matrix {run} using the software-kanban template.", True),
        ("jira-project-update", "jira_project_update", f"Rename Jira project MX{run[-4:]} to Matrix renamed {run}.", True),
        ("jira-project-delete", "jira_project_delete", f"Delete Jira project MX{run[-4:]}.", True),
        # Archived projects cannot be deleted, so archiving uses its own project.
        ("jira-project-create-archive-target", "jira_project_create", f"Create a Jira project with key MY{run[-4:]} named Matrix archive {run} using the software-kanban template.", True),
        ("jira-project-archive", "jira_project_archive", f"Archive Jira project MY{run[-4:]}.", True),
        # --- tickets (Jira) ---
        ("ticket-search", "ticket_search", f"Search Jira tickets in project {project}.", None),
        ("ticket-create", "ticket_create", f"Create a Jira ticket in project {project} with summary: Matrix check {run}.", True),
        ("ticket-get", "ticket_get", "Get Jira ticket {ticket}.", None),
        ("ticket-comment", "ticket_add_comment", "Add a comment to Jira ticket {ticket}: matrix comment " + run, True),
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
        ("atlassian-invite", "atlassian_user_invite", f"Invite matrix.{run}@itsm.test to the configured Jira cloud product.", True),
        # --- Jira depth ---
        ("ticket-update", "ticket_update", "Set the priority of Jira ticket {ticket} to {ticket_priority} and add the label matrix-check.", True),
        ("ticket-link", "ticket_link", "Link Jira ticket {ticket} to {other_ticket}: {ticket} relates to {other_ticket}.", True),
        ("ticket-links", "ticket_links", "Which issues are linked to Jira ticket {ticket}?", None),
        ("ticket-assigned", "ticket_assigned", "Which Jira tickets are assigned to {assignee}?", None),
        ("ticket-worklog", "ticket_worklog_add", "Log 15m of work on Jira ticket {ticket} with the comment: matrix worklog.", True),
        # --- identity depth ---
        ("account-search", "account_search", "Find directory accounts matching jdoe.", None),
        ("user-groups", "user_groups", "Which groups is jdoe a member of?", None),
        ("group-members", "group_members", "Who is in the vpn access group?", None),
        ("access-explain", "access_explain", "jdoe cannot connect to the VPN. Find out why without changing anything.", None),
        # --- operations ---
        ("ops-container-status", "container_status", "Show the status of the platform containers.", None),
        ("ops-container-logs", "container_logs", "Show the last 20 log lines of the itsm-surrealdb container.", None),
        ("ops-service-health", "service_health", "Check the health of the platform services.", None),
        ("ops-notify-user", "notify_user", "Send a notification to {user_email} with the message: matrix notification check.", True),
        # --- service desk ---
        ("service-request", "service_request_create",
         "Raise a {request_type} request in the {service_desk} service desk with the summary: Matrix request " + run + ".", True),
        ("sla-status", "sla_status", "What is the SLA status of request {request}?", None),
        ("queue-list", "queue_list", "List the queues of the {service_desk} service desk with their issue counts.", None),
        # --- network clients ---
        ("network-dhcp-leases", "network_dhcp_leases", "Show the current DHCP leases on the OpenWrt router.", None),
        ("network-wireless-clients", "network_wireless_clients", "Which wireless clients are connected to the OpenWrt router?", None),
        ("network-router-health", "network_router_health", "How long has the OpenWrt router been up, and how busy is it?", None),
        ("network-interfaces", "network_interfaces", "List every interface on the OpenWrt router with its addresses.", None),
        ("network-known-hosts", "network_known_hosts", "Which hosts does the OpenWrt router know about?", None),
        ("network-dns-config", "network_dns_config", "Which DNS servers does the OpenWrt router use, and is rebind protection on?", None),
        ("network-tailnet-devices", "network_tailnet_devices", "Which devices are on our Tailscale network, and are they online?", None),
        ("network-tailnet-dns", "network_tailnet_dns", "Do our Tailscale devices resolve names over encrypted DNS?", None),
        ("network-dns-integrity", "network_dns_integrity_check", "Is our ISP rewriting DNS answers? Check vimeo.com and example.com.", None),
        # --- Proxmox VE (read only; not_configured until a host is set up) ---
        ("proxmox-cluster", "proxmox_cluster_status", "What is the Proxmox VE cluster status?", None),
        ("proxmox-nodes", "proxmox_nodes", "List the Proxmox VE nodes and their status.", None),
        ("proxmox-guests", "proxmox_guests", "Which virtual machines and containers run on Proxmox VE?", None),
        ("proxmox-storage", "proxmox_node_storage", "Show the storage pools of Proxmox node {proxmox_node}.", None),
        ("proxmox-tasks", "proxmox_recent_tasks", "Show recent tasks on Proxmox node {proxmox_node}.", None),
    ]


CASES = build_cases(RUN, PROJECT)

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


def atlassian_admin_configured() -> bool:
    from config import get_settings

    return bool(get_settings().atlassian_admin_api_key)


def discover_fixtures(context: dict) -> None:
    """Fill prompt placeholders that need read-only lookups of disposable lab data."""
    # --- Jira depth fixtures ---
    if "other_ticket" not in context or "ticket_priority" not in context:
        with jira_admin_client() as client:
            found = client.post("/rest/api/3/search/jql", json={
                "jql": f'project = "{PROJECT}" ORDER BY created DESC', "maxResults": 1, "fields": ["priority"]})
            issues = found.json().get("issues", []) if found.status_code == 200 else []
            if issues:
                context.setdefault("other_ticket", issues[0]["key"])
                # A priority this project's edit screen allows, other than the
                # newest issue's (usually the default), so the update changes it.
                meta = client.get(f"/rest/api/3/issue/{issues[0]['key']}/editmeta")
                fields = meta.json().get("fields", {}) if meta.status_code == 200 else {}
                current = ((issues[0].get("fields") or {}).get("priority") or {}).get("name")
                names = [value.get("name") for value in (fields.get("priority") or {}).get("allowedValues", [])
                         if value.get("name") and value.get("name") != current]
                if names:
                    context.setdefault("ticket_priority", names[0])
    # --- identity fixtures ---
    # --- service desk fixtures ---
    if "service_desk" not in context or "request_type" not in context:
        with jira_admin_client() as client:
            desks = client.get("/rest/servicedeskapi/servicedesk")
            values = desks.json().get("values", []) if desks.status_code == 200 else []
            desk = next((item for item in values if item.get("projectKey") == PROJECT), None)
            if desk:
                types = client.get(f"/rest/servicedeskapi/servicedesk/{desk['id']}/requesttype")
                for request_type in types.json().get("values", []) if types.status_code == 200 else []:
                    fields = client.get(f"/rest/servicedeskapi/servicedesk/{desk['id']}/requesttype/{request_type['id']}/field")
                    listed = fields.json().get("requestTypeFields", []) if fields.status_code == 200 else []
                    required = {field.get("fieldId") for field in listed if field.get("required")}
                    # A request type this run can fill with a summary alone.
                    if listed and required <= {"summary"}:
                        context.setdefault("service_desk", PROJECT)
                        context.setdefault("request_type", request_type["name"])
                        break


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
    if tool == "service_request_create" and tool_result.get("ticket_key"):
        context["request"] = tool_result["ticket_key"]


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


# Steps the host always runs on its own local model, whatever the selected profile.
HOST_STAGES = {"CONTROL_ROUTING", "CANDIDATE_NARROWING"}
NETWORKING_STAGES = {"NETWORKING", "NETWORKING_SELECTED"}
LOCAL_ADAPTERS = {"local", "hf-causal"}


def _profile_step(metric) -> str | None:
    """The stage of a successful step the chosen profile runs, or None for any other record."""
    if not isinstance(metric, dict) or metric.get("operation") != "generate" or metric.get("outcome") != "success":
        return None
    stage = metric.get("repair_of") if metric.get("stage") == "REPAIR" else metric.get("stage")
    return None if stage in HOST_STAGES else stage


def inference_evidence(metrics, profile: dict) -> dict:
    """Prove that every successful step of a job ran where the chosen profile says.

    Host routing always runs locally; the networking specialist follows its own
    assignment; a repair counts as the step it repairs. A read that routing fully
    determines has no specialist step, so the answer step is evidence too. Failed
    steps and warm-ups prove nothing.
    """
    adapters, stages = set(), set()
    for metric in metrics if isinstance(metrics, list) else []:
        if not isinstance(metric, dict) or metric.get("operation") != "generate" or metric.get("outcome") != "success":
            continue
        stage = metric.get("repair_of") if metric.get("stage") == "REPAIR" else metric.get("stage")
        networking = stage in NETWORKING_STAGES or metric.get("role") == "networking-specialist"
        expected = ("local" if stage in HOST_STAGES
                    else profile["networking"]["adapter"] if networking else profile["adapter"])
        adapter = metric.get("adapter")
        if adapter not in (LOCAL_ADAPTERS if expected == "local" else {expected}):
            raise AcceptanceFailure(f"{stage} ran on {adapter}, not {expected}")
        stages.add(stage)
        if stage not in HOST_STAGES:
            adapters.add(adapter)
    if not adapters:
        raise AcceptanceFailure("no step ran on the chosen profile for this job")
    return {"adapters": sorted(adapters), "stages": sorted(stages)}


def job_inference_metrics(client, job_id: str, attempt, *, tries: int = 6, pause_seconds: float = 0.5) -> list:
    """The job attempt's recorded inference steps; the AI reports them just after the job ends."""
    metrics: list = []
    for remaining in range(tries, 0, -1):
        response = client.client.get("/api/v1/operations", params={"limit": 100})
        response.raise_for_status()
        row = next((item for item in response.json().get("jobs", [])
                    if isinstance(item, dict) and item.get("job_id") == job_id), None)
        if row is not None:
            if row.get("attempt") != attempt:
                raise AcceptanceFailure("the recorded inference steps belong to another attempt")
            recorded = row.get("metrics")
            metrics = recorded if isinstance(recorded, list) else []
            if any(_profile_step(metric) for metric in metrics):
                return metrics
        if remaining > 1:
            time.sleep(pause_seconds)
    return metrics


def profile_request(args) -> dict | None:
    """The requested inference profile, refused before any source or credential access."""
    if args.adapter is None:
        if args.model_id is not None or args.networking is not None:
            raise ValueError("inference profile: --model-id and --networking need --adapter")
        return None
    if args.adapter == "local" and args.model_id is not None:
        raise ValueError("inference profile: the local profile takes no --model-id")
    if args.adapter != "local" and not args.model_id:
        raise ValueError(f"inference profile: --adapter {args.adapter} needs --model-id")
    return {"adapter": args.adapter, "model_id": args.model_id, "networking": args.networking or "preserve"}


# Verdicts. plan_limited: the request was routed, approved and executed, but the
# provider account's plan (for example free Atlassian) does not include the API.
# not_configured: routed and executed, but this host has not set the provider up
# (for example no Proxmox VE host yet); the tool said so instead of guessing.
VERDICTS = ("pass", "fail", "plan_limited", "not_configured", "blocked", "skipped")
REPORT_SCHEMA = "live-tool-matrix.v2"


def select_matrix_cases(cases, requested):
    known = {case[0] for case in cases}
    unknown = set(requested) - known
    if unknown:
        raise ValueError("Unknown case identities: " + ", ".join(sorted(unknown)))
    selected = [case for case in cases if not requested or case[0] in requested]
    if not selected:
        raise ValueError("Acceptance needs at least one selected case")
    return selected


def source_identity():
    """Bind resumed evidence to source content, including uncommitted work.

    Only source extensions are hashed; no .env, credentials or runtime records
    are collected. Git metadata alone would miss the paused, dirty AI worktree.
    """
    roots = {"ai": PROJECT_ROOT,
             "backend": PROJECT_ROOT.parents[1] / "agenticBackend" / "itsm_backend",
             "frontend": PROJECT_ROOT.parents[1] / "agenticFrontend"}
    identities = {}
    source_extensions = {".py", ".ex", ".exs", ".nim", ".js", ".mjs", ".toml", ".md", ".json", ".yaml", ".yml"}
    for name, root in roots.items():
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        files = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root,
        ).decode().split("\0")
        digest = hashlib.sha256()
        for filename in sorted(set(files)):
            path = root / filename
            if path.suffix not in source_extensions or not path.is_file() or path.is_symlink():
                continue
            digest.update(filename.encode())
            digest.update(b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        identities[name] = {"head": head, "source_sha256": digest.hexdigest()}
    return identities


def load_matrix_report(path, *, resume, identity, run, project,
                       schema=REPORT_SCHEMA, execution_context=None):
    if not resume:
        if path.exists():
            raise ValueError("Report already exists; select a new report or explicitly resume it")
        report = {"schema": schema, "run": run, "project": project,
                  "source_identity": identity, "results": {}}
        if execution_context is not None:
            report["execution_context"] = execution_context
        return report
    if not path.exists():
        raise ValueError("Cannot resume a missing report")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(report, dict) or report.get("schema") != schema
            or report.get("source_identity") != identity
            or report.get("project") != project
            or report.get("execution_context") != execution_context):
        raise ValueError("Resume requires matching source/project/execution context evidence; use a new report after changes")
    if not isinstance(report.get("run"), str) or not re.fullmatch(r"[0-9]{8}", report["run"]):
        raise ValueError("Invalid saved fixture identity")
    if not isinstance(report.get("results"), dict):
        raise ValueError("Invalid saved results")
    return report


def bind_principal(report, email):
    if not isinstance(email, str) or not email.strip():
        raise ValueError("Acceptance needs an explicit sign-in identity")
    principal = hashlib.sha256(email.strip().casefold().encode()).hexdigest()
    previous = report.get("principal_sha256")
    if previous is not None and previous != principal:
        raise ValueError("Acceptance account changed; use a new report")
    report["principal_sha256"] = principal


def acceptance_summary(results, selected, *, registered_tools, full_scope, source_unchanged=True):
    """Zero is reserved for genuinely passing, nonempty declared coverage."""
    rows = [results.get(case[0], {}) for case in selected]
    counts = {verdict: sum(row.get("verdict") == verdict for row in rows) for verdict in VERDICTS}
    invalid = [case[0] for case, row in zip(selected, rows)
               if row.get("verdict") not in VERDICTS or row.get("tool") != case[1]]
    missing_tools = sorted(set(registered_tools) - {case[1] for case in selected})
    passed = bool(selected) and not invalid and counts["pass"] == len(selected)
    complete = passed and source_unchanged and (not full_scope or not missing_tools)
    exit_code = 1 if counts["fail"] else (0 if complete else 2)
    return {**counts, "selected_cases": len(selected), "invalid_or_missing_cases": invalid,
            "tools_outside_selected_coverage": missing_tools,
            "scope": "full" if full_scope else "selected", "selected_scope_passed": passed,
            "all_registered_tools_passed": passed and not missing_tools and source_unchanged,
            "source_unchanged": source_unchanged,
            "overall": "PASS" if complete else ("FAIL" if counts["fail"] else "INCOMPLETE"),
            "exit_code": exit_code}


def unset_provider_families(rows) -> dict[str, list[str]]:
    """Provider families this host has not set up, with the setting names each still needs."""
    return {row["id"]: list(row.get("missing") or []) for row in rows if row.get("state") == "not_configured"}


def host_unset_provider_families() -> dict[str, list[str]]:
    from config import get_settings
    from services.integration_readiness import integration_readiness

    return unset_provider_families(integration_readiness(get_settings()))


def provider_not_configured(tool: str, job: dict, unset: dict[str, list[str]]) -> str | None:
    """The provider's own setup message when this host has not set it up (not a product failure).

    Phoenix keeps only the error text of a failed job, so the verdict needs
    both facts: integration readiness says the tool's family is unset, and the
    job failed with that provider's own "not set up" message.
    """
    from services.integration_readiness import provider_for_capability

    if provider_for_capability(tool) not in unset or job.get("status") != "failed":
        return None
    error = job.get("error")
    if isinstance(error, str) and "is not set up on this Janus host." in error:
        return error[:200]
    return None


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
    global RUN
    parser = argparse.ArgumentParser(description="Live tool matrix through the product path.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--backend-url", default="http://127.0.0.1:4000")
    parser.add_argument("--login-env", type=Path,
                        default=Path("/mnt/c/project/agenticBackend/itsm_backend/.runtime/acceptance/login.env"))
    parser.add_argument("--adapter", choices=sorted(ADAPTERS),
                        help="Run every case on this inference profile; the account's selection is restored")
    parser.add_argument("--model-id", help="The owner's ready connection for an outbound --adapter")
    parser.add_argument("--networking", choices=("preserve", "local", "follow"),
                        help="Networking specialist: keep its assignment, use the local model, or follow --model-id")
    args = parser.parse_args()

    try:
        if args.only == []:
            raise ValueError("--only needs at least one case identity")
        select_matrix_cases(build_cases(RUN, PROJECT), args.only or [])
        requested_profile = profile_request(args)
    except ValueError as exc:
        parser.error(str(exc))

    identity = source_identity()
    execution_context = {"backend_url": args.backend_url.rstrip("/"), "login_env": str(args.login_env.resolve())}
    if requested_profile is not None:
        execution_context["inference_profile"] = requested_profile
    report = load_matrix_report(args.report, resume=args.resume, identity=identity, run=RUN, project=PROJECT,
                                execution_context=execution_context)
    RUN = report["run"]
    selected = select_matrix_cases(build_cases(RUN, PROJECT), args.only or [])
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        tmp = args.report.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(args.report)

    login = dotenv_values(args.login_env)
    bind_principal(report, login["ACCEPTANCE_EMAIL"])
    client = BrowserPathClient(args.backend_url, timeout_seconds=args.timeout)
    client.preflight()
    client.login(login["ACCEPTANCE_EMAIL"], login["ACCEPTANCE_PASSWORD"])
    context: dict = dict(report.get("context", {}))
    for key, value in (("org_id", os.environ.get("ATLASSIAN_ORG_ID")),
                       ("account_id", os.environ.get("ATLASSIAN_ACCOUNT_ID")),
                       ("assignee", os.environ.get("ACCEPTANCE_JIRA_ASSIGNEE")),
                       ("status", os.environ.get("ACCEPTANCE_JIRA_STATUS")),
                       ("proxmox_node", os.environ.get("ACCEPTANCE_PROXMOX_NODE"))):
        if value and key not in context:
            context[key] = value
    # Notifications go to the signed-in lab user, a registered application user.
    context.setdefault("user_email", login["ACCEPTANCE_EMAIL"])
    discover_fixtures(context)
    pending_git = [case for case in selected if case[0] in GIT_MUTATION_CASES
                   and case[0] not in report["results"]]
    if pending_git:
        attempted_git = any(case_id in GIT_MUTATION_CASES for case_id in report["results"])
        preparation = report.get("git_fixture_preparation", {})
        if not attempted_git and not preparation:
            report["git_fixture_preparation"] = {"status": "outcome_unknown"}
            save()
            preparation = {"status": prepare_git_fixture()}
            report["git_fixture_preparation"] = preparation
            save()
        if attempted_git or preparation.get("status") != "ready":
            for case in pending_git:
                report["results"][case[0]] = {
                    "tool": case[1], "verdict": "blocked", "reconciliation_required": True,
                    "reason": "Reconcile the previously attempted Git fixture before new mutations.",
                }
            save()
    admin_ready = atlassian_admin_configured()
    unset_families = host_unset_provider_families()
    scope = profile = None
    if requested_profile is not None:
        scope = AcceptanceProfileScope(client, **requested_profile)
        try:
            entered = scope.__enter__()
        except Exception:
            client.close()
            raise
        profile, original = entered.fingerprint, entered.original
        if report.setdefault("inference_profile", profile) != profile:
            scope.restore()
            client.close()
            raise ValueError("The inference profile differs from the one this report was started with; use a new report")
        # Identifiers only, so an unconfirmed restoration can be reconciled by hand.
        if original is not None:
            report.setdefault("original_selection", {"selected": original.selected, "roles": original.roles})
        save()
    try:
        for identity, tool, prompt, approve in selected:
            # Resume continues only unattempted cases. Never replay a recorded
            # failure/unknown side effect by creating another independent job.
            if identity in report["results"]:
                continue
            if tool in ATLASSIAN_ADMIN_TOOLS and not admin_ready:
                report["results"][identity] = {"tool": tool, "verdict": "blocked",
                                               "reason": "ATLASSIAN_ADMIN_API_KEY is not configured"}
                save()
                print(f"BLOCK {identity:<25} needs ATLASSIAN_ADMIN_API_KEY", flush=True)
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
                if scope is not None:
                    # Every case must run on the chosen profile; a changed selection stops the run.
                    scope.assert_unchanged()
                chat = client.create_chat(f"Tool matrix {RUN}: {identity}")
                row["chat_id"] = chat
                row["verdict"] = "blocked"
                row["reason"] = "Submission/execution may be in progress; reconcile before another attempt."
                report["results"][identity] = row
                save()
                job = client.create_job(chat_id=chat, message=message)
                row["job_id"] = job["job_id"]
                save()
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
                if profile is not None:
                    row["inference"] = inference_evidence(
                        job_inference_metrics(client, job["job_id"], terminal.get("attempts")), profile)
                limited = plan_limited(worker)
                unset = provider_not_configured(tool, terminal, unset_families)
                if limited:
                    row["verdict"] = "plan_limited"
                    row["reason"] = limited
                elif unset:
                    row["verdict"] = "not_configured"
                    row["reason"] = unset
                elif terminal.get("status") not in TERMINAL_OK or not ok:
                    raise AcceptanceFailure(f"status={terminal.get('status')} {detail}")
                else:
                    row["verdict"] = "pass"
                    row.pop("reason", None)
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
        if scope is not None:
            # Put the account's own selection back; a restore that cannot be confirmed fails the run.
            report["inference_restoration"] = scope.restore()
        client.close()
        archived = report["results"].get("jira-project-archive", {})
        if archived.get("verdict") == "pass":
            report["cleanup_required"] = {"archived_project": f"MY{RUN[-4:]}",
                                          "action": "Reconcile and clean up this disposable project explicitly."}
        from tools.registry import TOOLS
        report["summary"] = acceptance_summary(report["results"], selected,
                                                registered_tools=TOOLS, full_scope=args.only is None,
                                                source_unchanged=source_identity() == report["source_identity"])
        if scope is not None and report["inference_restoration"].get("status") != "restored":
            report["summary"].update(overall="FAIL", exit_code=1,
                                     reason="The account's inference selection could not be restored; reconcile it")
        save()
        print("summary:", report["summary"])
    return report["summary"]["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
