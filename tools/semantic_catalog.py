from __future__ import annotations

from typing import Any


SEMANTIC_FIELDS = (
    "resource_type",
    "operation_kind",
    "effect",
    "permission",
)


CAPABILITY_SEMANTICS: dict[str, dict[str, str]] = {
    "network_system_info": {"resource_type": "network.device", "operation_kind": "read", "effect": "read", "permission": "network.device.read"},
    "network_interface_status": {"resource_type": "network.interface", "operation_kind": "read", "effect": "read", "permission": "network.interface.read"},
    "network_firewall_config": {"resource_type": "network.firewall", "operation_kind": "read", "effect": "read", "permission": "network.firewall.read"},
    "account_create": {"resource_type": "account", "operation_kind": "create", "effect": "mutation", "permission": "account.create"},
    "account_status": {"resource_type": "account", "operation_kind": "read", "effect": "read", "permission": "account.read"},
    "asset_assign": {"resource_type": "asset", "operation_kind": "action", "effect": "mutation", "permission": "asset.assign"},
    "asset_get": {"resource_type": "asset", "operation_kind": "read", "effect": "read", "permission": "asset.read"},
    "asset_search": {"resource_type": "asset", "operation_kind": "search", "effect": "read", "permission": "asset.search"},
    "asset_unassign": {"resource_type": "asset", "operation_kind": "action", "effect": "mutation", "permission": "asset.unassign"},
    "atlassian_api_token_metadata": {"resource_type": "atlassian.api_token", "operation_kind": "read", "effect": "read", "permission": "atlassian.api_token.read"},
    "atlassian_credential_status": {"resource_type": "atlassian.credentials", "operation_kind": "read", "effect": "read", "permission": "atlassian.credentials.read"},
    "atlassian_org_get": {"resource_type": "atlassian.organization", "operation_kind": "read", "effect": "read", "permission": "atlassian.organization.read"},
    "atlassian_org_list": {"resource_type": "atlassian.organization", "operation_kind": "search", "effect": "read", "permission": "atlassian.organization.search"},
    "atlassian_user_invite": {"resource_type": "atlassian.user_access", "operation_kind": "action", "effect": "mutation", "permission": "atlassian.user_access.invite"},
    "atlassian_workspace_list": {"resource_type": "atlassian.workspace", "operation_kind": "search", "effect": "read", "permission": "atlassian.workspace.search"},
    "check_access": {"resource_type": "access", "operation_kind": "read", "effect": "read", "permission": "access.read"},
    "disable_user": {"resource_type": "account", "operation_kind": "action", "effect": "mutation", "permission": "account.disable"},
    "enable_user": {"resource_type": "account", "operation_kind": "action", "effect": "mutation", "permission": "account.enable"},
    "grant_access": {"resource_type": "access", "operation_kind": "action", "effect": "mutation", "permission": "access.grant"},
    "jira_project_archive": {"resource_type": "jira.project", "operation_kind": "action", "effect": "mutation", "permission": "jira.project.archive"},
    "jira_project_create": {"resource_type": "jira.project", "operation_kind": "create", "effect": "mutation", "permission": "jira.project.create"},
    "jira_project_delete": {"resource_type": "jira.project", "operation_kind": "delete", "effect": "mutation", "permission": "jira.project.delete"},
    "jira_project_get": {"resource_type": "jira.project", "operation_kind": "read", "effect": "read", "permission": "jira.project.read"},
    "jira_project_list": {"resource_type": "jira.project", "operation_kind": "search", "effect": "read", "permission": "jira.project.search"},
    "jira_project_update": {"resource_type": "jira.project", "operation_kind": "update", "effect": "mutation", "permission": "jira.project.update"},
    "knowledge_get": {"resource_type": "knowledge", "operation_kind": "read", "effect": "read", "permission": "knowledge.read"},
    "knowledge_search": {"resource_type": "knowledge", "operation_kind": "search", "effect": "read", "permission": "knowledge.search"},
    "palo_alto_system_info": {"resource_type": "network.firewall", "operation_kind": "read", "effect": "read", "permission": "network.firewall.read"},
    "process_exec": {"resource_type": "workspace.process", "operation_kind": "action", "effect": "mutation", "permission": "workspace.process.execute"},
    "reset_password": {"resource_type": "account", "operation_kind": "action", "effect": "mutation", "permission": "account.reset_password"},
    "revoke_access": {"resource_type": "access", "operation_kind": "action", "effect": "mutation", "permission": "access.revoke"},
    "runbook_get": {"resource_type": "runbook", "operation_kind": "read", "effect": "read", "permission": "runbook.read"},
    "ticket_add_comment": {"resource_type": "ticket.comment", "operation_kind": "create", "effect": "mutation", "permission": "ticket.comment.create"},
    "ticket_assign": {"resource_type": "ticket", "operation_kind": "action", "effect": "mutation", "permission": "ticket.assign"},
    "ticket_comments": {"resource_type": "ticket.comment", "operation_kind": "read", "effect": "read", "permission": "ticket.comment.read"},
    "ticket_create": {"resource_type": "ticket", "operation_kind": "create", "effect": "mutation", "permission": "ticket.create"},
    "ticket_get": {"resource_type": "ticket", "operation_kind": "read", "effect": "read", "permission": "ticket.read"},
    "ticket_history": {"resource_type": "ticket.history", "operation_kind": "read", "effect": "read", "permission": "ticket.history.read"},
    "ticket_search": {"resource_type": "ticket", "operation_kind": "search", "effect": "read", "permission": "ticket.search"},
    "ticket_transition": {"resource_type": "ticket", "operation_kind": "action", "effect": "mutation", "permission": "ticket.transition"},
    "unlock_user": {"resource_type": "account", "operation_kind": "action", "effect": "mutation", "permission": "account.unlock"},
    "workspace_file_info": {"resource_type": "workspace.file", "operation_kind": "read", "effect": "read", "permission": "workspace.file.read"},
    "workspace_git_branches": {"resource_type": "workspace.git", "operation_kind": "search", "effect": "read", "permission": "workspace.git.read"},
    "workspace_git_changed_files": {"resource_type": "workspace.git", "operation_kind": "search", "effect": "read", "permission": "workspace.git.read"},
    "workspace_git_commit": {"resource_type": "workspace.git", "operation_kind": "action", "effect": "mutation", "permission": "workspace.git.commit"},
    "workspace_git_create_branch": {"resource_type": "workspace.git", "operation_kind": "create", "effect": "mutation", "permission": "workspace.git.branch.create"},
    "workspace_git_diff": {"resource_type": "workspace.git", "operation_kind": "read", "effect": "read", "permission": "workspace.git.read"},
    "workspace_git_log": {"resource_type": "workspace.git", "operation_kind": "search", "effect": "read", "permission": "workspace.git.read"},
    "workspace_git_push": {"resource_type": "workspace.git", "operation_kind": "action", "effect": "mutation", "permission": "workspace.git.push"},
    "workspace_git_stage_all": {"resource_type": "workspace.git", "operation_kind": "action", "effect": "mutation", "permission": "workspace.git.stage"},
    "workspace_git_stage_files": {"resource_type": "workspace.git", "operation_kind": "action", "effect": "mutation", "permission": "workspace.git.stage"},
    "workspace_git_staged_diff": {"resource_type": "workspace.git", "operation_kind": "read", "effect": "read", "permission": "workspace.git.read"},
    "workspace_git_status": {"resource_type": "workspace.git", "operation_kind": "read", "effect": "read", "permission": "workspace.git.read"},
    "workspace_git_switch_branch": {"resource_type": "workspace.git", "operation_kind": "action", "effect": "mutation", "permission": "workspace.git.branch.switch"},
    "workspace_git_unstage_files": {"resource_type": "workspace.git", "operation_kind": "action", "effect": "mutation", "permission": "workspace.git.unstage"},
    "workspace_list": {"resource_type": "workspace.directory", "operation_kind": "search", "effect": "read", "permission": "workspace.directory.search"},
    "workspace_mkdir": {"resource_type": "workspace.directory", "operation_kind": "create", "effect": "mutation", "permission": "workspace.directory.create"},
    "workspace_process_snapshot": {"resource_type": "workspace.process", "operation_kind": "read", "effect": "read", "permission": "workspace.process.read"},
    "workspace_project_info": {"resource_type": "workspace.project", "operation_kind": "read", "effect": "read", "permission": "workspace.project.read"},
    "workspace_read_text": {"resource_type": "workspace.file", "operation_kind": "read", "effect": "read", "permission": "workspace.file.read"},
    "workspace_run_build": {"resource_type": "workspace.build", "operation_kind": "action", "effect": "mutation", "permission": "workspace.build.run"},
    "workspace_run_tests": {"resource_type": "workspace.test", "operation_kind": "action", "effect": "mutation", "permission": "workspace.test.run"},
    "workspace_search": {"resource_type": "workspace.file", "operation_kind": "search", "effect": "read", "permission": "workspace.file.search"},
    "workspace_service_logs": {"resource_type": "workspace.service", "operation_kind": "read", "effect": "read", "permission": "workspace.service.read"},
    "workspace_service_status": {"resource_type": "workspace.service", "operation_kind": "read", "effect": "read", "permission": "workspace.service.read"},
}


def apply_capability_semantics(
    tools: dict[str, dict[str, Any]],
) -> None:
    tool_names = set(tools)
    semantic_names = set(CAPABILITY_SEMANTICS)

    missing = sorted(tool_names - semantic_names)
    unknown = sorted(semantic_names - tool_names)

    if missing or unknown:
        raise ValueError(
            "Capability semantic catalog coverage mismatch; "
            f"missing={missing} unknown={unknown}"
        )

    for tool_name in sorted(tool_names):
        tool = tools[tool_name]
        semantic = CAPABILITY_SEMANTICS[tool_name]

        if not isinstance(tool, dict):
            raise ValueError(
                f"Capability '{tool_name}' metadata must be a mutable dict."
            )

        for field in SEMANTIC_FIELDS:
            expected = semantic[field].strip().lower()
            existing = tool.get(field)

            if isinstance(existing, str) and existing.strip():
                normalized = existing.strip().lower()
                if normalized != expected:
                    raise ValueError(
                        f"Capability '{tool_name}' {field} disagrees with "
                        f"the explicit semantic catalog; "
                        f"catalog={normalized!r} semantic={expected!r}"
                    )

            tool[field] = expected
