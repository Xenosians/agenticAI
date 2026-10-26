"""Setup readiness for each provider family behind the capability catalog.

SRS 1.9 FR-081 and NFR-036: the catalog shows whether each operation's
provider is set up, from the same configuration the provider factories use.
Readiness is configuration only. It never contacts a provider, so it cannot
claim that a provider is reachable, healthy or that an action succeeded.

States:
    configured      every setting the real provider needs is present
    simulated       a mock provider was explicitly selected (demo/test)
    not_configured  a required setting is missing; ``missing`` names it

Setting names are reported, never their values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from tools.access.catalog import ACCESS_TOOLS
from tools.account.catalog import ACCOUNT_TOOLS
from tools.assets.catalog import ASSET_TOOLS
from tools.atlassian.catalog import ATLASSIAN_TOOLS
from tools.developer.catalog import DEVELOPER_TOOLS
from tools.directory.catalog import DIRECTORY_TOOLS
from tools.git.catalog import GIT_TOOLS
from tools.jira.catalog import JIRA_TOOLS
from tools.knowledge.catalog import KNOWLEDGE_TOOLS
from tools.mailbox.catalog import MAILBOX_TOOLS
from tools.networking.catalog import NETWORKING_TOOLS
from tools.operations.catalog import OPERATIONS_TOOLS
from tools.registry import TOOLS
from tools.servicedesk.catalog import SERVICE_DESK_TOOLS
from tools.ticketing.catalog import TICKETING_TOOLS


def _present(settings: Any, name: str) -> bool:
    value = getattr(settings, name, None)
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def _missing(settings: Any, *names: str) -> list[str]:
    return [name.upper() for name in names if not _present(settings, name)]


def _selected_backend(settings: Any, field: str, real: str, requirements: Callable[[Any], list[str]]):
    """(state, missing) for a provider chosen by an explicit *_BACKEND setting."""
    backend = (getattr(settings, field, None) or "unconfigured").strip().lower()
    if backend == "mock":
        return "simulated", []
    if backend != real:
        return "not_configured", [field.upper()]
    missing = requirements(settings)
    return ("not_configured" if missing else "configured"), missing


def _directory(settings):
    def requirements(settings):
        missing = _missing(settings, "ad_host")
        if not (_present(settings, "ad_bind_user") or _present(settings, "ad_bind_dn")):
            missing.append("AD_BIND_USER or AD_BIND_DN")
        return missing + _missing(settings, "ad_bind_password", "ad_base_dn")
    return _selected_backend(settings, "directory_backend", "ldap", requirements)


def _jira_credentials(settings):
    return _missing(settings, "jira_base_url", "jira_email", "jira_api_token")


def _ticketing(settings):
    return _selected_backend(settings, "ticketing_backend", "jira", _jira_credentials)


def _service_desk(settings):
    # Queues and SLAs exist only on real Jira Service Management.
    if (getattr(settings, "ticketing_backend", None) or "").strip().lower() != "jira":
        return "not_configured", ["TICKETING_BACKEND=jira"]
    missing = _jira_credentials(settings)
    return ("not_configured" if missing else "configured"), missing


def _jira_admin(settings):
    missing = _jira_credentials(settings)
    return ("not_configured" if missing else "configured"), missing


def _atlassian_admin(settings):
    missing = _missing(settings, "atlassian_admin_api_key", "atlassian_org_id")
    return ("not_configured" if missing else "configured"), missing


def _assets(settings):
    return _selected_backend(settings, "asset_backend", "sqlite",
                             lambda settings: _missing(settings, "asset_sqlite_path"))


def _knowledge(settings):
    return _selected_backend(settings, "knowledge_backend", "local",
                             lambda settings: _missing(settings, "knowledge_snapshot_path",
                                                       "knowledge_snapshot_sha256"))


def _openwrt(settings):
    missing = _missing(settings, "openwrt_base_url", "openwrt_username", "openwrt_password")
    return ("not_configured" if missing else "configured"), missing


def _mailbox_lab(settings):
    missing = [] if getattr(settings, "mailbox_lab_enabled", False) is True else ["MAILBOX_LAB_ENABLED"]
    return ("not_configured" if missing else "configured"), missing


def _operations(settings):
    # The same targets the operations tools read: an allowlist of containers,
    # or health endpoints (explicit, or derived from the backend and AI addresses).
    from services.operations import health_targets

    if _present(settings, "operations_containers") or health_targets(settings):
        return "configured", []
    return "not_configured", ["OPERATIONS_CONTAINERS or OPERATIONS_HEALTH_URLS"]


def _workspace(_settings):
    # Repository and process work runs inside the configured workspace root.
    return "configured", []


@dataclass(frozen=True)
class ProviderFamily:
    id: str
    name: str
    kind: str
    readiness: Callable[[Any], tuple[str, list[str]]]
    capabilities: frozenset[str]


_DIRECTORY_INLINE = {"account_status", "check_access", "unlock_user"}
_WORKSPACE_INLINE = {"process_exec", "workspace_mkdir", "workspace_read_text", "workspace_list",
                     "workspace_search", "workspace_file_info"}

PROVIDER_FAMILIES: tuple[ProviderFamily, ...] = (
    ProviderFamily("directory", "Directory (LDAP)", "provider", _directory, frozenset(
        (set(ACCOUNT_TOOLS) - set(MAILBOX_TOOLS)) | set(ACCESS_TOOLS) | set(DIRECTORY_TOOLS)
        | _DIRECTORY_INLINE)),
    ProviderFamily("mailbox_lab", "Lab mailboxes", "provider", _mailbox_lab, frozenset(MAILBOX_TOOLS)),
    ProviderFamily("ticketing", "Ticketing (Jira)", "provider", _ticketing, frozenset(TICKETING_TOOLS)),
    ProviderFamily("service_desk", "Service desk (Jira Service Management)", "provider", _service_desk,
                   frozenset(SERVICE_DESK_TOOLS)),
    ProviderFamily("jira_admin", "Jira project administration", "provider", _jira_admin,
                   frozenset(JIRA_TOOLS)),
    ProviderFamily("atlassian_admin", "Atlassian administration", "provider", _atlassian_admin,
                   frozenset(ATLASSIAN_TOOLS)),
    ProviderFamily("assets", "Asset inventory", "provider", _assets, frozenset(ASSET_TOOLS)),
    ProviderFamily("knowledge", "Knowledge and runbooks", "provider", _knowledge, frozenset(KNOWLEDGE_TOOLS)),
    ProviderFamily("openwrt", "OpenWrt", "provider", _openwrt, frozenset(NETWORKING_TOOLS)),
    ProviderFamily("operations", "Platform operations", "provider", _operations, frozenset(OPERATIONS_TOOLS)),
    ProviderFamily("workspace", "Workspace and Git", "local", _workspace, frozenset(
        set(DEVELOPER_TOOLS) | set(GIT_TOOLS) | _WORKSPACE_INLINE)),
)

_PROVIDER_BY_CAPABILITY = {
    capability: family.id for family in PROVIDER_FAMILIES for capability in family.capabilities
}


def provider_for_capability(name: str) -> str | None:
    """The provider family whose setup a capability depends on."""
    return _PROVIDER_BY_CAPABILITY.get(name)


def _mode(capabilities: frozenset[str]) -> str:
    changes = any(TOOLS[name].get("risk") != "read" for name in capabilities if name in TOOLS)
    return "read_write" if changes else "read_only"


def integration_readiness(settings: Any) -> list[dict]:
    """One row per provider family: setup state, missing setting names and its capabilities."""
    rows = []
    for family in PROVIDER_FAMILIES:
        state, missing = family.readiness(settings)
        rows.append({
            "id": family.id,
            "name": family.name,
            "kind": family.kind,
            "state": state,
            # Kept for existing integration-status readers: true only for a
            # fully set-up real provider, never for a simulated one.
            "configured": state == "configured",
            "missing": missing,
            "capabilities": sorted(family.capabilities),
            "mode": _mode(family.capabilities),
        })
    return rows
